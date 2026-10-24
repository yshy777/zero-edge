from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from utils.side_channel import load_side_channel
from .graph_sampler import *
import pdb


# lmdb >= 2.x refuses to open the same path twice in one process (even with
# identical readonly config), so cache the readonly env and share it across
# SubgraphDataset instances (e.g. the train and valid sets).
_lmdb_env_cache = {}


def _get_readonly_env(db_path):
    if db_path not in _lmdb_env_cache:
        _lmdb_env_cache[db_path] = lmdb.open(db_path, readonly=True, max_dbs=5, lock=False)
    return _lmdb_env_cache[db_path]


def load_product_fingerprint(fp_path, id2entity):
    """Return (entity_fingerprint [n_entities, D], D) from a build_kg product
    fingerprint JSON ({cve: [float, ...]}). Returns (None, 0) when the file is
    absent/empty. Non-CVE entities (CWE/CAPEC/technique/tactic) get zero vectors;
    CVEs without NVD CPE data also fall back to zero (they have no entry)."""
    if not os.path.exists(fp_path):
        return None, 0
    with open(fp_path, encoding='utf-8') as f:
        fp = json.load(f)
    if not fp:
        return None, 0
    D = len(next(iter(fp.values())))
    n_ent = max(id2entity) + 1  # entity ids are 0..n_ent-1
    arr = np.zeros((n_ent, D), dtype=np.float32)
    for eid, name in id2entity.items():
        if name in fp:
            arr[eid] = np.asarray(fp[name], dtype=np.float32)
    return arr, D


def build_cwe2id(id2entity):
    """CWE entity name -> embedding index (1..n_cwes), sorted-name order.

    Index 0 is reserved for non-CWE / OOV entities. Keyed by entity *string* so
    the eval scripts can remap against the model's saved cwe2id (the CWE answer
    entities are seen at training time; only the CVE query is unseen)."""
    return {name: i + 1 for i, name in enumerate(
        sorted(n for n in id2entity.values() if n.startswith('CWE-')))}


def build_cwe_index(id2entity, cwe2id):
    """global entity id -> cwe embedding index (0 = non-CWE/OOV), by name."""
    n_ent = max(id2entity) + 1
    arr = np.zeros(n_ent, dtype=np.int64)
    for eid, name in id2entity.items():
        arr[eid] = cwe2id.get(name, 0)
    return arr


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, saved_relation2id)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))
    data_path = os.path.join(params.main_dir, f'../data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        print(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, '../data/{}/'.format(params.dataset))
        save_to_file(directory, f'negative_test_triples_by_{params.model}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)


# def get_kge_embeddings(dataset, kge_model):
#
#     path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
#     node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
#     with open(os.path.join(path, 'id2entity.json')) as json_file:
#         kge_id2entity = json.load(json_file)
#         kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}
#
#     return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, cluster_assign=None, cwe2id=None):

        self.main_env = _get_readonly_env(db_path)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.num_neg_samples_per_link = num_neg_samples_per_link

        if cluster_assign is not None:
            self.id2cluster, self.id2soft, self.n_clusters = cluster_assign
            self.has_cluster = True
        else:
            self.id2cluster = self.id2soft = None
            self.n_clusters = 0
            self.has_cluster = False

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)

        # Scheme 3: graph-external side channel (C1). Generic loader: reads
        # side_channel.npz (+ side_channel_entities.txt) when present, falls
        # back to the legacy product_fingerprint.json, else (None, 0) and the
        # model skips injection.
        self.entity_fingerprint, self.prod_dim = load_side_channel(
            os.path.join(os.path.dirname(db_path), 'side_channel.npz'),
            id2entity)

        # Per-CWE tail embedding (seen answer entity) index, keyed by name so the
        # eval scripts can remap against the model's saved cwe2id. When the caller
        # supplies the model's saved cwe2id (AUC path), use it verbatim — a local
        # rebuild over an eval-time entity set that contains unseen CWE names
        # would lexicographically shift every index after the insertion point and
        # silently misalign the tail embeddings (P1 fix).
        if cwe2id is None:
            cwe2id = build_cwe2id(id2entity)
        self.cwe2id = cwe2id
        self.n_cwes = len(self.cwe2id)
        self.cwe_index = build_cwe_index(id2entity, self.cwe2id)

        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph


        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            # self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            # self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            # self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            # self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))
            #
            # self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            # self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            # self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            # self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))
            #
            # self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            # self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            # self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            # self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        print(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")


        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            en_nodes_pos, r_label_pos, g_label_pos, en_n_labels_pos, dis_nodes_pos, dis_n_labels_pos = deserialize(txn.get(str_id)).values()
            en_subgraph_pos = self._prepare_subgraphs(en_nodes_pos, r_label_pos, en_n_labels_pos)
            dis_subgraph_pos = self._prepare_subgraphs(dis_nodes_pos, r_label_pos, dis_n_labels_pos)
        en_subgraphs_neg = []
        dis_subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                en_nodes_neg, r_label_neg, g_label_neg, en_n_labels_neg, dis_nodes_neg, dis_n_labels_neg = deserialize(txn.get(str_id)).values()
                en_subgraphs_neg.append(self._prepare_subgraphs(en_nodes_neg, r_label_neg, en_n_labels_neg))
                dis_subgraphs_neg.append(self._prepare_subgraphs(dis_nodes_neg, r_label_neg, dis_n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return en_subgraph_pos, dis_subgraph_pos, g_label_pos, r_label_pos, en_subgraphs_neg, dis_subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.node_subgraph(self.graph, nodes)
        edge_type = self.graph.edata['type'][subgraph.edata[dgl.EID]].clone()

        u, v = subgraph.edges()

        # All edges between the two roots (node 0 -> node 1), regardless of type.
        edges_btw_roots = ((u == 0) & (v == 1)).nonzero().squeeze(1)

        # Does at least one root-to-root edge carry the target relation?
        rel_link = (edge_type[edges_btw_roots] == r_label).nonzero()

        if rel_link.nelement() == 0:
            # No (0,1) edge with the target relation -> append a synthetic one.
            old_eids = subgraph.edata[dgl.EID].clone()
            u = torch.cat([u, torch.tensor([0], dtype=u.dtype)])
            v = torch.cat([v, torch.tensor([1], dtype=v.dtype)])
            edge_type = torch.cat([edge_type, torch.tensor([r_label], dtype=torch.long)])
            subgraph = dgl.graph((u, v), num_nodes=subgraph.num_nodes())
            # Rebuilt from scratch -> re-attach the parent node/edge ids so the
            # schema stays uniform with node_subgraph outputs in the same batch
            # (dgl.batch rejects mixed schemas). Fires for self-loop triples
            # (head==tail: the self-loop collapses onto one node copy, so the
            # (0,1) pair never appears — WN18RR train carries 7 such triples)
            # and for negatives. The synthetic edge has no parent id (-1).
            subgraph.ndata[dgl.NID] = torch.LongTensor(nodes)
            subgraph.edata[dgl.EID] = torch.cat(
                [old_eids, torch.tensor([-1], dtype=old_eids.dtype)])
            e_ids = np.zeros(subgraph.num_edges())
            e_ids[-1] = 1  # only the synthetic edge is the target
        else:
            e_ids = np.zeros(subgraph.num_edges())
            e_ids[edges_btw_roots.numpy()] = 1  # ALL (0,1) edges are targets

        subgraph.edata['type'] = edge_type
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.num_edges()), dtype=torch.long)
        subgraph.edata['id'] = torch.FloatTensor(e_ids)

        subgraph = self._prepare_features_new(subgraph, n_labels, nodes)


        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.num_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, nodes=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.num_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1

        n_feats = label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        # Scheme 3: per-node product-family fingerprint (zero for non-CVE nodes).
        if self.entity_fingerprint is not None and nodes is not None:
            subgraph.ndata['prod'] = torch.FloatTensor(self.entity_fingerprint[nodes])

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        # ProtoRMPI: attach per-node cluster assignment (hard label + soft vector)
        # so the model can read head/tail prototypes directly from the subgraph.
        # `nodes` holds the global entity IDs in subgraph-node order (node 0 = head,
        # node 1 = tail). Entities unseen during clustering -> label -1 / zero soft.
        if self.has_cluster and nodes is not None:
            n_ent = len(self.id2cluster)
            clusters = [int(self.id2cluster[e]) if e < n_ent else -1 for e in nodes]
            subgraph.ndata['cluster'] = torch.LongTensor(clusters)
            soft = np.stack([
                self.id2soft[e] if e < n_ent else np.zeros(self.n_clusters, dtype=np.float32)
                for e in nodes
            ])
            subgraph.ndata['soft'] = torch.FloatTensor(soft)

        # Per-CWE tail embedding index (0 = non-CWE/OOV), read by the model's
        # proto branch to break the coarse-cluster tie over the CWE candidates.
        if getattr(self, 'cwe_index', None) is not None and nodes is not None:
            subgraph.ndata['cwe'] = torch.LongTensor(self.cwe_index[nodes])

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
