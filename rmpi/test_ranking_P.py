import os
import random
import argparse
import logging
import json
import time
from collections import OrderedDict

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl

from utils.cluster_assign import (
    load_cluster_assignments, load_cluster_meta,
    graph_degrees, structural_feature_from_graph, soft_assign_feature,
    precompute_structural_features, soft_assign_features_batch,
)
from subgraph_extraction.datasets import build_cwe_index


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        # if file_type == 'links':
        #    data = random.sample(data, 20)
        triplets[file_type] = np.array(data)


    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation


cluster_assign_ = None
cluster_meta_ = None
degrees_ = None
entity_fingerprint_ = None
cwe_index_ = None
A_incidence_ = None  # 预计算的 incidence matrix（只依赖 adj_list，见 intialize_worker）
_bfs_cache_ = None  # per-node 2-hop 邻域缓存（LRU，见 intialize_worker）
_BFS_CACHE_MAXSIZE = 3000  # LRU 上限（条目数），可由 --cache_size 覆盖
struct_feat_ = None  # 预计算的结构特征矩阵 [n_nodes, 6+num_relations]，仅 proto_hard 归纳路径
soft_assign_ = None  # 预计算的软指派 [n_nodes, n_clusters]，仅 proto_hard 归纳路径


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, cluster_assign, cluster_meta, degrees, entity_fingerprint, cwe_index):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, cluster_assign_, cluster_meta_, degrees_, entity_fingerprint_, cwe_index_, A_incidence_, _bfs_cache_, _BFS_CACHE_MAXSIZE, struct_feat_, soft_assign_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, cluster_assign_, cluster_meta_, degrees_, entity_fingerprint_, cwe_index_ = (
        model, adj_list, dgl_adj_list, id2entity, params, cluster_assign, cluster_meta, degrees, entity_fingerprint, cwe_index
    )
    # 预计算 incidence matrix（只依赖 adj_list，与具体 link 无关）。原实现每个子图都重建一次
    # （148k 实体图每次 ~200 万边、每个 test link ×101 子图），是 full_ind 排名 ~9.5s/link 的主因。
    A_incidence_ = incidence_matrix(adj_list)
    A_incidence_ = A_incidence_ + A_incidence_.T
    # 同一节点（固定 head/tail + 重复出现的随机负样本 + 跨 link 共享的 CWE hub）
    # 的 2-hop BFS 会被反复重算（每个 test link ×51 子图）。BFS 确定性（与 RNG 无关），
    # 故按节点缓存，把 BFS 从「每子图重算」降到「每节点一次」。用 LRU 上限封顶内存，
    # 避免 full_ind 全量排名（49535 link）时 hub 的巨型 2-hop 邻域把 worker 内存撑爆。
    _bfs_cache_ = OrderedDict()
    _BFS_CACHE_MAXSIZE = int(getattr(params, 'cache_size', 3000))
    # proto_hard 归纳路径：把每个节点 O(E) 的结构特征（dgl_graph.edges() per node）
    # 一次性向量化预计算，避免每子图每节点重算（否则 proto_hard 排名 ~100s/link）。
    if cluster_meta_ is not None and degrees_ is not None:
        struct_feat_ = precompute_structural_features(
            dgl_adj_list_, int(cluster_meta_['num_relations']), degrees_)
        # 软指派也只依赖节点本身（struct_feat_[e] + centroids + temp），同样一次性
        # 向量化预计算成 [n_nodes, n_clusters]，attach_cluster 直接查表。否则每 link
        # 对 ~102 个子图 × 每节点做 Python 循环 soft_assign（hub 子图 ~26k 节点 → ~2s/link）。
        soft_assign_ = soft_assign_features_batch(
            struct_feat_, cluster_meta_['centroids'], float(cluster_meta_['temperature']))
    else:
        struct_feat_ = None
        soft_assign_ = None
    # Eval mode: disable dropout (the saved model is in train mode; the
    # validation evaluator uses .eval(), so ranking must too) and disable
    # autograd (no graph built during forward -> less memory).
    model_.eval()
    torch.set_grad_enabled(False)


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50,
                                        tail_pools=None, head_pools=None,
                                        known_pos=None):
    """Sampled negative triples, same format as the original protocol.

    tail_pools/head_pools: dict rel_id -> np.array of candidate entity ids, or
    None = full entity space. known_pos: set of (h, rel, t) id triples to
    exclude (Bordes filtered semantics) on top of the train-adjacency check.
    C6: the --seed argument is wired upstream (np.random.seed), so the draws
    are bit-exact reproducible."""

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]
    all_ids = np.arange(n)

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        rel_i = int(rel)
        tail_pool = tail_pools.get(rel_i) if tail_pools is not None else None
        head_pool = head_pools.get(rel_i) if head_pools is not None else None
        tail_pool = all_ids if tail_pool is None else tail_pool
        head_pool = all_ids if head_pool is None else head_pool

        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        max_tries = max(10000, num_samples * 100)
        tries = 0
        while len(neg_triplet['head'][0]) < num_samples:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f'negative sampling stuck (head dir, link {i})')
            neg_head = head
            neg_tail = int(np.random.choice(tail_pool))
            if neg_head != neg_tail and adj_list[rel_i][neg_head, neg_tail] == 0 \
                    and (known_pos is None or (neg_head, neg_tail, rel_i) not in known_pos):
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        tries = 0
        while len(neg_triplet['tail'][0]) < num_samples:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f'negative sampling stuck (tail dir, link {i})')
            neg_head = int(np.random.choice(head_pool))
            neg_tail = tail
            if neg_head != neg_tail and adj_list[rel_i][neg_head, neg_tail] == 0 \
                    and (known_pos is None or (neg_head, neg_tail, rel_i) not in known_pos):
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


# def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
#     with open(ruleN_pred_path) as f:
#         pred_data = [line.split() for line in f.read().split('\n')[:-1]]
#
#     neg_triplets = []
#     for i in range(len(pred_data) // 3):
#         neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
#         if pred_data[3 * i][1] in saved_relation2id:
#             head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
#             for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
#                 neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
#                 if entity2id[new_head] == head:
#                     neg_triplet['head'][1] = j
#             for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
#                 neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
#                 if entity2id[new_tail] == tail:
#                     neg_triplet['tail'][1] = j
#
#             neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
#             neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])
#
#             neg_triplets.append(neg_triplet)
#
#     return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def _cached_neighbors(node, adj, h):
    """Per-node 2-hop neighbourhood with an LRU-bounded memoization.

    get_neighbor_nodes is deterministic (BFS, no RNG when max_nodes_per_hop is
    None), and ranking re-evaluates the same node's neighbourhood once per neg
    link (51x per test link) and again across links/runs, so caching collapses
    that to once per node. The cache is capped at _BFS_CACHE_MAXSIZE entries
    (LRU eviction) so the hub-heavy full-ind graph cannot blow worker memory —
    the hot CWE hubs stay resident while cold nodes are evicted.
    """
    cache = _bfs_cache_
    if cache is None:  # 非 worker 上下文（profile 等），退回直接计算
        return get_neighbor_nodes(set([node]), adj, h, None)
    v = cache.get(node)
    if v is not None:
        cache.move_to_end(node)
        return v
    v = get_neighbor_nodes(set([node]), adj, h, None)
    cache[node] = v
    if len(cache) > _BFS_CACHE_MAXSIZE:
        cache.popitem(last=False)  # evict LRU
    return v


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None, need_disclosing=True):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = A_incidence_
    if A_incidence is None:  # 防御：非 worker 上下文（正常不会走到）
        A_incidence = incidence_matrix(A_list)
        A_incidence = A_incidence + A_incidence.T

    # could pack these two into a function
    root1_nei = _cached_neighbors(ind[0], A_incidence, h)
    root2_nei = _cached_neighbors(ind[1], A_incidence, h)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    # if enclosing_sub_graph:
    enclosing_subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)

    enclosing_subgraph = [adj[enclosing_subgraph_nodes, :][:, enclosing_subgraph_nodes] for adj in A_list]

    # labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    enclosing_labels, enclosing_subgraph_nodes_labeled = node_label_new(incidence_matrix(enclosing_subgraph),
                                                                    max_distance=h, enclosing_flag=True)

    pruned_enclosing_subgraph_nodes = np.array(enclosing_subgraph_nodes)[enclosing_subgraph_nodes_labeled].tolist()
    pruned_enclosing_labels = enclosing_labels[enclosing_subgraph_nodes_labeled]

    if max_node_label_value is not None:
        pruned_enclosing_labels = np.array(
            [np.minimum(label, max_node_label_value).tolist() for label in pruned_enclosing_labels])

    # Disclosing (union) subgraph is only consumed by ablation >= 1 (entity
    # aggregation). For base (GraIL, ablation 0) it is built then discarded by the
    # model, but on hub links (e.g. CWE-79 as a fixed tail) each of the 51 negative
    # samples has a ~25k-node disclosing subgraph, so skipping it removes a
    # ~2.5GB/worker peak (the 4th-OOM root cause). Ablation 0 never touches dis_g.
    if not need_disclosing:
        return pruned_enclosing_subgraph_nodes, pruned_enclosing_labels, None, None

    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    disclosing_subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
    disclosing_subgraph = [adj[disclosing_subgraph_nodes, :][:, disclosing_subgraph_nodes] for adj in A_list]
    disclosing_labels, disclosing_subgraph_nodes_labeled = node_label_new(incidence_matrix(disclosing_subgraph),
                                                                      max_distance=h, enclosing_flag=False)

    pruned_disclosing_subgraph_nodes = np.array(disclosing_subgraph_nodes)[disclosing_subgraph_nodes_labeled].tolist()
    pruned_disclosing_labels = disclosing_labels[disclosing_subgraph_nodes_labeled]

    if max_node_label_value is not None:
        pruned_disclosing_labels = np.array(
            [np.minimum(label, max_node_label_value).tolist() for label in pruned_disclosing_labels])

    return pruned_enclosing_subgraph_nodes, pruned_enclosing_labels, pruned_disclosing_subgraph_nodes, pruned_disclosing_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def bfs_dist(adj, source):
    """Unweighted undirected single-source shortest-path distances, exactly
    equivalent to ssp.csgraph.dijkstra(adj, indices=[source], directed=False,
    unweighted=True, limit=1e6)[0] but far faster on large graphs: the scipy
    dijkstra spends ~500ms per call on a 25k-node hub neighbourhood (the
    full-ind ranking bottleneck), whereas a level-by-level vectorized BFS is
    ~1ms. `adj` is symmetrized internally and converted to CSR for row slicing.
    Returns a float64 (n,) array; unreachable nodes are inf (matching dijkstra).
    """
    adj = (adj + adj.T).tocsr()
    n = adj.shape[0]
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    frontier = np.array([source], dtype=np.int64)
    d = 0
    while frontier.size:
        d += 1
        nbrs = adj[frontier].indices
        if nbrs.size == 0:
            break
        nbrs = nbrs[np.isinf(dist[nbrs])]  # keep only unvisited
        if nbrs.size == 0:
            break
        nbrs = np.unique(nbrs)
        dist[nbrs] = d
        frontier = nbrs
    return dist


def node_label_new(subgraph, max_distance=1, enclosing_flag=False):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    # 无权图上 dijkstra == BFS；用向量化 bfs_dist 替换 ssp.csgraph.dijkstra，
    # 逐元素结果一致（已在 random/star/真实子图上核对），但快约两个数量级。
    dist_to_roots = [np.clip(bfs_dist(sg, 0)[1:], 0, 1e7)[None, :] for sg in sgs_single_root]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    # enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]

    if enclosing_flag:

        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    else:
        # enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) < 1e6)[0]
        # process the unconnected node (neg samples)
        indices_dim0, indices_dim1 = np.where(labels == 1e7)

        indices_dim1_convert = indices_dim1 + 1
        indices_dim1_convert[indices_dim1_convert == 2] = 0
        new_indices = [indices_dim0.tolist(), indices_dim1_convert.tolist()]
        ori_indices = [indices_dim0.tolist(), indices_dim1.tolist()]

        # values = labels[new_indices] + 1
        # labels[ori_indices] = values
        values = labels[tuple(new_indices)] + 1
        labels[tuple(ori_indices)] = values
        # process the unconnected node (neg samples)

        # print(labels)
        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]

    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    src, dst, etypes = [], [], []
    for rel, adj in enumerate(graph):
        adjcoo = adj.tocoo()
        src += adjcoo.row.tolist()
        dst += adjcoo.col.tolist()
        etypes += [rel] * len(adjcoo.row)

    # Build the graph directly from the sparse adjacencies so that parallel
    # edges (same (u, v) under different relation types) are preserved.
    g_dgl = dgl.graph((torch.LongTensor(src), torch.LongTensor(dst)),
                      num_nodes=graph[0].shape[0])
    g_dgl.edata['type'] = torch.LongTensor(etypes)
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.num_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph

def prepare_subgraph(dgl_adj_list, nodes, rel, node_labels, max_node_label_value):
    subgraph = dgl.node_subgraph(dgl_adj_list, nodes)
    edge_type = dgl_adj_list.edata['type'][subgraph.edata[dgl.EID]].clone()

    u, v = subgraph.edges()

    # All edges between the two roots (node 0 -> node 1), regardless of type.
    edges_btw_roots = ((u == 0) & (v == 1)).nonzero().squeeze(1)

    # Does at least one root-to-root edge carry the target relation?
    rel_link = (edge_type[edges_btw_roots] == rel).nonzero()

    if rel_link.nelement() == 0:
        # No (0,1) edge with the target relation -> append a synthetic one.
        u = torch.cat([u, torch.tensor([0], dtype=u.dtype)])
        v = torch.cat([v, torch.tensor([1], dtype=v.dtype)])
        edge_type = torch.cat([edge_type, torch.tensor([rel], dtype=torch.long)])
        subgraph = dgl.graph((u, v), num_nodes=subgraph.num_nodes())
        e_ids = np.zeros(subgraph.num_edges())
        e_ids[-1] = 1  # only the synthetic edge is the target
    else:
        e_ids = np.zeros(subgraph.num_edges())
        e_ids[edges_btw_roots.numpy()] = 1  # ALL (0,1) edges are targets

    subgraph.edata['type'] = edge_type
    subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.num_edges()), dtype=torch.long)
    subgraph.edata['id'] = torch.FloatTensor(e_ids)

    subgraph = prepare_features(subgraph, node_labels, max_node_label_value)
    # Scheme 3: per-node product-family fingerprint (zero for non-CVE nodes).
    if entity_fingerprint_ is not None and nodes is not None:
        subgraph.ndata['prod'] = torch.FloatTensor(entity_fingerprint_[nodes])
    # Per-CWE tail embedding index (0 = non-CWE/OOV).
    if cwe_index_ is not None and nodes is not None:
        subgraph.ndata['cwe'] = torch.LongTensor(cwe_index_[nodes])
    subgraph = attach_cluster(subgraph, nodes)
    return subgraph


def attach_cluster(subgraph, nodes):
    if nodes is None:
        return subgraph
    if cluster_meta_ is not None:
        # Inductive path: compute each entity's structural feature from the eval
        # graph and soft-assign against the training centroids (unseen entities).
        # structural_feature_from_graph is O(E) per node; instead index the
        # precomputed struct_feat_ matrix (built once in intialize_worker). The
        # soft assignment is likewise precomputed as soft_assign_ [n_nodes, n_clusters].
        soft = soft_assign_[nodes].astype(np.float32)
        subgraph.ndata['soft'] = torch.FloatTensor(soft)
        subgraph.ndata['cluster'] = torch.LongTensor(soft.argmax(axis=1))
        return subgraph
    if cluster_assign_ is None:
        return subgraph
    id2cluster, id2soft, n_clusters = cluster_assign_
    n_ent = len(id2cluster)
    clusters = [int(id2cluster[e]) if e < n_ent else -1 for e in nodes]
    subgraph.ndata['cluster'] = torch.LongTensor(clusters)
    soft = np.stack([id2soft[e] if e < n_ent else np.zeros(n_clusters, dtype=np.float32) for e in nodes])
    subgraph.ndata['soft'] = torch.FloatTensor(soft)
    return subgraph

def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, need_disclosing=True):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    en_subgraphs = []
    dis_subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        en_nodes, en_node_labels, dis_nodes, dis_node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params_.enclosing_sub_graph, max_node_label_value=max_node_label_value, need_disclosing=need_disclosing)

        en_subgraph = prepare_subgraph(dgl_adj_list, en_nodes, rel, en_node_labels, max_node_label_value)
        en_subgraphs.append(en_subgraph)
        if need_disclosing:
            dis_subgraph = prepare_subgraph(dgl_adj_list, dis_nodes, rel, dis_node_labels, max_node_label_value)
            dis_subgraphs.append(dis_subgraph)
        r_labels.append(rel)

    batched_en_graph = dgl.batch(en_subgraphs)
    batched_dis_graph = dgl.batch(dis_subgraphs) if need_disclosing else None
    r_labels = torch.LongTensor(r_labels)

    return (batched_en_graph, batched_dis_graph, r_labels)


def get_rank(neg_links):
    # Base (GraIL, ablation 0) forward ignores the disclosing subgraph; ablation >= 1
    # (NE/proto entity aggregation) needs it. Skipping it for base removes the
    # hub-link ~2.5GB/worker peak (see subgraph_extraction_labeling).
    need_disclosing = (getattr(model_.params, 'ablation', 0) >= 1)
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, params_.max_label_value, need_disclosing=need_disclosing)
        head_scores = model_(data).squeeze(1).detach().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, params_.max_label_value, need_disclosing=need_disclosing)
        tail_scores = model_(data).squeeze(1).detach().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def save_negative_triples_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('../data', params.dataset, f'{params.expri_name}_ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('../data', params.dataset, f'{params.expri_name}_ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')






def main(params):
    # C6: --seed was a dead parameter; wire it so the negative sampler (and any
    # torch sampling) is bit-exact reproducible across runs.
    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    model = torch.load(params.model_path, map_location='cpu', weights_only=False)
    model.params.gpu = -1
    model.device = torch.device('cpu')
    # Security-KG ranking subgraphs pull in most of the graph via shared hub
    # nodes (e.g. CWE-94), so the dense path (.to_dense() over num_nodes x
    # num_edges) OOMs; force the memory-efficient sparse_index_select path.
    # The saved model baked in is_big_dataset=False at training time, so we
    # override it here rather than retrain.
    if params.dataset.startswith('security_kg'):
        model.is_big_dataset = True
    model.eval()  # dropout off for ranking (matches evaluator.py validation)
    params.max_label_value = np.array([2, 2])
    params.main_dir = 'utils/..'  # resolve the cluster-assign cache path (same as train.py)
    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, model.relation2id, add_traspose_rels=False)

    if getattr(params, 'max_links', 0) > 0:
        # 代表性随机子集：先按 seed 全量乱序再截断（默认 off，保持原「取前 N」行为）。
        # 注：文件头常按年代排序（早期 CVE），取前 N 是偏态子集——分层分解交叉验证须用乱序版。
        if getattr(params, 'subset_shuffle', False):
            rng = np.random.default_rng(params.seed)
            triplets['links'] = triplets['links'][rng.permutation(len(triplets['links']))]
        triplets['links'] = triplets['links'][:params.max_links]

    # Scheme 3: graph-external side channel (C1). Generic loader: reads
    # side_channel.npz when present (standard KGs), falls back to the legacy
    # product_fingerprint.json (security KG), else no injection.
    from utils.side_channel import load_side_channel
    entity_fingerprint, _ = load_side_channel(
        os.path.join('../data', params.dataset, 'side_channel.npz'),
        id2entity)

    # Per-CWE tail embedding index (0 = non-CWE/OOV), remapped by name against the
    # model's saved cwe2id (training CWE set; unseen eval CWEs -> OOV slot 0).
    cwe_index = build_cwe_index(id2entity, getattr(model, 'cwe2id', {}))

    # ---- C6: candidate pools + known positives (Bordes filtered semantics) ----
    # Answer-space pools: relation-consistent train tails/heads on standard KGs,
    # the CWE/CVE entity pools on the security KG. P0②: sampling negatives from
    # the answer space (not all entities) removes the "looks like a CWE" shortcut
    # the all-entity pool leaked.
    tail_pools = head_pools = None
    known_pos = None
    link_rels = np.asarray(triplets['links'])[:, 2]
    uniq_rels = np.unique(link_rels).astype(int).tolist()
    cwe_pool = np.asarray([eid for eid, name in id2entity.items()
                           if str(name).startswith('CWE-')], dtype=np.int64)
    cve_pool = np.asarray([eid for eid, name in id2entity.items()
                           if str(name).startswith('CVE-')], dtype=np.int64)
    if params.mode in ('sample', 'filtered') and params.dataset.startswith('security_kg'):
        tail_pools = {r: cwe_pool for r in uniq_rels}
        head_pools = {r: cve_pool for r in uniq_rels}
    if params.mode == 'filtered':
        # Known positives = train (from the ranking adjacency) + valid + test.
        # Head-cold queries have no train edges, so the test-set exclusion is
        # what matters there.
        known_pos = set()
        for rel_i, adj in enumerate(adj_list):
            hh, tt = adj.nonzero()
            for a, b in zip(hh.tolist(), tt.tolist()):
                known_pos.add((int(a), int(b), rel_i))
        for fname in ('valid.txt', 'test.txt'):
            fpath = os.path.join('../data', params.dataset, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    h = entity2id.get(parts[0])
                    t = entity2id.get(parts[2])
                    rel = relation2id.get(parts[1])
                    if h is not None and t is not None and rel is not None:
                        known_pos.add((int(h), int(t), int(rel)))
        if not params.dataset.startswith('security_kg'):
            tail_pools, head_pools = {}, {}
            for rel_i, adj in enumerate(adj_list):
                coo = adj.tocoo()
                tail_pools[rel_i] = np.unique(coo.col).astype(np.int64)
                head_pools[rel_i] = np.unique(coo.row).astype(np.int64)

    # ProtoRMPI: load the cached cluster assignment (reuses the training entity-id mapping).
    cluster_assign = None
    cluster_meta = None
    degrees = None
    if getattr(model, 'use_proto', False):
        n_clusters = getattr(model, 'n_clusters', 0)
        train_dataset = getattr(model.params, 'dataset', params.dataset)
        # Fully-inductive splits (same or different dataset name) must score
        # UNSEEN CVEs; the transductive remap would collapse them all to cluster
        # -1 -> clamp(0). Route B (structural features -> soft-assign to the
        # training centroids) gives every entity, seen or unseen, a real prototype.
        is_inductive = (params.dataset != train_dataset) or params.dataset.endswith('_ind')

        if is_inductive:
            # Inductive split: entity IDs are disjoint from training, so the
            # cached id2cluster/id2soft are meaningless; use the centroids meta
            # and compute features from the eval graph.
            cluster_meta = load_cluster_meta(params, n_clusters, dataset=train_dataset)
            if cluster_meta is None:
                raise RuntimeError(
                    f"Cluster meta missing for n_clusters={n_clusters} in data/{train_dataset}; "
                    "run training with ablation>=2 first to build it.")
            degrees = graph_degrees(dgl_adj_list)
            print(f"[ProtoRMPI] inductive eval: centroids {cluster_meta['centroids'].shape}, "
                  f"graph degrees over {len(degrees)} nodes")
        else:
            # entity2id is this script's (train+test) id space; the cache is keyed to
            # the training (train+valid) space, so pass it in to remap by name.
            cluster_assign = load_cluster_assignments(
                params, n_clusters, dataset=train_dataset, entity2id=entity2id)
            if cluster_assign is None:
                raise RuntimeError(
                    f"Cluster-assign cache missing for n_clusters={n_clusters}; "
                    "run training with ablation>=2 first.")
            print(f"[ProtoRMPI] loaded cluster assign: {cluster_assign[0].shape}, n_clusters={cluster_assign[2]}")

    all_mrr = []
    all_hit1 = []
    all_hit5 = []
    all_hit10 = []

    # ---- per-relation 拆分 ----------------------------------------------------
    # ranks 按链接顺序成对追加（head、tail），故 link i 对应 ranks[2i] 与 ranks[2i+1]，
    # 其关系即 triplets['links'][i][2]。两种负采样都按 links 顺序 1:1 生成，且
    # p.imap 保序，所以下标可直接对齐；数量对不上就整体跳过，不影响主结果。
    rel_ids = None
    try:
        rel_ids = np.asarray(triplets['links'])[:, 2]
    except Exception as e:
        print(f'[per-relation] disabled: {e}')
    per_rel = {}  # rel_id -> {'mrr': [...], 'h1': [...], 'h5': [...], 'h10': [...]}
    # 复用单个 worker pool（run 之间不再重建），避免内存 reap 杀掉 worker 后
    # 在 run 之间重建 pool 时主进程死锁（imap 等待永远不来的 worker 结果）。
    # get_rank 是纯函数，pool 复用 vs 每 run 重建结果完全一致。
    # maxtasksperchild=500：full_ind 单 worker 堆内存无界慢涨（torch/DGL C++ 分配器
    # 碎片，非 Python 泄漏、非大子图），实测 1000 link 爬到 5.4GB 顶爆整机。每 500 条
    # 强制重启 worker 释放堆，压住累计增长；但单条 hub-CWE 链接的 BFS 稀疏 matmul 会
    # 在窗口内瞬发冲 ~5.5GB，maxtasksperchild 压不住这个峰值——所以 workers 必须 =1，
    # 别用多 worker 叠加（见 rank_full_ind_serial_A.sh 七次 OOM 定案）。
    with mp.Pool(processes=params.workers, initializer=intialize_worker,
                 initargs=(model, adj_list, dgl_adj_list, id2entity, params,
                           cluster_assign, cluster_meta, degrees, entity_fingerprint, cwe_index),
                 maxtasksperchild=500) as p:
        for r in range(1, params.runs + 1):
            if params.mode in ('sample', 'filtered'):
                neg_triplets = get_neg_samples_replacing_head_tail(
                    triplets['links'], adj_list, num_samples=params.num_samples,
                    tail_pools=tail_pools, head_pools=head_pools, known_pos=known_pos)
                save_negative_triples_to_file(neg_triplets, id2entity, id2relation)
            elif params.mode == 'all':
                neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)

            ranks = []
            all_head_scores = []
            all_tail_scores = []
            for head_scores, head_rank, tail_scores, tail_rank in tqdm(p.imap(get_rank, neg_triplets), total=len(neg_triplets)):
                ranks.append(head_rank)
                ranks.append(tail_rank)

                all_head_scores += head_scores.tolist()
                all_tail_scores += tail_scores.tolist()

            # per-relation 累加：每条链接贡献 head/tail 两个 rank
            if rel_ids is not None and len(rel_ids) == len(ranks) // 2:
                ranks_arr = np.asarray(ranks, dtype=float)
                for rid in np.unique(rel_ids):
                    rr = ranks_arr[np.repeat(rel_ids == rid, 2)]
                    if rr.size == 0:
                        continue
                    d = per_rel.setdefault(int(rid), {'mrr': [], 'h1': [], 'h5': [], 'h10': []})
                    d['mrr'].append(float(np.mean(1.0 / rr)))
                    d['h1'].append(float(np.mean(rr <= 1)))
                    d['h5'].append(float(np.mean(rr <= 5)))
                    d['h10'].append(float(np.mean(rr <= 10)))

            isHit1List = [x for x in ranks if x <= 1]
            isHit5List = [x for x in ranks if x <= 5]
            isHit10List = [x for x in ranks if x <= 10]
            hits_1 = len(isHit1List) / len(ranks)
            hits_5 = len(isHit5List) / len(ranks)
            hits_10 = len(isHit10List) / len(ranks)

            mrr = np.mean(1 / np.array(ranks))

            # ---- 分层日志（§8.2 分层分解的 mini 交叉验证；默认关闭）----
            # 每链接一行：head tail head_rank tail_rank covered(0/1)
            # covered = 头实体在侧信道矩阵中非零（security-KG: CPE 指纹非空）。
            strat_log = getattr(params, 'strat_log', '')
            if strat_log and len(ranks) == 2 * len(triplets['links']):
                links_arr = np.asarray(triplets['links'])
                with open(strat_log, 'w', encoding='utf-8') as sf:
                    for i in range(len(links_arr)):
                        hname = id2entity.get(int(links_arr[i, 0]), '')
                        tname = id2entity.get(int(links_arr[i, 1]), '')
                        hr, tr = int(ranks[2 * i]), int(ranks[2 * i + 1])
                        cov = 0
                        if entity_fingerprint is not None:
                            cov = int(np.abs(np.asarray(entity_fingerprint[int(links_arr[i, 0])])).max() > 0)
                        sf.write(f'{hname}\t{tname}\t{hr}\t{tr}\t{cov}\n')
                print(f'[strat-log] wrote {len(links_arr)} rows to {strat_log}')

            all_mrr.append(mrr)
            all_hit1.append(hits_1)
            all_hit5.append(hits_5)
            all_hit10.append(hits_10)

            # print(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')
            print(f'MRR: {mrr: .4f}')
            print(f'Hits@1  : {hits_1: .4f}')
            print(f'Hits@5  : {hits_5: .4f}')
            print(f'Hits@10 : {hits_10: .4f}')

    avg_mrr = np.mean(all_mrr)
    avg_hit1 = np.mean(all_hit1)
    avg_hit5 = np.mean(all_hit5)
    avg_hit10 = np.mean(all_hit10)

    print('\nAvg test Set Performance ')
    print(f'MRR: {avg_mrr: .4f}')
    print(f'Hits@1  : {avg_hit1: .4f}')
    print(f'Hits@5  : {avg_hit5: .4f}')
    print(f'Hits@10 : {avg_hit10: .4f}')

    if per_rel:
        print(f'\nPer-relation (avg over {params.runs} run(s)) ')
        print(f"{'relation':<32}{'n_ranks':>9}{'MRR':>9}{'Hits@1':>9}{'Hits@5':>9}{'Hits@10':>9}")
        for rid in sorted(per_rel, key=lambda x: -int(np.sum(rel_ids == x))):
            d = per_rel[rid]
            n_ranks = int(np.sum(rel_ids == rid)) * 2
            print(f"{str(id2relation.get(rid, rid)):<32}{n_ranks:>9}"
                  f"{np.mean(d['mrr']):>9.4f}{np.mean(d['h1']):>9.4f}"
                  f"{np.mean(d['h5']):>9.4f}{np.mean(d['h10']):>9.4f}")
    elif rel_ids is not None:
        print('\n[per-relation] skipped: link/rank 数量不匹配')

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--expri_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2", help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "filtered", "ruleN"],
                        help="Negative sampling mode (filtered: answer-space pool + known-positive exclusion)")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2, help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('--target2nei_atten', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--conc', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--ablation', type=int, default=0, help='0,1 correspond to base, NE')
    parser.add_argument("--runs", type=int, default=5, help="How many runs to perform for mean and std?")
    parser.add_argument("--workers", type=int, default=2, help="Number of mp.Pool workers (lower to reduce peak memory)")
    parser.add_argument("--max_links", type=int, default=0, help="如果 >0，只评测前 N 个 test link（烟雾测试/测速用）")
    parser.add_argument("--subset_shuffle", action='store_true', help="max_links 截断前按 seed 全量乱序（代表性随机子集，默认 off）")
    parser.add_argument("--strat_log", type=str, default='', help="写逐链接分层日志 head tail head_rank tail_rank covered(0/1)（默认关）")
    parser.add_argument("--cache_size", type=int, default=3000, help="per-node BFS LRU 缓存条目上限（封顶 worker 内存）")
    parser.add_argument("--num_samples", type=int, default=50, help="每个 test link 的负样本数（替换 head/tail），降低可提速 ranking（50→25 提速约 2 倍）")

    params = parser.parse_args()

    params.file_paths = {
        'graph': os.path.join('../data', params.dataset, 'train.txt'),
        'links': os.path.join('../data', params.dataset, 'test.txt')
    }

    params.model_path = os.path.join('expri_save_models', params.expri_name, 'best_graph_classifier.pth')


    print('============ Params ============')
    print('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    print('============================================')

    main(params)
