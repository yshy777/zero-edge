"""
cluster_assign.py — Build entity -> cluster assignments (hard + soft) for the
ProtoRMPI prototype injection, keyed by the SAME integer entity IDs used by the
subgraph-extraction pipeline (process_files(params.file_paths)).

Flow: ClusterManager (structural features only, see use_semantic) -> KMeans ->
hard label `id2cluster[N]` and soft assignment `id2soft[N, C]` (temperature
softmax over cosine similarity to the KMeans centroids). Results are cached on
disk so consecutive ablation runs don't re-cluster.

The feature-space mismatch note in INTEGRATION_PLAN.md §2.2 is handled upstream:
these hand-crafted features are used ONLY to derive the (soft/hard) assignment
weights; the actual prototype vectors that enter the model are *learnable*
embeddings living in the relation-embedding space (see graph_classifier.py).

Inductive (route B): the cached `id2cluster`/`id2soft` are keyed to the training
(v1) entity-ID space. Inductive splits (e.g. `WN18RR_v1_ind`) use a fully
disjoint entity set, so the lookup is meaningless there. For those we persist the
KMeans centroids + config in `cluster_meta_nc{n}.npz` and, at eval time, compute
each entity's structural feature from the *eval* graph (`structural_feature_from_graph`)
and soft-assign it against the training centroids (`soft_assign_feature`). This
maps unseen entities into the same prototype space the model learned.
"""

import os
import sys
import json
import numpy as np
import scipy.sparse as ssp
import torch

# Make the repo-root `cluster_manager.py` importable from RMPI/utils/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cluster_manager import ClusterManager  # noqa: E402
from utils.data_utils import process_files  # noqa: E402


def _data_dir(params, dataset=None):
    ds = dataset if dataset is not None else params.dataset
    return os.path.abspath(os.path.join(params.main_dir, f'../data/{ds}'))


def _cache_paths(params, n_clusters, dataset=None):
    base = _data_dir(params, dataset)
    return (
        os.path.join(base, f'id2cluster_nc{n_clusters}.npy'),
        os.path.join(base, f'id2soft_nc{n_clusters}.npy'),
    )


def _id2entity_path(params, n_clusters, dataset=None):
    """Path to the training-space entity-id -> entity-string table.

    train.py builds entity ids over {train, valid} while the eval scripts build
    them over {train, test}. Both enumerate train.txt first, so ids for train
    entities agree, but the tail (ids >= |train|) is filled from valid.txt at
    train time and from test.txt at eval time -- i.e. it denotes *different*
    entities in the two spaces. Persisting this table lets eval remap the cached
    arrays by entity string instead of by integer position.
    """
    base = _data_dir(params, dataset)
    return os.path.join(base, f'id2entity_nc{n_clusters}.npy')


def _meta_path(params, n_clusters, dataset=None):
    base = _data_dir(params, dataset)
    return os.path.join(base, f'cluster_meta_nc{n_clusters}.npz')


def _save_id2entity(id2entity, num_entities, path):
    """Persist the training-space id -> entity-string table (object array)."""
    table = np.array([id2entity[i] for i in range(num_entities)], dtype=object)
    np.save(path, table, allow_pickle=True)
    print(f"[cluster_assign] Saved entity table ({table.shape[0]} entities) -> "
          f"{os.path.basename(path)}")


def remap_cluster_assignments(id2cluster, id2soft, train_id2entity, eval_entity2id):
    """Re-key cached cluster arrays from the training id space into the eval id space.

    Matching is by entity *string*; eval entities absent from training get the
    "unseen" encoding (-1 / all-zero), the same convention build_cluster_assignments
    uses for entities it never clustered.
    """
    n_clusters = id2soft.shape[1]
    n_eval = (max(eval_entity2id.values()) + 1) if eval_entity2id else 0
    out_cluster = np.full(n_eval, -1, dtype=np.int64)
    out_soft = np.zeros((n_eval, n_clusters), dtype=np.float32)

    pos = {}
    for i, ent in enumerate(train_id2entity):
        pos.setdefault(ent, i)  # first occurrence wins, matching process_files

    matched = 0
    for ent, eid in eval_entity2id.items():
        j = pos.get(ent)
        if j is not None:
            out_cluster[eid] = id2cluster[j]
            out_soft[eid] = id2soft[j]
            matched += 1
    return out_cluster, out_soft, matched


def load_cluster_assignments(params, n_clusters, dataset=None, entity2id=None):
    """
    Load cached (id2cluster, id2soft) for a given n_clusters, or None if absent.
    Never re-clusters.

    entity2id: the *eval* script's entity2id (str -> int). Pass it whenever the
        eval script builds its ids from a different file set than training did
        (train+test vs train+valid), which is the case for every eval script here.
        The arrays are then remapped by entity string into the eval id space.
        Passing None is the legacy path and returns the arrays in the TRAINING id
        space -- only correct if the caller also works in that space.
    """
    id2cluster_path, id2soft_path = _cache_paths(params, n_clusters, dataset)
    if not (os.path.exists(id2cluster_path) and os.path.exists(id2soft_path)):
        return None

    id2cluster = np.load(id2cluster_path)
    id2soft = np.load(id2soft_path)

    if entity2id is None:
        return id2cluster, id2soft, n_clusters

    ent_path = _id2entity_path(params, n_clusters, dataset)
    if not os.path.exists(ent_path):
        raise RuntimeError(
            f"Cluster cache in '{os.path.dirname(id2cluster_path)}' has no entity table "
            f"({os.path.basename(ent_path)}), so it cannot be remapped into the eval id space. "
            "Indexing it directly would silently attach the WRONG prototype to every entity "
            "whose id is >= |train|. Regenerate the table without retraining via "
            "`python utils/build_cluster_entity_table.py --dataset <ds> --n_clusters <C>`, "
            "or retrain with ablation>=2.")
    train_id2entity = np.load(ent_path, allow_pickle=True)

    out_cluster, out_soft, matched = remap_cluster_assignments(
        id2cluster, id2soft, train_id2entity, entity2id)
    print(f"[cluster_assign] remapped to eval id space: {matched}/{len(entity2id)} "
          f"entities matched by name ({len(entity2id) - matched} unseen)")
    return out_cluster, out_soft, n_clusters


def load_cluster_meta(params, n_clusters, dataset=None):
    """
    Load the persisted clustering meta (centroids + config) needed for the
    inductive eval path, or None if absent.
    """
    meta_path = _meta_path(params, n_clusters, dataset)
    if os.path.exists(meta_path):
        z = np.load(meta_path, allow_pickle=True)
        return {
            'centroids': z['centroids'],
            'num_relations': int(z['num_relations']),
            'temperature': float(z['temperature']),
            'n_clusters': int(z['n_clusters']),
        }
    return None


def _fit_cluster_manager(params, n_clusters, use_semantic):
    """Build + cluster a ClusterManager over the training triples (deterministic seed)."""
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, None)

    with open(params.file_paths['train']) as f:
        train_data = [line.split() for line in f.read().split('\n')[:-1]]
    triple_set = [(h, r, t) for h, r, t in train_data]

    # WN18RR entities carry no text semantics -> identity text map.
    entity2text = {eid: eid for eid in entity2id}

    cm = ClusterManager(entity2text, relation2id, triple_set, use_semantic=use_semantic)
    cm.cluster_entities(n_clusters=n_clusters, method='kmeans', random_state=params.seed)
    return cm, entity2id, id2entity


def _save_meta(cm, n_clusters, temperature, meta_path):
    centroids = np.stack([cm.prototypes[c] for c in range(n_clusters)]).astype(np.float64)
    np.savez(
        meta_path,
        centroids=centroids,
        num_relations=np.array(cm.num_relations),
        temperature=np.array(temperature),
        n_clusters=np.array(n_clusters),
    )
    print(f"[cluster_assign] Saved cluster meta (centroids={centroids.shape}, "
          f"num_relations={cm.num_relations})")


def build_cluster_assignments(params, n_clusters=None, use_semantic=False, temperature=0.1):
    """
    Cluster the training entities (structural features only) and return
        id2cluster : np.ndarray[int]   shape [num_entities]           hard label, -1 if unseen
        id2soft    : np.ndarray[float] shape [num_entities, C]        soft assignment (sums to 1)
        n_clusters : int
    keyed by the integer entity IDs from process_files(params.file_paths).

    Returns cached arrays if already computed for this n_clusters (re-generating
    only the cluster meta if it is missing).
    """
    if n_clusters is None:
        n_clusters = 64

    id2cluster_path, id2soft_path = _cache_paths(params, n_clusters)
    meta_path = _meta_path(params, n_clusters)
    ent_path = _id2entity_path(params, n_clusters)

    if os.path.exists(id2cluster_path) and os.path.exists(id2soft_path):
        id2cluster = np.load(id2cluster_path)
        id2soft = np.load(id2soft_path)
        if not os.path.exists(meta_path) or not os.path.exists(ent_path):
            # Assignments cached but meta (centroids) and/or the entity table
            # missing -> regenerate them via a deterministic re-cluster.
            cm, entity2id, id2entity = _fit_cluster_manager(params, n_clusters, use_semantic)
            if not os.path.exists(meta_path):
                mismatch = 0
                for i, eid_str in enumerate(cm.entities):
                    if eid_str in entity2id and int(cm.cluster_labels[i]) != int(id2cluster[entity2id[eid_str]]):
                        mismatch += 1
                if mismatch:
                    print(f"[cluster_assign] WARNING: re-cluster disagrees with cached "
                          f"id2cluster on {mismatch} entities (non-deterministic KMeans?)")
                _save_meta(cm, n_clusters, temperature, meta_path)
            if not os.path.exists(ent_path):
                _save_id2entity(id2entity, id2soft.shape[0], ent_path)
        print(f"[cluster_assign] Loaded cached assignments (n_clusters={n_clusters}, "
              f"{id2soft.shape[0]} entities, {int((id2cluster >= 0).sum())} seen)")
        return id2cluster, id2soft, n_clusters

    cm, entity2id, id2entity = _fit_cluster_manager(params, n_clusters, use_semantic)

    num_entities = len(entity2id)
    id2cluster = np.full(num_entities, -1, dtype=np.int64)
    id2soft = np.zeros((num_entities, cm.n_clusters), dtype=np.float32)

    features = cm._feature_matrix  # [n_train_entities, feat_dim], aligned with cm.entities
    for i, eid_str in enumerate(cm.entities):
        if eid_str in entity2id:
            idx = entity2id[eid_str]
            id2cluster[idx] = int(cm.cluster_labels[i])
            id2soft[idx] = cm.soft_assign(features[i], temperature=temperature).astype(np.float32)

    np.save(id2cluster_path, id2cluster)
    np.save(id2soft_path, id2soft)
    _save_meta(cm, n_clusters, temperature, meta_path)
    _save_id2entity(id2entity, num_entities, ent_path)
    print(f"[cluster_assign] Built + cached assignments (n_clusters={n_clusters}, "
          f"{num_entities} entities, {int((id2cluster >= 0).sum())} seen, feat_dim={features.shape[1]})")

    return id2cluster, id2soft, cm.n_clusters


# =============================================================================
# Inductive (route B) helpers — compute cluster assignment from the eval graph.
# =============================================================================

def graph_degrees(dgl_graph):
    """Unique undirected degree of every node, as an int64 numpy array [num_nodes]."""
    src, dst = dgl_graph.edges()
    u = torch.cat([src, dst])
    v = torch.cat([dst, src])
    edges = torch.stack([torch.minimum(u, v), torch.maximum(u, v)], dim=0)
    edges = torch.unique(edges, dim=1)
    n = dgl_graph.num_nodes()
    deg = torch.zeros(n, dtype=torch.long)
    flat = edges.reshape(-1)
    deg = deg.scatter_add(0, flat, torch.ones(flat.numel(), dtype=torch.long))
    return deg.numpy()


def structural_feature_from_graph(dgl_graph, entity_id, num_relations, degrees):
    """
    Mirror ClusterManager._structural_features, but derive the entity's features
    from the *eval* graph (for unseen entities). `degrees` is the precomputed
    unique undirected degree array (see graph_degrees).
    """
    src, dst = dgl_graph.edges()
    etypes = dgl_graph.edata['type']
    e = int(entity_id)

    out_mask = src == e
    out_nbr = dst[out_mask]
    out_rel = etypes[out_mask]
    in_mask = dst == e
    in_nbr = src[in_mask]
    in_rel = etypes[in_mask]

    neighbors = torch.cat([out_nbr, in_nbr]).unique()
    relations = torch.cat([out_rel, in_rel]).unique()

    degree = int(neighbors.numel())
    num_rels = int(relations.numel())

    if degree > 0:
        degs = degrees[neighbors.numpy()].astype(np.float64)
        deg_stats = np.array([np.mean(degs), np.max(degs), np.min(degs), np.std(degs)])
    else:
        deg_stats = np.zeros(4)

    # Binary relation indicator over the entity's incident relation types,
    # normalized by sum — exactly what _structural_features does.
    rel_vec = np.zeros(num_relations)
    for r in relations.numpy().tolist():
        if 0 <= r < num_relations:
            rel_vec[r] = 1.0
    rel_vec = rel_vec / (rel_vec.sum() + 1e-8)

    basic_feat = np.array([degree, num_rels], dtype=np.float64)
    struct_f = np.concatenate([basic_feat, deg_stats, rel_vec])
    return struct_f / (np.linalg.norm(struct_f) + 1e-8)


def precompute_structural_features(dgl_graph, num_relations, degrees):
    """Vectorized mirror of structural_feature_from_graph over *all* nodes at once.

    Row e equals structural_feature_from_graph(dgl_graph, e, num_relations, degrees),
    but computed in O(E log E) instead of O(N * E) (the per-node implementation calls
    dgl_graph.edges() on every node, which is the proto_hard ranking bottleneck).
    Returns a float64 [num_nodes, 6 + num_relations] matrix.
    """
    src = dgl_graph.edges()[0].numpy().astype(np.int64)
    dst = dgl_graph.edges()[1].numpy().astype(np.int64)
    etypes = dgl_graph.edata['type'].numpy().astype(np.int64)
    n = dgl_graph.num_nodes()
    deg = np.asarray(degrees, dtype=np.float64)

    # Binary incident-relation indicator (out + in), normalized by row sum.
    rel_cnt = np.zeros((n, num_relations), dtype=np.float64)
    np.add.at(rel_cnt, (src, etypes), 1.0)
    np.add.at(rel_cnt, (dst, etypes), 1.0)
    rel_bin = (rel_cnt > 0).astype(np.float64)
    rel_vec = rel_bin / (rel_bin.sum(axis=1, keepdims=True) + 1e-8)
    num_rels = rel_bin.sum(axis=1)

    # Unique undirected neighbours via a symmetrized, deduped adjacency (==
    # cat([out_nbr, in_nbr]).unique() and == graph_degrees).
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    und = np.unique(np.stack([u, v], axis=1), axis=0)
    A_und = ssp.csr_matrix((np.ones(len(und), dtype=np.float64), (und[:, 0], und[:, 1])),
                           shape=(n, n))
    A_und = A_und + A_und.T
    A_und = A_und.tocsr()

    feat_dim = 6 + num_relations
    F = np.zeros((n, feat_dim), dtype=np.float64)
    F[:, 0] = deg
    F[:, 1] = num_rels
    indptr = A_und.indptr
    indices = A_und.indices
    for e in range(n):
        nbrs = indices[indptr[e]:indptr[e + 1]]
        if nbrs.size:
            ds = deg[nbrs]
            F[e, 2] = ds.mean()
            F[e, 3] = ds.max()
            F[e, 4] = ds.min()
            F[e, 5] = ds.std()
    F[:, 6:] = rel_vec
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    return F / (norms + 1e-8)


def soft_assign_feature(feature, centroids, temperature):
    """
    Soft-assign a feature vector to the K clusters by cosine similarity against
    the training centroids (mirrors ClusterManager.soft_assign).
    """
    f = np.asarray(feature, dtype=np.float64)
    sims = centroids @ f / (np.linalg.norm(centroids, axis=1) * np.linalg.norm(f) + 1e-8)
    alpha = np.exp(sims / temperature)
    return alpha / alpha.sum()


def soft_assign_features_batch(features, centroids, temperature):
    """Vectorized soft_assign_feature over a batch of feature rows.

    Row n of the result equals soft_assign_feature(features[n], centroids, temperature).
    Returns float64 [N, K]. Mirrors the per-node math exactly (incl. the 1e-8 eps),
    but does one [N, D] @ [D, K] matmul instead of N Python-level calls — this is the
    proto_hard ranking fast path (a hub disclosing subgraph touches ~26k nodes x 102
    subgraphs per link, so the per-node loop alone was ~2s/link).
    """
    F = np.asarray(features, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    fn = np.linalg.norm(F, axis=1, keepdims=True)   # [N, 1]
    cn = np.linalg.norm(c, axis=1, keepdims=True)   # [K, 1]
    denom = fn @ cn.T + 1e-8                        # [N, K]
    sims = (F @ c.T) / denom
    alpha = np.exp(sims / temperature)
    return alpha / alpha.sum(axis=1, keepdims=True)
