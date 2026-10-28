"""dump_completion_cands.py — Stage-2 completion candidate source.

For each unseen test CVE, score (CVE, has_weakness, CWE) over every CWE entity
with a trained RMPI/ProtoRMPI checkpoint, and dump sorted candidates as JSON:

    { "<CVE>": [["<CWE>", score], ...], ... }   # descending score, top-K per CVE

This is the "completion confidence" that the completion-aware ACO consumes at
hop-1 (CVE -> CWE). We reuse test_ranking_P.py's subgraph + proto-cluster
machinery verbatim (it is proven on the security_kg_ind ranking runs) instead
of re-implementing subgraph extraction / cluster attachment.

Why we need this file: the `*_ranking_head/tail.txt` files are NOT score-sorted
(they are raw negative samples written in `sample` mode before scoring), so the
rank order there carries no confidence. We re-score all CWEs per unseen CVE.

Usage (from RMPI/):
  python dump_completion_cands.py -d security_kg_ind -e security_kg_ind_RMPI_proto_hard \
      --max_cves 0 --top_k 50 --chunk 50
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

import test_ranking_P as trp


def load_model_and_graph(dataset, expri_name):
    params = argparse.Namespace()
    params.dataset = dataset
    params.expri_name = expri_name
    params.file_paths = {
        'graph': os.path.join('..', 'data', dataset, 'train.txt'),
        'links': os.path.join('..', 'data', dataset, 'test.txt'),
    }
    params.model_path = os.path.join('expri_save_models', expri_name,
                                     'best_graph_classifier.pth')
    params.max_label_value = np.array([2, 2])
    params.main_dir = 'utils/..'

    model = torch.load(params.model_path, map_location='cpu', weights_only=False)
    model.params.gpu = -1
    model.device = torch.device('cpu')
    if dataset in ('security_kg', 'security_kg_ind',
                   'security_kg_full', 'security_kg_full_ind'):
        model.is_big_dataset = True
    model.eval()

    # Subgraph extraction settings come from the model's own training params.
    params.hop = getattr(model.params, 'hop', 2)
    params.enclosing_sub_graph = getattr(model.params, 'enclosing_sub_graph', True)

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = \
        trp.process_files(params.file_paths, model.relation2id, add_traspose_rels=False)

    # Scheme 3: product-family fingerprint for CVE nodes (zero for non-CVEs).
    from subgraph_extraction.datasets import load_product_fingerprint
    entity_fingerprint, _ = load_product_fingerprint(
        os.path.join('..', 'data', dataset, 'product_fingerprint.json'),
        id2entity)

    # Per-CWE tail embedding index (0 = non-CWE/OOV), remapped by name against the
    # model's saved cwe2id (training CWE set; unseen eval CWEs -> OOV slot 0).
    cwe_index = trp.build_cwe_index(id2entity, getattr(model, 'cwe2id', {}))

    # Proto cluster setup — replicate test_ranking_P.main() exactly.
    cluster_assign = None
    cluster_meta = None
    degrees = None
    if getattr(model, 'use_proto', False):
        n_clusters = getattr(model, 'n_clusters', 0)
        train_dataset = getattr(model.params, 'dataset', params.dataset)
        is_inductive = (params.dataset != train_dataset) or params.dataset.endswith('_ind')
        if is_inductive:
            cluster_meta = trp.load_cluster_meta(params, n_clusters, dataset=train_dataset)
            degrees = trp.graph_degrees(dgl_adj_list)
        else:
            cluster_assign = trp.load_cluster_assignments(
                params, n_clusters, dataset=train_dataset, entity2id=entity2id)

    # Publish globals + precompute the proto-hard inductive machinery via the same
    # intialize_worker() the ranking path uses. Doing it inline (just assigning
    # cluster_meta_/degrees_) leaves soft_assign_=None -> attach_cluster crashes on
    # soft_assign_[nodes], and leaves A_incidence_/_bfs_cache_=None -> subgraph
    # extraction falls back to per-link incidence_matrix rebuild + uncached BFS.
    trp.intialize_worker(model, adj_list, dgl_adj_list, id2entity, params,
                         cluster_assign, cluster_meta, degrees,
                         entity_fingerprint, cwe_index)

    return params, model, adj_list, dgl_adj_list, entity2id, id2entity


def collect_entity_sets(dataset, entity2id):
    """Return (cve_names, cwe_names) — unseen test CVEs and all CWE entities.

    "Unseen" is keyed on has_weakness only: a CVE is unseen iff it is a
    has_weakness head in test.txt but not in train.txt. (After the
    hasVulnerability fix, test CVEs DO appear in train.txt as *tails* of
    product edges, so a naive head/tail-presence check would wrongly mark
    them seen and score nothing.)"""
    def read(fn):
        out = set()
        path = os.path.join('..', 'data', dataset, fn)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    p = line.rstrip('\n').split('\t')
                    if len(p) == 3:
                        out.add(tuple(p))
        return out

    cwe_names = set()
    for fn in ('train.txt', 'valid.txt', 'test.txt'):
        for p in read(fn):
            for name in (p[0], p[2]):
                if name.startswith('CWE-'):
                    cwe_names.add(name)

    def hw_heads(fn):
        return {p[0] for p in read(fn)
                if p[1] == 'has_weakness' and p[0].startswith('CVE-')}

    cve_names = {c for c in (hw_heads('test.txt') - hw_heads('train.txt'))
                 if c in entity2id}
    cwe_names = {c for c in cwe_names if c in entity2id}
    return cve_names, cwe_names


def load_kev_labels(dataset):
    """KEV positive CVE set from the transductive sidecar (strips '_ind' fallback).

    The threat-profile label is extremely imbalanced (~0.23% KEV among unseen
    CVEs); the dump sorts KEV CVEs first so a capped --max_cves still captures
    every positive instead of a ~1-in-400 random sliver."""
    base = dataset[:-4] if dataset.endswith('_ind') else dataset
    path = os.path.join('..', 'data', base, 'kev_labels.json')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return set(json.load(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', '-d', required=True)
    ap.add_argument('--expri_name', '-e', required=True)
    ap.add_argument('--max_cves', type=int, default=0, help='0 = all; else cap for a smoke run')
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--chunk', type=int, default=50)
    args = ap.parse_args()

    params, model, adj_list, dgl_adj_list, entity2id, id2entity = \
        load_model_and_graph(args.dataset, args.expri_name)

    rel_id = model.relation2id['has_weakness']
    cve_names, cwe_names = collect_entity_sets(args.dataset, entity2id)
    # KEV-first: with the label this rare, a capped dump must front-load all
    # positives or the threat-profile AUC is computed over ~0 KEV. Order among
    # non-KEV CVEs is alphabetical (deterministic, reproducible).
    kev = load_kev_labels(args.dataset)
    cve_names = sorted(cve_names, key=lambda c: (c not in kev, c))
    n_kev = sum(1 for c in cve_names if c in kev)
    if args.max_cves:
        cve_names = cve_names[:args.max_cves]
    cwe_ids = [entity2id[c] for c in sorted(cwe_names)]

    print(f'[dump] {len(cve_names)} unseen CVEs (KEV={n_kev} capped={bool(args.max_cves)}) '
          f'x {len(cwe_ids)} CWEs, rel=has_weakness({rel_id})', flush=True)

    out = {}
    t0 = __import__('time').time()
    for ci, cve_name in enumerate(cve_names):
        cve_id = entity2id[cve_name]
        all_scores = []
        for i in range(0, len(cwe_ids), args.chunk):
            chunk = cwe_ids[i:i + args.chunk]
            links = [[cve_id, cwe, rel_id] for cwe in chunk]
            data = trp.get_subgraphs(links, adj_list, dgl_adj_list,
                                     params.max_label_value)
            s = model(data).squeeze(1).detach().numpy()
            all_scores.append(s)
        all_scores = np.concatenate(all_scores)
        order = np.argsort(all_scores)[::-1][:args.top_k]
        out[cve_name] = [[sorted(cwe_names)[j], float(all_scores[j])] for j in order]
        if (ci + 1) % 100 == 0:
            print(f'[dump] {ci + 1}/{len(cve_names)} '
                  f'({(ci + 1) * 1.0 / max(len(cve_names), 1) * 100:.1f}%, '
                  f'{__import__("time").time() - t0:.0f}s)', flush=True)

    out_path = os.path.join('..', 'data', args.dataset,
                            f'completion_cands_{args.expri_name}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f'[dump] wrote {out_path} ({len(out)} CVEs)', flush=True)


if __name__ == '__main__':
    main()
