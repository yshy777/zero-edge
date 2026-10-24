"""orthogonality_probe.py — Stage-3 层0预探：拓扑特征 vs CVSS/KEV 的正交性上限探针。

不需要 cwe-fix 重训。用 test.txt 的 ground-truth has_weakness 边把 unseen CVE 接入
train 攻击图，算纯拓扑中心性（PageRank / betweenness / in-degree）能否区分 KEV、
以及是否与 CVSS 正交。这是「补全完美时，拓扑还有没有 CVSS 之外的信号」的上限探针：
若连 ground-truth 接入的纯拓扑都零信号，则 attack centrality（受补全误差拖累）更没戏；
若有正交信号，则 ③ 重定义风险分有戏。

与 attack_centrality_baselines.py 的差别：hop-1 不用 completion cands（重训产物），
改用 test.txt 的 ground-truth CWE。每个 CVE 取所有 gt-CWE 下游链风险的最大值（上限）。

Usage (from repo root):
  python security_kg/orthogonality_probe.py --kg security_kg_full_ind
"""
import json
import os
import sys
from collections import defaultdict

import networkx as nx

import aco_completion as ac

DOWNSTREAM = ['exploited_by', 'maps_to', 'has_tactic']


def build_attack_graph(kg_dir):
    rels = {'has_weakness', 'exploited_by', 'maps_to', 'has_tactic'}
    G = nx.DiGraph()
    for fn in ('train.txt',):
        path = os.path.join(kg_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) == 3 and p[1] in rels:
                    G.add_edge(p[0], p[2])
    return G


def gt_cwes(kg_dir):
    """{cve: [cwe...]} ground-truth has_weakness from test.txt."""
    out = defaultdict(list)
    path = os.path.join(kg_dir, 'test.txt')
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                out[p[0]].append(p[2])
    return out


def chain_from(cwe, G, indeg):
    """greedy highest-in-degree downstream chain from a ground-truth CWE.

    The ground-truth CWE may not be in the train attack graph (semi-inductive:
    the unseen CVE's held-out weakness can point at a CWE never seen in train);
    in that case the chain is just the isolated CWE (risk contribution 0).
    """
    chain = [cwe]
    cur = cwe
    for _ in DOWNSTREAM:
        if cur not in G:
            break
        nxts = list(G.successors(cur))
        if not nxts:
            break
        cur = max(nxts, key=lambda v: indeg.get(v, 0))
        chain.append(cur)
    return chain


def percentile_ranks(scores):
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    return {q: (i / (n - 1.0) if n > 1 else 0.5) for i, (q, _) in enumerate(items)}


def main():
    kg = 'security_kg_full_ind'
    if '--kg' in sys.argv:
        kg = sys.argv[sys.argv.index('--kg') + 1]

    kg_dir = os.path.join(ac.DATA_DIR, kg)
    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))

    print(f'[probe] building attack graph from train (4 rels) ...')
    G = build_attack_graph(kg_dir)
    print(f'[probe] nodes={G.number_of_nodes()} edges={G.number_of_edges()}')
    indeg = {v: d for v, d in G.in_degree()}

    print('[probe] pagerank ...')
    pr = nx.pagerank(G, alpha=0.85)
    print('[probe] betweenness (k=1000 approx) ...')
    bc = nx.betweenness_centrality(G, k=1000, normalized=True, seed=0)

    centralities = {
        'pagerank': pr,
        'betweenness': bc,
        'indegree': {v: float(d) for v, d in indeg.items()},
    }

    gt = gt_cwes(kg_dir)
    unseen = ac.unseen_cves(kg_dir)
    labeled = [c for c in unseen if c in gt and (c in kev or c in cvss)]
    print(f'[probe] unseen={len(unseen)} gt_covered={sum(1 for c in unseen if c in gt)} '
          f'labeled={len(labeled)}')

    # risk_q(name, q) = max over gt-CWEs of (sum over chain of centrality(node))
    risks = {name: {} for name in centralities}
    risks['cvss'] = {q: cvss[q] for q in labeled if q in cvss}
    for q in labeled:
        for cwe in gt[q]:
            chain = chain_from(cwe, G, indeg)
            for name, cent in centralities.items():
                r = sum(cent.get(v, 0.0) for v in chain)
                if r > risks[name].get(q, -1.0):
                    risks[name][q] = r

    # How many labeled CVEs have at least one gt-CWE actually in the train graph?
    in_graph = sum(1 for q in labeled if any(c in G for c in gt[q]))
    print(f'[probe] labeled with >=1 gt-CWE in train graph: {in_graph}/{len(labeled)}')

    def bench(name, scores):
        qs = [q for q in labeled if q in scores]
        kev_q = [q for q in qs if q in kev]
        y_true = [1 if q in kev else 0 for q in qs]
        y_score = [scores[q] for q in qs]
        auc = ac.roc_auc(y_true, y_score)
        cvss_q = [q for q in qs if q in cvss]
        rho = (ac.spearman([scores[q] for q in cvss_q], [cvss[q] for q in cvss_q])
               if cvss_q else float('nan'))
        return auc, rho, len(kev_q), len(qs)

    print('\n=== single measures (KEV ROC-AUC / CVSS rho) ===')
    for name in ['pagerank', 'betweenness', 'indegree', 'cvss']:
        auc, rho, kp, n = bench(name, risks[name])
        print(f'  {name:12s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    print('\n=== ensemble (topo + cvss, rank-average) ===')
    cvss_rank = percentile_ranks(risks['cvss'])
    for name in ['pagerank', 'betweenness', 'indegree']:
        base_rank = percentile_ranks(risks[name])
        ens = {q: (cvss_rank.get(q, 0.0) + base_rank.get(q, 0.0)) / 2.0
               for q in labeled if q in cvss_rank and q in base_rank}
        auc, rho, kp, n = bench(f'cvss+{name}', ens)
        print(f'  cvss+{name:9s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    out = {name: risks[name] for name in centralities}
    out['cvss'] = risks['cvss']
    out_path = os.path.join(kg_dir, 'orthogonality_probe.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'))
    print(f'[probe] wrote {out_path}')


if __name__ == '__main__':
    main()
