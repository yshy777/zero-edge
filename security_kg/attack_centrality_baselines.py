"""attack_centrality_baselines.py — Stage-3 baseline comparison + ensemble.

Benchmarks the attack centrality (completion-aware pheromone tau) against the
structural centrality baselines the paper must complement (INTEGRATION_PLAN
§8.4 #2 / stage-3 checklist item 4): PageRank, betweenness, degree, CVSS-only.

Fair comparison: every measure shares the SAME deterministic attack chain per CVE
(top-1 completion CWE at hop-1, greedy highest-degree downstream), so measures
differ only in the node-centrality function, not the chain extraction:

  risk_X(CVE)   = sum over chain nodes of X(node),  X in {tau, pagerank, betweenness, degree}
  risk_cvss(CVE)= cvss[CVE]

Ensemble = mean of percentile ranks of (attack-tau risk, baseline risk) — rank
averaging is robust to the tiny KEV positive count. The supervised-GNN risk head
is a separate (non-cheap) baseline, deferred.

Usage (from security_kg/):
  python attack_centrality_baselines.py --kg security_kg_ind \
      --expri security_kg_ind_RMPI_proto_hard_cwe --ablation C --top_k 20
"""

import json
import os
import sys
from collections import defaultdict

import networkx as nx

import aco_completion as ac

DOWNSTREAM = ['exploited_by', 'maps_to', 'has_tactic']


def load_tau_centrality(kg_dir, ablation, expri):
    path = os.path.join(kg_dir, f'attack_centrality_{ablation}_{expri}.json')
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}


def build_attack_graph(kg_dir):
    """Directed graph over the 4 attack-chain relations from train.txt."""
    rels = {'has_weakness', 'exploited_by', 'maps_to', 'has_tactic'}
    G = nx.DiGraph()
    path = os.path.join(kg_dir, 'train.txt')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) == 3 and p[1] in rels:
                    G.add_edge(p[0], p[2])
    return G


def deterministic_chain(q, cands, adj, degrees):
    """[cwe, capec, technique, tactic] — top-1 completion CWE + greedy degree."""
    if not cands.get(q):
        return []
    cwe = cands[q][0][0]  # highest completion score
    chain = [cwe]
    cur = cwe
    for rel in DOWNSTREAM:
        nxts = adj.get((rel, cur), [])
        if not nxts:
            break
        cur = max(nxts, key=lambda v: degrees.get(v, 0))
        chain.append(cur)
    return chain


def percentile_ranks(scores):
    """Map each score to its [0,1] percentile among all CVEs."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    out = {}
    for i, (q, _) in enumerate(items):
        out[q] = i / (n - 1.0) if n > 1 else 0.5
    return out


def main():
    kg = 'security_kg_ind'
    expri = None
    ablation = 'C'
    top_k = 20
    argv = sys.argv[1:]
    for flag in ('--kg', '--expri', '--ablation'):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            if flag == '--kg':
                kg = val
            elif flag == '--expri':
                expri = val
            else:
                ablation = val
            argv = argv[:i] + argv[i + 2:]
    if '--top_k' in argv:
        i = argv.index('--top_k')
        top_k = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    kg_dir = os.path.join(ac.DATA_DIR, kg)
    adj, degrees = ac.load_train_graph(kg_dir)
    cands = ac.load_completion_cands(kg_dir, expri) if expri else {}

    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))

    tau = load_tau_centrality(kg_dir, ablation, expri)

    print(f'[baselines] building attack graph (4 relations, train) ...')
    G = build_attack_graph(kg_dir)
    print(f'[baselines] nodes={G.number_of_nodes()} edges={G.number_of_edges()}')
    print(f'[baselines] pagerank ...')
    pr = nx.pagerank(G, alpha=0.85)
    print(f'[baselines] betweenness (k=1000 approximate) ...')
    bc = nx.betweenness_centrality(G, k=1000, normalized=True, seed=0)

    # Labeled unseen CVEs with candidates = the benchmark population.
    labeled = [c for c in ac.unseen_cves(kg_dir) if c in cands and (c in kev or c in cvss)]

    # Centrality functions (node -> score).
    centralities = {
        'attack_tau': tau,
        'pagerank': pr,
        'betweenness': bc,
        'degree': {v: float(degrees.get(v, 0)) for v in G.nodes()},
    }

    # Build per-measure CVE risk (sum over the shared deterministic chain).
    risks = {name: {} for name in centralities}
    risks['cvss'] = {q: cvss[q] for q in labeled if q in cvss}
    for q in labeled:
        chain = deterministic_chain(q, cands, adj, degrees)
        for name, cent in centralities.items():
            risks[name][q] = sum(cent.get(v, 0.0) for v in chain)

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

    print(f'[baselines] kg={kg} ablation={ablation} expri={expri} '
          f'labeled={len(labeled)}')

    print('\n=== single measures (KEV ROC-AUC / CVSS rho) ===')
    for name in ['attack_tau', 'pagerank', 'betweenness', 'degree', 'cvss']:
        auc, rho, kp, n = bench(name, risks[name])
        print(f'  {name:12s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    # Ensembles: attack-tau percentile-rank-averaged with each baseline.
    print('\n=== ensemble (attack_tau + baseline, rank-average) ===')
    tau_rank = percentile_ranks(risks['attack_tau'])
    for name in ['pagerank', 'betweenness', 'degree', 'cvss']:
        base_rank = percentile_ranks(risks[name])
        ens = {q: (tau_rank.get(q, 0.0) + base_rank.get(q, 0.0)) / 2.0
               for q in labeled if q in tau_rank and q in base_rank}
        auc, rho, kp, n = bench(f'ens_tau+{name}', ens)
        print(f'  tau+{name:9s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    out = {'risks': risks, 'chain': {q: deterministic_chain(q, cands, adj, degrees)
                                     for q in labeled}}
    out_path = os.path.join(kg_dir, f'centrality_baselines_{ablation}_{expri}.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'))
    print(f'[baselines] wrote {out_path}')


if __name__ == '__main__':
    main()
