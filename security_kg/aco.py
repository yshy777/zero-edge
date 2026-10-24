"""aco.py — Stage 2: ACO reasoning over the security KG.

Design (locked with the user):
  - Node-level pheromone tau on CWE / CAPEC / technique / tactic nodes.
  - K = 2 walk rounds per query: round 1 explores, round 2 converges under the
    reinforced tau.
  - Structural prior eta = degree centrality (log-scaled) — NO label leakage
    (KEV / CVSS never touch the transition probability).
  - Per-query tau reset (local pheromone; no cross-query contamination).

Attack chain walked by each ant (4 relations in order):
  CVE --has_weakness--> CWE --exploited_by--> CAPEC --maps_to--> technique --has_tactic--> tactic

Risk score of a query CVE = cumulative tau along its converged (highest-tau)
attack path. Benchmarked against CISA KEV (binary exploited, ROC-AUC) and
CVSS (regression, Spearman rho).

Usage:
  python security_kg/aco.py [--kg security_kg | security_kg_ind]
"""

import json
import math
import os
import random
import sys
from bisect import bisect_left
from collections import defaultdict

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Chain relations, walked in this order from a query CVE.
CHAIN = ['has_weakness', 'exploited_by', 'maps_to', 'has_tactic']

# ACO hyperparameters.
PARAMS = dict(rounds=2, n_ants=20, alpha=1.0, beta=1.0,
              rho=0.1, tau0=1.0, delta_tau=1.0, seed=41504)


def node_type(node):
    if node.startswith('CVE-'):
        return 'cve'
    if node.startswith('CWE-'):
        return 'cwe'
    if node.startswith('CAPEC-'):
        return 'capec'
    if node.startswith('tactic:'):
        return 'tactic'
    return 'technique'  # Txxxx / Txxxx.xxx


def load_graph(kg_dir):
    """Return (adj, degrees). adj[(relation, head)] -> [tail, ...]."""
    adj = defaultdict(list)
    degrees = defaultdict(int)
    for name in ('train.txt', 'valid.txt', 'test.txt'):
        path = os.path.join(kg_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h, r, t = line.split('\t')
                adj[(r, h)].append(t)
                degrees[h] += 1
                degrees[t] += 1
    return adj, degrees


def structural_eta(node, degrees):
    """Degree centrality, log-scaled (structural only, no label signal)."""
    return math.log1p(degrees.get(node, 0)) + 1.0


def aco_walk(start, tau, adj, degrees, params, rng):
    """One ant walk along the attack chain. Returns the list of visited nodes
    (starting with the query CVE). Transition: P(v) ~ tau(v)^alpha * eta(v)^beta."""
    path = [start]
    cur = start
    for rel in CHAIN:
        nxts = adj.get((rel, cur), [])
        if not nxts:
            break
        weights = [(tau.get(v, params['tau0']) ** params['alpha'])
                   * (structural_eta(v, degrees) ** params['beta']) for v in nxts]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        chosen = nxts[-1]
        for v, w in zip(nxts, weights):
            acc += w
            if r <= acc:
                chosen = v
                break
        path.append(chosen)
        cur = chosen
    return path


def aco_reason(query, adj, degrees, params, rng):
    """K-round ACO for one query. Returns (tau, best_path, best_score)."""
    tau = defaultdict(lambda: params['tau0'])  # per-query reset
    best_path = None
    best_score = -1.0
    for _ in range(params['rounds']):
        for _ in range(params['n_ants']):
            p = aco_walk(query, tau, adj, degrees, params, rng)
            # Deposit on downstream nodes only (the query CVE itself is the root).
            for v in p[1:]:
                tau[v] += params['delta_tau']
            score = sum(tau.get(v, params['tau0']) for v in p[1:])
            if score > best_score:
                best_score = score
                best_path = p
        # Evaporate all deposited pheromone.
        for v in list(tau.keys()):
            tau[v] *= (1.0 - params['rho'])
    return tau, best_path, best_score


def roc_auc(y_true, y_score):
    """Rank-based ROC-AUC (no sklearn dependency)."""
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs)
    ranks = []
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0  # 1-indexed average rank of the tie block
        ranks.extend([avg] * (j - i))
        i = j
    pos_ranks = [r for r, (_, t) in zip(ranks, pairs) if t == 1]
    n_pos = len(pos_ranks)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    R = sum(pos_ranks)
    return (R - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ranks(xs):
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            r[order[k]] = avg
        i = j
    return r


def spearman(a, b):
    n = len(a)
    ra, rb = _ranks(a), _ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return float('nan')
    return cov / (va * vb) ** 0.5


def main():
    kg = 'security_kg'
    if '--kg' in sys.argv:
        kg = sys.argv[sys.argv.index('--kg') + 1]
    kg_dir = os.path.join(DATA_DIR, kg)

    adj, degrees = load_graph(kg_dir)
    kev = json.load(open(os.path.join(kg_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(kg_dir, 'cvss_scores.json'), encoding='utf-8'))

    # Query every CVE in the graph that carries a KEV or CVSS label.
    queries = sorted(c for c in set(kev) | set(cvss)
                     if node_type(c) == 'cve' and adj.get(('has_weakness', c)))

    params = dict(PARAMS)
    rng = random.Random(params['seed'])

    scores = {}
    best_paths = {}
    for q in queries:
        _, best_path, best_score = aco_reason(q, adj, degrees, params, rng)
        scores[q] = best_score
        best_paths[q] = best_path

    n = len(scores)
    if n == 0:
        print('[aco] no labeled CVEs in graph; run build_kg.py first')
        return

    # Benchmark vs KEV (binary exploited) and CVSS (regression).
    kev_queries = [q for q in queries if q in kev]
    y_true = [1 if q in kev else 0 for q in queries]
    y_score = [scores[q] for q in queries]
    auc = roc_auc(y_true, y_score)

    cvss_queries = [q for q in queries if q in cvss]
    if cvss_queries:
        rho = spearman([scores[q] for q in cvss_queries],
                       [cvss[q] for q in cvss_queries])
    else:
        rho = float('nan')

    print(f'[aco] kg={kg}  queries={n}  kev_pos={len(kev_queries)}  '
          f'cvss={len(cvss_queries)}')
    print(f'[aco] KEV ROC-AUC = {auc:.4f}')
    print(f'[aco] CVSS Spearman rho = {rho:.4f}')

    # Interpretability: show the converged attack chains of a few exploited CVEs.
    sample = [q for q in queries if q in kev][:5]
    for q in sample:
        print(f'[aco] {q}  ->  {" -> ".join(best_paths[q])}')

    out = {q: scores[q] for q in queries}
    json.dump(out, open(os.path.join(kg_dir, 'aco_risk.json'), 'w', encoding='utf-8'),
              indent=0)
    print(f'[aco] wrote aco_risk.json ({n} scores)')


if __name__ == '__main__':
    main()
