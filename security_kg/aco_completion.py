"""aco_completion.py — Completion-aware ACO risk scoring (stage 2/3).

This is the "real" pheromone from INTEGRATION_PLAN §0.5 修订3, replacing the
structural-tau baseline in aco.py (which is kept as the *motivation ablation*).

The unseen CVE's has_weakness edge is held out, so the ACO cannot walk hop-1
(CVE -> CWE) from the graph. Instead it consumes the RMPI completion candidates
(dump_completion_cands.py) as the hop-1 transition distribution, then walks the
*seen* downstream chain CWE ->(exploited_by)-> CAPEC ->(maps_to)-> technique
->(has_tactic)-> tactic through the train backbone.

Ablations (all share the SAME top-K candidate set, so the only thing that
varies is whether/how completion confidence enters):
  A  = structural tau only  : hop-1 uniform over candidates, tau-sum fitness
  A' = completion-aware eta : hop-1 P(cwe) ~ conf^beta,        tau-sum fitness
  B  = completion-aware fit : hop-1 uniform,            conf-weighted fitness
  C  = full                 : both eta and fitness completion-aware

Risk(CVE) = best-path score under per-query tau (stage-2 semantics: tau reset
every query, see §8.10.1). Benchmarked vs KEV (ROC-AUC) and CVSS (Spearman rho).

Usage (from repo root):
  python security_kg/aco_completion.py --kg security_kg_ind \
      --expri security_kg_ind_RMPI_proto_hard --ablation C --top_k 20
"""

import json
import math
import os
import random
import sys
from collections import defaultdict

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Chain relations, walked in this order from a query CVE.
CHAIN = ['has_weakness', 'exploited_by', 'maps_to', 'has_tactic']
DOWNSTREAM = CHAIN[1:]  # seen relations: CWE -> CAPEC -> technique -> tactic

# ACO hyperparameters (mirrors aco.py).
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


def load_train_graph(kg_dir):
    """Adjacency + degrees from train.txt only (the *seen* part for inductive)."""
    adj = defaultdict(list)
    degrees = defaultdict(int)
    path = os.path.join(kg_dir, 'train.txt')
    if not os.path.exists(path):
        return adj, degrees
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


def sidecar_dir(kg_dir):
    """KEV/CVSS sidecars are written to the transductive OUT dir by build_kg.py,
    not the *_ind split dir. Fall back to the sibling (strip '_ind')."""
    if os.path.exists(os.path.join(kg_dir, 'kev_labels.json')):
        return kg_dir
    return kg_dir[:-4] if kg_dir.endswith('_ind') else kg_dir


def load_completion_cands(kg_dir, expri):
    """{cve: [[cwe, score], ...]} descending score (top-K), or {} if absent."""
    path = os.path.join(kg_dir, f'completion_cands_{expri}.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def unseen_cves(kg_dir):
    """CVEs present in test.txt but absent from train.txt (the inductive target)."""
    def cves_of(fn):
        out = set()
        path = os.path.join(kg_dir, fn)
        if not os.path.exists(path):
            return out
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                    out.add(p[0])
        return out
    return sorted(cves_of('test.txt') - cves_of('train.txt'))


def softmax_conf(cands, T=1.0):
    """Turn [(cwe, raw_score), ...] into [(cwe, p), ...] with p summing to 1."""
    if not cands:
        return []
    xs = [s for _, s in cands]
    mx = max(xs)
    exps = [math.exp((s - mx) / T) for s in xs]
    z = sum(exps)
    return [(cwe, e / z) for (cwe, _), e in zip(cands, exps)]


def _sample(items, weights, rng):
    """Weighted sample; returns the chosen item (items = list of node strings)."""
    total = sum(weights)
    if total <= 0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    chosen = items[-1]
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return chosen


def aco_walk(query, cands_conf, tau, adj, degrees, params, rng, use_conf_eta):
    """One ant walk. Returns (path, conf) where conf is the completion confidence
    of the chosen hop-1 CWE (1.0 when eta is not completion-aware)."""
    cwes = [c for c, _ in cands_conf]
    if use_conf_eta:
        weights = [(tau.get(c, params['tau0']) ** params['alpha'])
                   * (p ** params['beta']) for c, p in cands_conf]
    else:
        weights = [tau.get(c, params['tau0']) ** params['alpha'] for c in cwes]
    cwe = _sample(cwes, weights, rng)
    conf = dict(cands_conf).get(cwe, 1.0 / max(len(cwes), 1))

    path = [query, cwe]
    cur = cwe
    for rel in DOWNSTREAM:
        nxts = adj.get((rel, cur), [])
        if not nxts:
            break
        weights = [(tau.get(v, params['tau0']) ** params['alpha'])
                   * (structural_eta(v, degrees) ** params['beta']) for v in nxts]
        chosen = _sample(nxts, weights, rng)
        path.append(chosen)
        cur = chosen
    return path, conf


def aco_reason(query, cands_conf, adj, degrees, params, rng,
               use_conf_eta, use_conf_fitness):
    """Per-query ACO (tau reset). Returns (best_path, best_score)."""
    tau = defaultdict(lambda: params['tau0'])
    best_path = None
    best_score = -1.0
    for _ in range(params['rounds']):
        for _ in range(params['n_ants']):
            path, conf = aco_walk(query, cands_conf, tau, adj, degrees, params,
                                  rng, use_conf_eta)
            # Completion-weighted deposit: high-conf hop-1 reinforces harder.
            deposit = params['delta_tau'] * (conf if use_conf_fitness else 1.0)
            for v in path[1:]:
                tau[v] += deposit
            score = sum(tau.get(v, params['tau0']) for v in path[1:])
            if use_conf_fitness:
                score *= conf
            if score > best_score:
                best_score = score
                best_path = path
        for v in list(tau.keys()):
            tau[v] *= (1.0 - params['rho'])
    return best_path, best_score


def roc_auc(y_true, y_score):
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs)
    ranks = []
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks.extend([avg] * (j - i))
        i = j
    pos = [r for r, (_, t) in zip(ranks, pairs) if t == 1]
    n_pos = len(pos)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    R = sum(pos)
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

    use_conf_eta = ablation in ('A1', 'C')   # A' rendered "A1" on the CLI
    use_conf_fitness = ablation in ('B', 'C')

    kg_dir = os.path.join(DATA_DIR, kg)
    adj, degrees = load_train_graph(kg_dir)
    cands = load_completion_cands(kg_dir, expri) if expri else {}

    sc_dir = sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))

    queries = [c for c in unseen_cves(kg_dir)
               if (c in kev or c in cvss) and c in cands]

    params = dict(PARAMS)
    rng = random.Random(params['seed'])

    scores = {}
    best_paths = {}
    for q in queries:
        cc = softmax_conf(cands[q][:top_k])
        if not cc:
            continue
        path, score = aco_reason(q, cc, adj, degrees, params, rng,
                                 use_conf_eta, use_conf_fitness)
        scores[q] = score
        best_paths[q] = path

    n = len(scores)
    if n == 0:
        print('[aco_comp] no scored CVEs (run dump_completion_cands.py first)')
        return

    kev_queries = [q for q in queries if q in kev]
    y_true = [1 if q in kev else 0 for q in queries]
    y_score = [scores[q] for q in queries]
    auc = roc_auc(y_true, y_score)

    cvss_queries = [q for q in queries if q in cvss]
    rho = (spearman([scores[q] for q in cvss_queries], [cvss[q] for q in cvss_queries])
           if cvss_queries else float('nan'))

    print(f'[aco_comp] kg={kg} ablation={ablation} expri={expri} top_k={top_k}')
    print(f'[aco_comp] queries={n} kev_pos={len(kev_queries)} cvss={len(cvss_queries)}')
    print(f'[aco_comp] KEV ROC-AUC = {auc:.4f}')
    print(f'[aco_comp] CVSS Spearman rho = {rho:.4f}')

    sample = [q for q in queries if q in kev][:5]
    for q in sample:
        print(f'[aco_comp] {q}  ->  {" -> ".join(best_paths[q])}')

    out = {q: scores[q] for q in queries}
    out_path = os.path.join(kg_dir, f'aco_risk_{ablation}_{expri}.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'), indent=0)
    print(f'[aco_comp] wrote {out_path} ({n} scores)')


if __name__ == '__main__':
    main()
