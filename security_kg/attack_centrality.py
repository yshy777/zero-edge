"""attack_centrality.py — Stage 3: stateful-tau attack centrality.

Upgrades the per-query-reset tau in aco_completion.py (stage-2 semantics) to a
stateful tau that ACCUMULATES across queries (INTEGRATION_PLAN §8.10.1 / §4.2).

Two outputs:
  1. Node attack centrality: centrality(v) = global tau[v] after accumulation,
     i.e. how many attack chains (across ALL unseen CVEs) pass through v. This
     is the "attack centrality" concept — the interpretable attack landscape.
  2. Per-CVE risk: risk(CVE) = sum of global tau along its best downstream
     chain. A CVE is a root node (no incoming attack-chain edge), so its risk
     is the centrality of its downstream CWE/CAPEC/technique/tactic nodes.

Differences from aco_completion.py (the stateless stage-2 baseline):
  - Accumulation runs over ALL unseen CVEs (unlabeled included), not only the
    KEV/CVSS-labeled subset — otherwise the tau landscape is empty.
  - Two-phase: Phase 1 accumulates global tau; Phase 2 scores every labeled
    query against the SAME converged tau (order-independent, fair benchmark).

For a clean stateless-vs-stateful ablation, both are computed in one run with
independently-seeded rngs (stateless = aco_completion.aco_reason, per-query reset).

Two stateful modes:
  - STATIC attack centrality (default: --alpha 0 --rho 0): ants follow the
    completion+degree prior (no pheromone feedback), tau is a pure traversal
    count. Order-independent, directly comparable to PageRank/betweenness/degree.
  - TEMPORAL stateful tau (--alpha 1 --rho 0.05): ACO positive feedback +
    evaporation toward tau0; the "有状态性" memory — meaningful only under a
    chronological query order (risk evolution over time).

Usage (from security_kg/):
  python attack_centrality.py --kg security_kg_ind \
      --expri security_kg_ind_RMPI_proto_hard_cwe --ablation C --top_k 20
"""

import json
import os
import random
import sys
from collections import defaultdict

import aco_completion as ac


def stateful_accumulate(queries, cands, top_k, adj, degrees, params, rng,
                        use_conf_eta, use_conf_fitness):
    """Phase 1: accumulate one global tau across all queries (fixed seed order).

    Same deposit rule as aco_completion.aco_reason, except tau persists across
    queries and evaporation happens ONCE PER QUERY (not per round) with a floor
    at tau0:  tau <- tau0 + (tau - tau0) * (1 - rho).

    Why the floor + per-query schedule: aco_reason's per-round rho=0.1 is tuned
    for 2 rounds; across 6270 rounds it decays everything but the last ~5 queries
    to 0 (sliding window), destroying the cross-query centrality. Evaporating
    toward tau0 keeps never- vs rarely-trafficked nodes at a meaningful baseline.
    """
    rho = params.get('rho_q', params['rho'])
    global_tau = defaultdict(lambda: params['tau0'])
    for q in queries:
        cc = ac.softmax_conf(cands[q][:top_k])
        if not cc:
            continue
        for _ in range(params['rounds']):
            for _ in range(params['n_ants']):
                path, conf = ac.aco_walk(q, cc, global_tau, adj, degrees,
                                         params, rng, use_conf_eta)
                deposit = params['delta_tau'] * (conf if use_conf_fitness else 1.0)
                for v in path[1:]:
                    global_tau[v] += deposit
        if rho > 0:
            for v in list(global_tau.keys()):
                global_tau[v] = params['tau0'] + (global_tau[v] - params['tau0']) * (1.0 - rho)
    return global_tau

def score_query(q, cands_conf, global_tau, adj, degrees, params, rng,
                use_conf_eta, use_conf_fitness):
    """Phase 2: score one query against the converged global tau (read-only)."""
    best_path = None
    best_score = -1.0
    for _ in range(params['rounds']):
        for _ in range(params['n_ants']):
            path, conf = ac.aco_walk(q, cands_conf, global_tau, adj, degrees,
                                     params, rng, use_conf_eta)
            score = sum(global_tau.get(v, params['tau0']) for v in path[1:])
            if use_conf_fitness:
                score *= conf
            if score > best_score:
                best_score = score
                best_path = path
    return best_path, best_score


def top_by_type(global_tau, k=5):
    """Group centrality by node type and return the top-k of each."""
    groups = defaultdict(list)
    for v, t in global_tau.items():
        groups[ac.node_type(v)].append((t, v))
    out = {}
    for typ, items in groups.items():
        items.sort(reverse=True)
        out[typ] = [[v, round(t, 4)] for t, v in items[:k]]
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

    rho_q = 0.0
    if '--rho' in argv:
        i = argv.index('--rho')
        rho_q = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    alpha_q = 0.0
    if '--alpha' in argv:
        i = argv.index('--alpha')
        alpha_q = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    use_conf_eta = ablation in ('A1', 'C')
    use_conf_fitness = ablation in ('B', 'C')

    kg_dir = os.path.join(ac.DATA_DIR, kg)
    adj, degrees = ac.load_train_graph(kg_dir)
    cands = ac.load_completion_cands(kg_dir, expri) if expri else {}

    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))

    # Accumulation over ALL unseen CVEs with candidates; benchmark over the
    # labeled subset only (see module docstring).
    accum = [c for c in ac.unseen_cves(kg_dir) if c in cands]
    labeled = [c for c in accum if (c in kev or c in cvss)]

    if not accum:
        print('[atkcent] no unseen CVEs with candidates; run dump_completion_cands.py first')
        return

    params_sl = dict(ac.PARAMS)                 # stateless: original hyperparams (alpha=1, per-round rho)
    params_sf = dict(ac.PARAMS)                 # stateful: static centrality defaults (alpha=0, rho_q=0)
    params_sf['rho_q'] = rho_q                  # per-query evaporation (temporal variant sets >0)
    params_sf['alpha'] = alpha_q                # pheromone feedback (static centrality sets 0)
    rng_sl = random.Random(params_sl['seed'])   # stateless (per-query reset)
    rng_sf = random.Random(params_sf['seed'])   # stateful (cross-query)

    # --- stateless baseline (stage-2 semantics) ---
    scores_sl = {}
    for q in labeled:
        cc = ac.softmax_conf(cands[q][:top_k])
        if not cc:
            continue
        _, s = ac.aco_reason(q, cc, adj, degrees, params_sl, rng_sl,
                             use_conf_eta, use_conf_fitness)
        scores_sl[q] = s

    # --- stateful attack centrality (stage-3 semantics) ---
    global_tau = stateful_accumulate(accum, cands, top_k, adj, degrees, params_sf,
                                     rng_sf, use_conf_eta, use_conf_fitness)
    scores_sf = {}
    best_paths = {}
    for q in labeled:
        cc = ac.softmax_conf(cands[q][:top_k])
        if not cc:
            continue
        path, s = score_query(q, cc, global_tau, adj, degrees, params_sf, rng_sf,
                              use_conf_eta, use_conf_fitness)
        scores_sf[q] = s
        best_paths[q] = path

    def report(name, scores):
        qs = [q for q in labeled if q in scores]
        kev_q = [q for q in qs if q in kev]
        y_true = [1 if q in kev else 0 for q in qs]
        y_score = [scores[q] for q in qs]
        auc = ac.roc_auc(y_true, y_score)
        cvss_q = [q for q in qs if q in cvss]
        rho = (ac.spearman([scores[q] for q in cvss_q], [cvss[q] for q in cvss_q])
               if cvss_q else float('nan'))
        print(f'[{name}] queries={len(qs)} kev_pos={len(kev_q)} cvss={len(cvss_q)} '
              f'KEV ROC-AUC={auc:.4f} CVSS Spearman={rho:.4f}')
        return auc, rho

    print(f'[atkcent] kg={kg} ablation={ablation} expri={expri} top_k={top_k}')
    print(f'[atkcent] unseen-with-cands={len(accum)} labeled={len(labeled)}')
    report('stateless', scores_sl)
    report('stateful ', scores_sf)

    # Interpretability: attack landscape top-K per node type.
    print('[atkcent] attack centrality top-K (node -> tau):')
    for typ, items in sorted(top_by_type(global_tau).items()):
        print(f'  {typ:10s} ' + '  '.join(f'{v}={t}' for v, t in items))

    # A few exploited CVEs' backtracking chains under stateful tau.
    sample = [q for q in labeled if q in kev][:5]
    for q in sample:
        print(f'[atkcent] {q}  ->  {" -> ".join(best_paths[q])}')

    out_c = {v: global_tau[v] for v in global_tau}
    out_c_path = os.path.join(kg_dir, f'attack_centrality_{ablation}_{expri}.json')
    json.dump(out_c, open(out_c_path, 'w', encoding='utf-8'), indent=0)
    print(f'[atkcent] wrote {out_c_path} ({len(out_c)} nodes)')

    out_r = {q: scores_sf[q] for q in labeled if q in scores_sf}
    out_r_path = os.path.join(kg_dir, f'aco_risk_stateful_{ablation}_{expri}.json')
    json.dump(out_r, open(out_r_path, 'w', encoding='utf-8'), indent=0)
    print(f'[atkcent] wrote {out_r_path} ({len(out_r)} scores)')


if __name__ == '__main__':
    main()
