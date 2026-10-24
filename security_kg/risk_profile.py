"""risk_profile.py — Stage-3 重定义后的威胁画像分（补全驱动的可解释威胁画像）。

risk(CVE) = sigma(w1*CVSS + w2*CWE历史被利用率 + w3*产品数 + w4*攻击链热度)

三探针（orthogonality_probe 1/2/3）已定：拓扑零信号、n_prod 回填伪影、cwe_rate
微弱正信号（GT 上限 +0.0027）。本脚本是执行清单步骤 1/4：把四个信号做成组合
画像分，按三关 v2 口径评估：

  ① 组合分 KEV AUC 显著 > 0.5（5-fold CV 的 OOF 预测，Hanley-McNeil SE）
  ② CVSS 分桶内条件排序（CVSS 并列时的二次排序：桶内 AUC + 桶内十分位 KEV 率）
  ③ 组合分 vs CVSS 的 Spearman rho + 增量分解表（cvss → +cwe_rate → +n_prod → +chain）
     ——含 cwe_noise 对照（置换 cwe_rate 特征，证明增量非频率伪影）

CWE 来源两种模式（--cwe_src）：
  gt   : test.txt 的 ground-truth CWE，取 max（上限探针，与探针 3 同口径）
  cands: dump_completion_cands.py 的补全候选（真实管线），top-1 为主、top-3 均值变体

w4（攻击链热度）作可解释载体不作判别：三探针已证其 AUC<0.5，组合里的增量
预期≈0，如实报告。链本身留给 case study（回溯链 CVE->CWE->CAPEC->technique->tactic）。

Usage (from repo root):
  python security_kg/risk_profile.py --kg security_kg_full_ind --cwe_src gt
  python security_kg/risk_profile.py --kg security_kg_full_ind \
      --cands data/security_kg_full_ind/completion_cands_security_kg_full_ind_RMPI_proto_hard_cwe.json
"""
import argparse
import json
import math
import os
from collections import defaultdict

import networkx as nx
import numpy as np

import aco_completion as ac

RAW = os.path.join(ac.DATA_DIR, 'security_kg', 'raw', 'nvd_cve_cpe.jsonl')
CHAIN_RELS = ['exploited_by', 'maps_to', 'has_tactic']
BETA = 10.0          # cwe_rate 的 beta-prior 强度（探针 3 同口径）
BUCKET_W = 0.5       # CVSS 分桶宽
MIN_POS_PER_BUCKET = 3
N_FOLDS = 5
SEED = 0


# ---------------------------------------------------------------- data loading

def load_raw_products():
    """{cve: n_prod} 从 raw jsonl；返回后即释放大字典。"""
    out = {}
    with open(RAW, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec['cve']] = len(rec.get('products') or [])
    return out


def build_attack_graph(kg_dir):
    rels = {'has_weakness', 'exploited_by', 'maps_to', 'has_tactic'}
    G = nx.DiGraph()
    with open(os.path.join(kg_dir, 'train.txt'), encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] in rels:
                G.add_edge(p[0], p[2])
    return G


def gt_cwes(kg_dir):
    out = defaultdict(list)
    with open(os.path.join(kg_dir, 'test.txt'), encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                out[p[0]].append(p[2])
    return out


def load_cands(path):
    """{cve: [['CWE-xxx', score], ...]} 降序。"""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    out = {}
    for q, lst in raw.items():
        out[q] = [c for c, _ in lst]
    return out


def cwe_rate_table(kg_dir, kev):
    """train 侧 CWE -> beta 平滑历史被利用率（探针 3 同口径）。"""
    cwe_kev = defaultdict(int)
    cwe_tot = defaultdict(int)
    train_cves = set()
    with open(os.path.join(kg_dir, 'train.txt'), encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                cwe_tot[p[2]] += 1
                train_cves.add(p[0])
                if p[0] in kev:
                    cwe_kev[p[2]] += 1
    prior = sum(1 for c in train_cves if c in kev) / max(len(train_cves), 1)
    return {cwe: (cwe_kev[cwe] + BETA * prior) / (cwe_tot[cwe] + BETA)
            for cwe in cwe_tot}, prior


def chain_from(cwe, G, indeg):
    """greedy highest-in-degree downstream chain（探针 1 同口径）。"""
    chain = [cwe]
    cur = cwe
    for _ in CHAIN_RELS:
        if cur not in G:
            break
        nxts = list(G.successors(cur))
        if not nxts:
            break
        cur = max(nxts, key=lambda v: indeg.get(v, 0))
        chain.append(cur)
    return chain


def chain_heat(cwe, G, indeg):
    """log1p(下游节点入度之和)。孤立 CWE -> 0。"""
    chain = chain_from(cwe, G, indeg)
    return math.log1p(sum(indeg.get(v, 0) for v in chain[1:]))


# ---------------------------------------------------------------- metrics

def percentile_ranks(scores):
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    return {q: (i / (n - 1.0) if n > 1 else 0.5) for i, (q, _) in enumerate(items)}


def auc_se(y_true, y_score):
    auc = ac.roc_auc(y_true, y_score)
    n1 = sum(y_true)
    n0 = len(y_true) - n1
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) \
        / (n0 * n1)
    return auc, math.sqrt(max(var, 0.0))


def spearman_x_y(xs, ys):
    return ac.spearman(xs, ys)


# ---------------------------------------------------------------- logistic CV

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_logistic(X, y, iters=50):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = _sigmoid(Z @ w)
        S = p * (1.0 - p) + 1e-6
        g = Z.T @ (y - p) - 1.0 * w
        H = Z.T @ (S[:, None] * Z) + 1.0 * np.eye(Z.shape[1])
        w += np.linalg.solve(H, g)
    return w


def logistic_cv(X, y, folds=N_FOLDS):
    """分层 5-fold，返回 OOF 概率。X 应为 rank 特征（[0,1] 内）。"""
    rng = np.random.RandomState(SEED)
    pos = [i for i, yi in enumerate(y) if yi == 1]
    neg = [i for i, yi in enumerate(y) if yi == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold_of = {}
    for f in range(folds):
        for i in pos[len(pos) * f // folds: len(pos) * (f + 1) // folds]:
            fold_of[i] = f
        for i in neg[len(neg) * f // folds: len(neg) * (f + 1) // folds]:
            fold_of[i] = f
    oof = np.empty(len(y))
    for f in range(folds):
        tr = [i for i in range(len(y)) if fold_of[i] != f]
        te = [i for i in range(len(y)) if fold_of[i] == f]
        w = fit_logistic(X[tr], y[tr])
        oof[te] = _sigmoid(X[te] @ w[1:] + w[0])  # X 未标准化：w 对 rank 特征
    return oof


# ---------------------------------------------------------------- gates

def bucket_eval(cvss, y, key):
    """② CVSS 分桶内条件排序：桶内 rank 作二次排序键，pooled 桶内 AUC + 十分位表。"""
    buckets = defaultdict(list)
    for q in key:
        if q in cvss:
            buckets[int(cvss[q] / BUCKET_W)].append(q)
    pooled_y, pooled_s = [], []
    rows = []
    for b in sorted(buckets):
        qs = buckets[b]
        kp = sum(1 for q in qs if y[q])
        if kp < MIN_POS_PER_BUCKET:
            continue
        pr = percentile_ranks({q: key[q] for q in qs})
        ys = [y[q] for q in qs]
        ss = [pr[q] for q in qs]
        pooled_y += ys
        pooled_s += ss
        rows.append({'bucket_cvss': round(b * BUCKET_W, 1),
                     'n': len(qs), 'kev': kp,
                     'auc': round(auc_se(ys, ss)[0], 4)})
    if not pooled_y:
        return None, rows
    order = sorted(range(len(pooled_s)), key=lambda i: pooled_s[i])
    deciles = []
    for d in range(10):
        idx = order[len(order) * d // 10: len(order) * (d + 1) // 10]
        deciles.append({'decile': d + 1, 'n': len(idx),
                        'kev': sum(pooled_y[i] for i in idx)})
    return auc_se(pooled_y, [pooled_s[i] for i in range(len(pooled_y))]), rows, deciles


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg', default='security_kg_full_ind')
    ap.add_argument('--cwe_src', choices=['gt', 'cands'], default='gt')
    ap.add_argument('--cands', default=None,
                    help='completion cands json path（--cwe_src cands 时必填）')
    args = ap.parse_args()

    kg_dir = os.path.join(ac.DATA_DIR, args.kg)
    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))

    print('[risk] cwe_rate table (train) ...')
    cwe_rate, prior = cwe_rate_table(kg_dir, kev)
    print(f'[risk] train CWEs={len(cwe_rate)} prior={prior:.5f}')

    print('[risk] n_prod (raw jsonl) ...')
    n_prod = load_raw_products()
    print(f'[risk] raw CVE records={len(n_prod)}')

    print('[risk] attack graph (train, 4 rels) ...')
    G = build_attack_graph(kg_dir)
    indeg = {v: d for v, d in G.in_degree()}
    print(f'[risk] nodes={G.number_of_nodes()} edges={G.number_of_edges()}')

    unseen = ac.unseen_cves(kg_dir)
    labeled = [q for q in unseen if q in cvss]
    y = {q: 1 if q in kev else 0 for q in labeled}
    print(f'[risk] unseen={len(unseen)} labeled(cvss)={len(labeled)} '
          f'kev={sum(y.values())}')

    # ---- CWE assignment
    if args.cwe_src == 'gt':
        gt = gt_cwes(kg_dir)
        cwes_of = {q: gt[q] for q in labeled if q in gt}
        print(f'[risk] gt-mode: covered={len(cwes_of)}/{len(labeled)}')
    else:
        assert args.cands, '--cwe_src cands 需要 --cands'
        cands = load_cands(args.cands)
        cwes_of = {q: [cands[q][0]] for q in labeled if q in cands and cands[q]}
        top3 = {q: cands[q][:3] for q in labeled if q in cands and len(cands[q]) >= 3}
        print(f'[risk] cands-mode top-1: covered={len(cwes_of)}/{len(labeled)}')
        # Restrict the evaluation universe to cands-covered CVEs. A capped dump
        # covers a KEV-stratified sample (all KEV + representative non-KEV), not
        # all ~42k unseen, so without this cvss/n_prod would be computed over the
        # full set while cwe_rate/chain_heat sit on the sample — misaligned.
        labeled = [q for q in labeled if q in cwes_of]
        y = {q: 1 if q in kev else 0 for q in labeled}
        print(f'[risk] cands-mode universe(restricted)={len(labeled)} '
              f'kev={sum(y.values())}')

    # ---- features（rank 变换）
    feats = {}
    feats['cvss'] = {q: cvss[q] for q in labeled}
    feats['cwe_rate'] = {q: max(cwe_rate.get(c, 0.0) for c in cs)
                         for q, cs in cwes_of.items()}
    if args.cwe_src == 'cands':
        feats['cwe_rate_top3'] = {q: sum(cwe_rate.get(c, 0.0) for c in cs) / len(cs)
                                  for q, cs in top3.items()}
    feats['n_prod'] = {q: float(n_prod[q]) for q in labeled if q in n_prod}
    feats['chain_heat'] = {q: max(chain_heat(c, G, indeg) for c in cs)
                           for q, cs in cwes_of.items()}

    ranks = {name: percentile_ranks(f) for name, f in feats.items()}

    # ---- 单特征 bench（①③ 的一半）
    print('\n=== single features (KEV AUC±SE / CVSS rho) ===')
    single = {}
    for name in ['cvss', 'cwe_rate', 'n_prod', 'chain_heat']:
        if args.cwe_src == 'cands' and name == 'cwe_rate':
            for vname in ['cwe_rate', 'cwe_rate_top3']:
                qs = [q for q in labeled if q in feats[vname]]
                a, se = auc_se([y[q] for q in qs], [feats[vname][q] for q in qs])
                rho = spearman_x_y([feats[vname][q] for q in qs], [cvss[q] for q in qs])
                single[vname] = {'auc': round(a, 4), 'se': round(se, 4),
                                 'rho': round(rho, 4), 'n': len(qs),
                                 'kev': sum(y[q] for q in qs)}
                print(f'  {vname:14s} AUC={a:.4f}±{se:.4f}  rho={rho:+.4f}  '
                      f'(kev={single[vname]["kev"]}/{len(qs)})')
            continue
        qs = [q for q in labeled if q in feats[name]]
        a, se = auc_se([y[q] for q in qs], [feats[name][q] for q in qs])
        rho = spearman_x_y([feats[name][q] for q in qs], [cvss[q] for q in qs])
        single[name] = {'auc': round(a, 4), 'se': round(se, 4),
                        'rho': round(rho, 4), 'n': len(qs), 'kev': sum(y[q] for q in qs)}
        print(f'  {name:14s} AUC={a:.4f}±{se:.4f}  rho={rho:+.4f}  '
              f'(kev={single[name]["kev"]}/{len(qs)})')

    # ---- 组合 CV（① 的 OOF + ③ 的增量分解）
    base = ['cvss', 'cwe_rate', 'n_prod', 'chain_heat']
    pool = [q for q in labeled
            if all(q in feats[f] for f in base) and q in n_prod]
    print(f'\n=== combination (5-fold CV OOF, n={len(pool)}, kev={sum(y[q] for q in pool)}) ===')
    X = {f: np.array([ranks[f][q] for q in pool]) for f in base}
    Y = np.array([y[q] for q in pool])
    cwe_name = 'cwe_rate'

    # cwe_noise 对照：置换 cwe_rate 特征（保持边际分布，打掉身份信息）
    rng = np.random.RandomState(SEED)
    noise = rng.permutation(X['cwe_rate'].copy())

    subsets = [
        ('cvss', ['cvss']),
        ('cvss+cwe_rate', ['cvss', cwe_name]),
        ('cvss+cwe_rate+n_prod', ['cvss', cwe_name, 'n_prod']),
        ('cvss+cwe_rate+n_prod+chain', base),
    ]
    combos = {}
    print(f'  {"subset":26s} {"OOF AUC":>9s} {"SE":>7s} {"vs CVSS":>9s} {"rho":>7s}')
    cvss_only_auc = None
    for label_, feats_sel in subsets:
        oof = logistic_cv(np.column_stack([X[f] for f in feats_sel]), Y)
        a, se = auc_se(Y, oof)
        rho = spearman_x_y(oof.tolist(), [cvss[q] for q in pool])
        d = '' if cvss_only_auc is None else f'{a - cvss_only_auc:+.4f}'
        if cvss_only_auc is None:
            cvss_only_auc = a
        combos[label_] = {'auc': round(a, 4), 'se': round(se, 4),
                          'delta': d, 'rho': round(rho, 4)}
        print(f'  {label_:26s} {a:9.4f} {se:7.4f} {d:>9s} {rho:+7.4f}')
    oof_n = logistic_cv(np.column_stack([X['cvss'], noise, X['n_prod']]), Y)
    a_n, se_n = auc_se(Y, oof_n)
    combos['cvss+cwe_noise+n_prod'] = {'auc': round(a_n, 4), 'se': round(se_n, 4)}
    print(f'  {"cvss+cwe_noise+n_prod":26s} {a_n:9.4f} {se_n:7.4f}  (置换对照)')

    # ① 显著判据
    a_best, se_best = combos['cvss+cwe_rate+n_prod+chain']['auc'], \
        combos['cvss+cwe_rate+n_prod+chain']['se']
    z = (a_best - 0.5) / se_best
    print(f'\n[gate1] full combo AUC={a_best:.4f}±{se_best:.4f} '
          f'z={z:.1f} {"PASS" if z > 1.96 else "FAIL"} (H0: 0.5)')

    # ---- ② 分桶条件排序
    print('\n=== gate2: CVSS bucket conditional ranking (key=画像分/单特征) ===')
    full_oof = {q: float(v) for q, v in zip(pool, logistic_cv(
        np.column_stack([X[f] for f in base]), Y))}
    bucket = {}
    for keyname, key in [('cwe_rate', feats['cwe_rate']),
                         ('full_combo', full_oof),
                         ('chain_heat', feats['chain_heat'])]:
        res = bucket_eval(cvss, y, key)
        if res[0] is None:
            print(f'  {keyname:12s} 桶内 KEV<{MIN_POS_PER_BUCKET} 不足，跳过')
            continue
        (a, se), rows, deciles = res
        bucket[keyname] = {'pooled_auc': round(a, 4), 'se': round(se, 4),
                           'buckets': rows, 'deciles': deciles}
        kev_by_dec = ' '.join(f'{d["kev"]}' for d in deciles)
        print(f'  {keyname:12s} pooled in-bucket AUC={a:.4f}±{se:.4f}  '
              f'(buckets={len(rows)})')
        print(f'     decile KEV counts (低->高): {kev_by_dec}')

    # ---- ③ rho 汇总
    print('\n=== gate3: rho vs CVSS（非零 + 增量已在组合表） ===')

    out = {
        'kg': args.kg, 'cwe_src': args.cwe_src,
        'labeled': len(labeled), 'kev': sum(y.values()),
        'prior': prior, 'single': single, 'combos': combos,
        'bucket': bucket,
    }
    out_path = os.path.join(kg_dir, f'risk_profile_{args.cwe_src}.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'))
    print(f'[risk] wrote {out_path}')


if __name__ == '__main__':
    main()
