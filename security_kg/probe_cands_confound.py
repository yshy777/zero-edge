"""probe_cands_confound.py — cands 模式 cwe_rate 的 coverage 混淆法医（一次性诊断）。

问题：cands 模式下 cwe_rate(top-1 预测类的历史被利用率) AUC 0.762，但 KEV 99/99
有指纹、dump 样本里 ~68% 非 KEV 无指纹 → 无指纹 CVE 的预测=流行度模式类（常数），
cwe_rate 对它们也是常数 → 该信号可能部分骑在「覆盖 vs 未覆盖」上而非真实匹配。

输出：模式类及其 rate、覆盖/未覆盖两组的 cwe_rate 特征分布、covered 指示器 AUC、
仅覆盖子集上的 cwe_rate AUC（信号的真实成分下界）。
"""
import json
import os
from collections import Counter, defaultdict

DATA = os.environ.get('RMPI_DATA',
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
KG = os.path.join(DATA, 'security_kg_full_ind')
BETA = 10.0

kev = set(json.load(open(os.path.join(DATA, 'security_kg_full', 'kev_labels.json'),
                         encoding='utf-8')))
fp = json.load(open(os.path.join(KG, 'product_fingerprint.json'), encoding='utf-8'))
cands = json.load(open(os.path.join(KG,
    'completion_cands_security_kg_full_ind_RMPI_proto_hard_sc.json'), encoding='utf-8'))


def auc(y, s):
    pairs = [(a, b) for a, b in zip(y, s)]
    npos = sum(y)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    return sum(1 for i in range(len(pairs)) for j in range(len(pairs))
               if pairs[i][0] == 1 and pairs[j][0] == 0 and pairs[i][1] > pairs[j][1]) \
        / (npos * nneg)


# cwe_rate table（train，与 risk_profile.py 同口径）
cwe_kev, cwe_tot, train_cves = defaultdict(int), defaultdict(int), set()
with open(os.path.join(KG, 'train.txt'), encoding='utf-8') as f:
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
            cwe_tot[p[2]] += 1
            train_cves.add(p[0])
            if p[0] in kev:
                cwe_kev[p[2]] += 1
prior = sum(1 for c in train_cves if c in kev) / max(len(train_cves), 1)
rate = {c: (cwe_kev[c] + BETA * prior) / (cwe_tot[c] + BETA) for c in cwe_tot}

top1 = Counter()
rows = []
for q, lst in cands.items():
    c = lst[0][0]
    top1[c] += 1
    cov = any(v != 0 for v in fp.get(q, [0.0]))
    rows.append((q, c, rate.get(c, 0.0), cov, q in kev))

mode_c, mode_n = top1.most_common(1)[0]
print(f'queries={len(rows)} kev={sum(r[4] for r in rows)}')
print(f'mode top-1 = {mode_c} ({mode_n}/{len(rows)}), train rate = {rate.get(mode_c, 0):.4f}')
cov_n = sum(r[3] for r in rows)
print(f'covered={cov_n}/{len(rows)}; KEV covered={sum(r[3] and r[4] for r in rows)}'
      f'/{sum(r[4] for r in rows)}; non-KEV covered={sum(r[3] and not r[4] for r in rows)}'
      f'/{sum(not r[4] for r in rows)}')
print(f'AUC(covered, KEV) = {auc([r[4] for r in rows], [r[3] for r in rows]):.4f}')

# cwe_rate 特征在 covered / uncovered 上的分布（uncovered 全是模式类常数）
cov_rates = sorted(r[2] for r in rows if r[3])
unc_rates = sorted(set(r[2] for r in rows if not r[3]))
print(f'cwe_rate: covered min/med/max = {cov_rates[0]:.4f}/{cov_rates[len(cov_rates)//2]:.4f}/{cov_rates[-1]:.4f}')
print(f'cwe_rate: uncovered distinct values = {len(unc_rates)} -> {unc_rates[:3]}...')
all_rates = sorted(r[2] for r in rows)
mode_rank = sum(1 for x in all_rates if x < rate.get(mode_c, 0)) / len(all_rates)
print(f'mode-rate percentile in sample = {mode_rank:.3f}')

# 仅覆盖子集上的 cwe_rate AUC（真实匹配成分）
sub = [r for r in rows if r[3]]
print(f'AUC(cwe_rate|covered only) = {auc([r[4] for r in sub], [r[2] for r in sub]):.4f}'
      f'  (n={len(sub)}, kev={sum(r[4] for r in sub)})')
# 全部（含 uncovered 常数）
print(f'AUC(cwe_rate|all) = {auc([r[4] for r in rows], [r[2] for r in rows]):.4f}')

# 模式类 rate 在 covered 预测 rate 中的位置：有多少 covered 预测 rate 高于模式类
above = sum(1 for r in rows if r[3] and r[2] > rate.get(mode_c, 0))
print(f'covered queries with rate > mode-rate: {above}/{cov_n}')
print(f'  of which KEV: {sum(1 for r in rows if r[3] and r[2] > rate.get(mode_c, 0) and r[4])}')
