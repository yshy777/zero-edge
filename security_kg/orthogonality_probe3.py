"""orthogonality_probe3.py — Stage-3 层0预探·三：CWE 历史被利用率 vs KEV。

前两探针：图拓扑零信号、产品/时间无组合增量。本探针测最后一个机制性信号：

  CWE 类型的历史被利用率 —— train 侧每个 CWE 关联的 CVE 中 KEV 的比例。
  安全领域公认 memory-corruption 类 CWE（CWE-787 越界写等）比 XSS 更易被野外利用；
  CVSS 编码严重度但不编码 weakness 类型。该信号正是补全模型预测的对象（CVE->CWE），
  若成立则 Stage 3 原创性闭环：补全预测 CWE -> KG 中该 CWE 的历史利用画像 -> 风险分。

上限探针：unseen CVE 用 test.txt 的 ground-truth CWE（同探针1），risk = 其 gt-CWE
历史利用率的最大/均值（beta 平滑）。实际 Stage 3 用补全预测 CWE 会打折，但先探上限。

Usage (from repo root):
  python security_kg/orthogonality_probe3.py --kg security_kg_full_ind
"""
import json
import os
import sys
from collections import defaultdict

import aco_completion as ac


def main():
    kg = 'security_kg_full_ind'
    if '--kg' in sys.argv:
        kg = sys.argv[sys.argv.index('--kg') + 1]

    kg_dir = os.path.join(ac.DATA_DIR, kg)
    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))

    # train-side CWE -> {kev, total} counts over has_weakness
    cwe_kev = defaultdict(int)
    cwe_tot = defaultdict(int)
    for fn in ('train.txt',):
        with open(os.path.join(kg_dir, fn), encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                    cwe_tot[p[2]] += 1
                    if p[0] in kev:
                        cwe_kev[p[2]] += 1
    print(f'[probe3] train CWEs={len(cwe_tot)} '
          f'(kev-marked CVE-has_weakness links={sum(cwe_kev.values())})')

    # Global KEV rate prior for beta smoothing
    train_cves = set()
    with open(os.path.join(kg_dir, 'train.txt'), encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                train_cves.add(p[0])
    prior = sum(1 for c in train_cves if c in kev) / max(len(train_cves), 1)
    print(f'[probe3] train CVEs={len(train_cves)} train kev={sum(1 for c in train_cves if c in kev)} '
          f'prior={prior:.5f}')

    BETA = 10.0  # beta-prior strength
    def cwe_rate(cwe):
        return (cwe_kev[cwe] + BETA * prior) / (cwe_tot[cwe] + BETA)

    # ground-truth CWE per unseen CVE (test.txt has_weakness)
    gt = defaultdict(list)
    with open(os.path.join(kg_dir, 'test.txt'), encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] == 'has_weakness' and p[0].startswith('CVE-'):
                gt[p[0]].append(p[2])

    unseen = ac.unseen_cves(kg_dir)
    labeled = [c for c in unseen if c in gt and (c in kev or c in cvss)]
    print(f'[probe3] unseen={len(unseen)} labeled={len(labeled)}')

    risks = {n: {} for n in ['cwe_rate_max', 'cwe_rate_mean', 'cvss']}
    for q in labeled:
        rates = [cwe_rate(c) for c in gt[q]]
        if rates:
            risks['cwe_rate_max'][q] = max(rates)
            risks['cwe_rate_mean'][q] = sum(rates) / len(rates)
        if q in cvss:
            risks['cvss'][q] = cvss[q]

    def bench(name, scores):
        qs = [q for q in labeled if q in scores]
        y_true = [1 if q in kev else 0 for q in qs]
        y_score = [scores[q] for q in qs]
        auc = ac.roc_auc(y_true, y_score)
        cvss_q = [q for q in qs if q in cvss]
        rho = (ac.spearman([scores[q] for q in cvss_q], [cvss[q] for q in cvss_q])
               if cvss_q else float('nan'))
        return auc, rho, sum(y_true), len(qs)

    print('\n=== single measures (KEV ROC-AUC / CVSS rho) ===')
    for name in ['cwe_rate_max', 'cwe_rate_mean', 'cvss']:
        auc, rho, kp, n = bench(name, risks[name])
        print(f'  {name:14s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    out = {name: risks[name] for name in risks}
    out_path = os.path.join(kg_dir, 'orthogonality_probe3.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'))
    print(f'[probe3] wrote {out_path}')


if __name__ == '__main__':
    main()
