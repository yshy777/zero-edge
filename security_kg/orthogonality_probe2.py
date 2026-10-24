"""orthogonality_probe2.py — Stage-3 层0预探·二：产品攻击面 + 时间信号 vs KEV。

探针1证明 4-关系攻击链拓扑对 KEV 零信号。本探针测 raw NVD 数据里 KG 之外的两个
维度，判断它们是否携带「被利用性」信号、是否与 CVSS 正交：

  1. 产品攻击面：受影响产品(vendor:product)的流行度（该产品被多少 CVE 关联）、
     受影响产品个数。流行产品（Windows/Chrome/Apache 类）的漏洞更可能被野外利用。
  2. 时间信号：CVE 发布时间。新爆发的漏洞更可能被立即利用（KEV 收录节奏）。

三个信号都从 nvd_cve_cpe.jsonl 直接提取，不依赖重训。产品流行度用全量统计
（产品属性非 CVE 标签，部署时分析师同样可查该产品的 CVE 历史，无泄漏问题）。

Usage (from repo root):
  python security_kg/orthogonality_probe2.py --kg security_kg_full_ind
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import aco_completion as ac

RAW = os.path.join(ac.DATA_DIR, 'security_kg', 'raw', 'nvd_cve_cpe.jsonl')


def load_raw(path):
    """{cve: {'products': [...], 'published': 'YYYY-MM-DD...', 'cvss': f}}"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec['cve']] = rec
    return out


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], '%Y-%m-%d')
    except ValueError:
        return None


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

    print(f'[probe2] loading {RAW} ...')
    raw = load_raw(RAW)
    print(f'[probe2] raw CVE records = {len(raw)}')

    # Product popularity over ALL raw CVEs (product attribute, not a CVE label).
    prod_freq = defaultdict(int)
    for rec in raw.values():
        for p in rec.get('products') or []:
            prod_freq[p] += 1
    print(f'[probe2] distinct products = {len(prod_freq)}')

    unseen = ac.unseen_cves(kg_dir)
    labeled = [c for c in unseen if c in raw and (c in kev or c in cvss)]
    n_raw_missing = sum(1 for c in unseen if c not in raw)
    print(f'[probe2] unseen={len(unseen)} labeled={len(labeled)} raw-missing={n_raw_missing}')

    # Reference: latest published date in the whole corpus (relative age).
    all_dates = [d for rec in raw.values()
                 if (d := parse_date(rec.get('published'))) is not None]
    max_date = max(all_dates)
    print(f'[probe2] corpus published range: {min(all_dates).date()} .. {max_date.date()}')

    risks = {name: {} for name in ['prod_max', 'prod_sum', 'n_prod', 'time', 'cvss']}
    for q in labeled:
        rec = raw[q]
        prods = rec.get('products') or []
        if prods:
            pops = [prod_freq[p] for p in prods]
            risks['prod_max'][q] = max(pops)
            risks['prod_sum'][q] = sum(pops)
            risks['n_prod'][q] = len(prods)
        if q in cvss:
            risks['cvss'][q] = cvss[q]
        d = parse_date(rec.get('published'))
        if d is not None:
            risks['time'][q] = (d - max_date).days  # <=0; 0 = newest

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
    for name in ['prod_max', 'prod_sum', 'n_prod', 'time', 'cvss']:
        auc, rho, kp, n = bench(name, risks[name])
        print(f'  {name:10s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    print('\n=== ensemble (cvss + signal, rank-average) ===')
    cvss_rank = percentile_ranks(risks['cvss'])
    for name in ['prod_max', 'prod_sum', 'n_prod', 'time']:
        base_rank = percentile_ranks(risks[name])
        ens = {q: (cvss_rank.get(q, 0.0) + base_rank.get(q, 0.0)) / 2.0
               for q in labeled if q in cvss_rank and q in base_rank}
        auc, rho, kp, n = bench(f'cvss+{name}', ens)
        print(f'  cvss+{name:6s} KEV AUC={auc:.4f}  CVSS rho={rho:+.4f}  (kev_pos={kp}/{n})')

    out = {name: risks[name] for name in risks}
    out_path = os.path.join(kg_dir, 'orthogonality_probe2.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'))
    print(f'[probe2] wrote {out_path}')


if __name__ == '__main__':
    main()
