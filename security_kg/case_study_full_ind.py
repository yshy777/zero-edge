"""case_study_full_ind.py — paper qualitative: per-CVE case-study rows for the
production (full-ind, 2+SC) pipeline.

Re-runs the EXACT risk_profile cands-mode pipeline (same loaders, same SEED=0)
to recover the per-CVE out-of-fold profile scores that risk_profile.py computes
but does not persist, verifies the aggregates bit-match the published
risk_profile_cands.json, then dumps per-CVE rows (top-1/top-3 predicted CWEs,
GT CWE + rank, CVSS, n_prod, cwe_rate, chain with ontology names, OOF score,
percentile, bucket decile) for the case-study section.

Usage (from repo root):
  python security_kg/case_study_full_ind.py
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

import aco_completion as ac
import risk_profile as rp

SC_DUMP = os.path.join(ac.DATA_DIR, 'security_kg_full_ind',
                       'completion_cands_security_kg_full_ind_RMPI_proto_hard_sc.json')
OUT = os.path.join(ac.DATA_DIR, 'security_kg_full_ind', 'case_study_full_ind.json')


def ontology_names(raw_dir):
    cwe, capec, tech = {}, {}, {}
    try:
        for _, el in ET.iterparse(os.path.join(raw_dir, 'cwec_v4.20.xml'),
                                  events=('end',)):
            if el.tag.endswith('Weakness'):
                cwe[el.get('ID')] = el.get('Name')
                el.clear()
    except Exception as e:
        print(f'[case] cwe names skipped: {e}', file=sys.stderr)
    try:
        for _, el in ET.iterparse(os.path.join(raw_dir, 'capec_latest.xml'),
                                  events=('end',)):
            if el.tag.endswith('Attack_Pattern'):
                capec[el.get('ID')] = el.get('Name')
                el.clear()
    except Exception as e:
        print(f'[case] capec names skipped: {e}', file=sys.stderr)
    try:
        with open(os.path.join(raw_dir, 'enterprise-attack.json'), encoding='utf-8') as f:
            data = json.load(f)
        for o in data.get('objects', []):
            if o.get('type') != 'attack-pattern':
                continue
            for ref in o.get('external_references', []):
                eid = ref.get('external_id', '')
                if eid.startswith('T'):
                    tech[eid] = o.get('name', '')
    except Exception as e:
        print(f'[case] technique names skipped: {e}', file=sys.stderr)
    return cwe, capec, tech


def label(v, cwe, capec, tech):
    t = ac.node_type(v)
    if t == 'cwe':
        return f'{v} ({cwe.get(v.split("-", 1)[1], "?")})'
    if t == 'capec':
        return f'{v} ({capec.get(v.split("-", 1)[1], "?")})'
    if t == 'technique':
        return f'{v} ({tech.get(v.split(".")[0], "?")})'
    return v


def main():
    kg_dir = os.path.join(ac.DATA_DIR, 'security_kg_full_ind')
    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))
    saved = json.load(open(os.path.join(kg_dir, 'risk_profile_cands.json'), encoding='utf-8'))

    print('[case] cwe_rate table (train) ...')
    cwe_rate, prior = rp.cwe_rate_table(kg_dir, kev)
    print(f'[case] train CWEs={len(cwe_rate)} prior={prior:.5f}')

    print('[case] n_prod (raw jsonl) ...')
    n_prod = rp.load_raw_products()

    print('[case] attack graph (train, 4 rels) ...')
    G = rp.build_attack_graph(kg_dir)
    indeg = {v: d for v, d in G.in_degree()}

    unseen = ac.unseen_cves(kg_dir)
    labeled = [q for q in unseen if q in cvss]
    y = {q: 1 if q in kev else 0 for q in labeled}

    cands = rp.load_cands(SC_DUMP)
    raw_dump = json.load(open(SC_DUMP, encoding='utf-8'))
    score_of = {q: {c: s for c, s in raw_dump[q]} for q in raw_dump}
    cwes_of = {q: [cands[q][0]] for q in labeled if q in cands and cands[q]}
    top3 = {q: cands[q][:3] for q in labeled if q in cands and len(cands[q]) >= 3}
    labeled = [q for q in labeled if q in cwes_of]
    y = {q: 1 if q in kev else 0 for q in labeled}
    print(f'[case] universe={len(labeled)} kev={sum(y.values())}')

    feats = {
        'cvss': {q: cvss[q] for q in labeled},
        'cwe_rate': {q: max(cwe_rate.get(c, 0.0) for c in cs)
                     for q, cs in cwes_of.items()},
        'cwe_rate_top3': {q: sum(cwe_rate.get(c, 0.0) for c in cs) / len(cs)
                          for q, cs in top3.items()},
        'n_prod': {q: float(n_prod[q]) for q in labeled if q in n_prod},
        'chain_heat': {q: max(rp.chain_heat(c, G, indeg) for c in cs)
                       for q, cs in cwes_of.items()},
    }
    ranks = {name: rp.percentile_ranks(f) for name, f in feats.items()}

    # ---- verify single/combos against the saved artifact
    ok = True
    for name in ['cvss', 'cwe_rate', 'cwe_rate_top3', 'n_prod', 'chain_heat']:
        qs = [q for q in labeled if q in feats[name]]
        a, se = rp.auc_se([y[q] for q in qs], [feats[name][q] for q in qs])
        s = saved['single'].get(name)
        if s is None or abs(a - s['auc']) > 5e-4 or abs(se - s['se']) > 5e-4:
            ok = False
            print(f'[case] MISMATCH single {name}: recomputed {a:.4f}±{se:.4f} '
                  f'vs saved {s}')
        else:
            print(f'[case] single {name} OK ({a:.4f}±{se:.4f}, kev={s["kev"]}/{s["n"]})')

    base = ['cvss', 'cwe_rate', 'n_prod', 'chain_heat']
    pool = [q for q in labeled
            if all(q in feats[f] for f in base) and q in n_prod]
    X = {f: np.array([ranks[f][q] for q in pool]) for f in base}
    Y = np.array([y[q] for q in pool])
    subsets = [
        ('cvss', ['cvss']),
        ('cvss+cwe_rate', ['cvss', 'cwe_rate']),
        ('cvss+cwe_rate+n_prod', ['cvss', 'cwe_rate', 'n_prod']),
        ('cvss+cwe_rate+n_prod+chain', base),
    ]
    oof_full = None
    for label_, fs in subsets:
        oof = rp.logistic_cv(np.column_stack([X[f] for f in fs]), Y)
        a, se = rp.auc_se(Y, oof)
        s = saved['combos'].get(label_)
        if s is None or abs(a - s['auc']) > 5e-4 or abs(se - s['se']) > 5e-4:
            ok = False
            print(f'[case] MISMATCH combo {label_}: recomputed {a:.4f}±{se:.4f} '
                  f'vs saved {s}')
        else:
            print(f'[case] combo {label_} OK ({a:.4f}±{se:.4f})')
        if label_ == 'cvss+cwe_rate+n_prod+chain':
            oof_full = {q: float(v) for q, v in zip(pool, oof)}
    rng = np.random.RandomState(rp.SEED)
    noise = rng.permutation(X['cwe_rate'].copy())
    oof_n = rp.logistic_cv(np.column_stack([X['cvss'], noise, X['n_prod']]), Y)
    a_n, se_n = rp.auc_se(Y, oof_n)
    if abs(a_n - saved['combos']['cvss+cwe_noise+n_prod']['auc']) > 5e-4:
        ok = False
        print(f'[case] MISMATCH noise combo: recomputed {a_n:.4f} vs saved')
    else:
        print(f'[case] noise combo OK ({a_n:.4f})')
    print(f'[case] verification: {"PASS" if ok else "FAIL"}')

    # ---- bucket decile membership (gate-2 protocol)
    res = rp.bucket_eval(cvss, y, oof_full)
    (a, se), rows, deciles = res
    print(f'[case] gate2 pooled {a:.4f}±{se:.4f} (saved '
          f'{saved["bucket"]["full_combo"]["pooled_auc"]})')
    bucket_of = {}
    for row in rows:
        b = row['bucket_cvss']
        qs = [q for q in pool if q in cvss and int(cvss[q] / rp.BUCKET_W) == int(b / rp.BUCKET_W)]
        pr = rp.percentile_ranks({q: oof_full[q] for q in qs})
        order = sorted(qs, key=lambda q: pr[q])
        for d in range(10):
            for q in order[len(order) * d // 10: len(order) * (d + 1) // 10]:
                bucket_of[q] = (b, d + 1)

    # ---- GT CWE + rank in the dump list
    gt = rp.gt_cwes(kg_dir)
    print(f'[case] gt cwes={len(gt)}')

    # ---- ontology names
    raw_dir = os.path.join(ac.DATA_DIR, 'security_kg', 'raw')
    cwe, capec, tech = ontology_names(raw_dir)
    print(f'[case] names: cwe={len(cwe)} capec={len(capec)} tech={len(tech)}')

    # ---- per-CVE rows
    pct = rp.percentile_ranks(oof_full)
    rows_out = {}
    for q in pool:
        lst = cands[q]
        top1 = lst[0]
        gt_lst = gt.get(q, [])
        rank = next((i + 1 for i, c in enumerate(lst) if c in gt_lst), None)
        chain = rp.chain_from(top1, G, indeg)
        rows_out[q] = {
            'kev': bool(y[q]),
            'cvss': cvss[q],
            'n_prod': int(n_prod[q]),
            'top1': top1,
            'top3': lst[:3],
            'top1_score': score_of[q].get(top1),
            'top3_scores': [score_of[q].get(c) for c in lst[:3]],
            'gt_cwe': gt_lst,
            'gt_rank': rank,
            'cwe_rate_top1': cwe_rate.get(top1, 0.0),
            'cwe_rate_top3': feats['cwe_rate_top3'].get(q),
            'chain_heat': feats['chain_heat'].get(q),
            'chain': [label(v, cwe, capec, tech) for v in chain],
            'oof': oof_full[q],
            'oof_pct': pct[q],
            'bucket': bucket_of.get(q),
            'kev_meta': kev.get(q),
        }
    json.dump(rows_out, open(OUT, 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print(f'[case] wrote {OUT} ({len(rows_out)} rows)')

    # ---- shortlist KEV rows for the narrative
    print('\n=== KEV rows (sorted by oof desc) ===')
    kev_rows = sorted((r for q, r in rows_out.items() if r['kev']),
                      key=lambda r: -r['oof'])
    for q, r in [(q_, rows_out[q_]) for q_ in
                 sorted(rows_out, key=lambda q_: -rows_out[q_]['oof']) if rows_out[q_]['kev']][:15]:
        m = r['kev_meta'] or {}
        print(f'{q}  cvss={r["cvss"]}  n_prod={r["n_prod"]}  oof={r["oof"]:.4f} '
              f'pct={r["oof_pct"]:.3f}  bucket={r["bucket"]}')
        print(f'   top1={r["top1"]} rate={r["cwe_rate_top1"]:.4f}  '
              f'top3={r["top3"]}  gt={r["gt_cwe"]} gt_rank={r["gt_rank"]}')
        print(f'   chain: {" -> ".join(r["chain"])}')
        print(f'   kev: {m.get("dateAdded")} {m.get("vendor")} {m.get("product")}')


if __name__ == '__main__':
    main()
