"""case_study_chains.py — paper qualitative: readable attack chains for KEV CVEs.

For each KEV CVE, prints its ProRMPI attack-centrality story:
  1. top-3 completion CWEs (the inductive completion step),
  2. best downstream chain under converged stateful tau (the centrality step),
  3. final risk score + CVSS.

Node labels resolved from the raw ontologies (CWE/CAPEC XML + enterprise-attack.json).
Reads the already-computed global tau (attack_centrality_<abl>_<expri>.json), so it
does NOT re-accumulate. Best path is recomputed with a fresh rng (representative).

Usage (from security_kg/):
  python case_study_chains.py --kg security_kg_ind \
      --expri security_kg_ind_RMPI_proto_hard_cwe --ablation C
"""

import json
import os
import sys
import random
import xml.etree.ElementTree as ET

import aco_completion as ac
import attack_centrality as atk


def cwe_names(raw_dir):
    out = {}
    try:
        for _, el in ET.iterparse(os.path.join(raw_dir, 'cwec_v4.20.xml'),
                                  events=('end',)):
            if el.tag.endswith('Weakness'):
                out[el.get('ID')] = el.get('Name')
                el.clear()
    except Exception as e:
        print(f'[case] cwe names skipped: {e}', file=sys.stderr)
    return out


def capec_names(raw_dir):
    out = {}
    try:
        for _, el in ET.iterparse(os.path.join(raw_dir, 'capec_latest.xml'),
                                  events=('end',)):
            if el.tag.endswith('Attack_Pattern'):
                out[el.get('ID')] = el.get('Name')
                el.clear()
    except Exception as e:
        print(f'[case] capec names skipped: {e}', file=sys.stderr)
    return out


def technique_names(raw_dir):
    out = {}
    try:
        with open(os.path.join(raw_dir, 'enterprise-attack.json'), encoding='utf-8') as f:
            data = json.load(f)
        for o in data.get('objects', []):
            if o.get('type') != 'attack-pattern':
                continue
            for ref in o.get('external_references', []):
                eid = ref.get('external_id', '')
                if eid.startswith('T'):
                    out[eid] = o.get('name', '')
    except Exception as e:
        print(f'[case] technique names skipped: {e}', file=sys.stderr)
    return out


def label(v, cwe, capec, tech):
    t = ac.node_type(v)
    if t == 'cwe':
        return f'{v} ({cwe.get(v.split("-", 1)[1], "?")})'
    if t == 'capec':
        return f'{v} ({capec.get(v.split("-", 1)[1], "?")})'
    if t == 'technique':
        base = v.split('.')[0]
        return f'{v} ({tech.get(base, "?")})'
    return v  # tactic:xxx already readable


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
    raw_dir = os.path.join(ac.DATA_DIR, 'security_kg', 'raw')
    adj, degrees = ac.load_train_graph(kg_dir)
    cands = ac.load_completion_cands(kg_dir, expri)
    sc_dir = ac.sidecar_dir(kg_dir)
    kev = json.load(open(os.path.join(sc_dir, 'kev_labels.json'), encoding='utf-8'))
    cvss = json.load(open(os.path.join(sc_dir, 'cvss_scores.json'), encoding='utf-8'))
    global_tau = json.load(open(os.path.join(
        kg_dir, f'attack_centrality_{ablation}_{expri}.json'), encoding='utf-8'))
    risk = json.load(open(os.path.join(
        kg_dir, f'aco_risk_stateful_{ablation}_{expri}.json'), encoding='utf-8'))

    print(f'[case] resolving ontology names ...')
    cwe = cwe_names(raw_dir)
    capec = capec_names(raw_dir)
    tech = technique_names(raw_dir)
    print(f'[case] cwe={len(cwe)} capec={len(capec)} tech={len(tech)}')

    params = dict(ac.PARAMS)
    params['rho_q'] = 0.0
    params['alpha'] = 0.0
    rng = random.Random(params['seed'])
    use_conf_eta = ablation in ('A1', 'C')
    use_conf_fitness = ablation in ('B', 'C')

    kev_cves = [c for c in sorted(kev) if c in cands]
    print(f'[case] {len(kev_cves)} KEV CVEs with candidates\n')
    for q in kev_cves:
        cc = ac.softmax_conf(cands[q][:top_k])
        if not cc:
            continue
        path, score = atk.score_query(q, cc, global_tau, adj, degrees, params,
                                      rng, use_conf_eta, use_conf_fitness)
        top3 = ' | '.join(f'{c}({s:.2f})' for c, s in cc[:3])
        chain = '  ->  '.join(
            f"{label(v, cwe, capec, tech)} [tau={global_tau.get(v, params['tau0']):.0f}]"
            for v in path[1:])
        print(f'CVE {q}   CVSS={cvss.get(q, "?")}   risk={risk.get(q, "?")}')
        print(f'   top-3 CWE: {top3}')
        print(f'   chain: {q}  ->  {chain}')
        print()

    out = {q: {'path': [label(v, cwe, capec, tech) for v in
                        atk.score_query(q, ac.softmax_conf(cands[q][:top_k]),
                                        global_tau, adj, degrees, params, rng,
                                        use_conf_eta, use_conf_fitness)[0][1:]],
               'risk': risk.get(q), 'cvss': cvss.get(q)}
           for q in kev_cves}
    out_path = os.path.join(kg_dir, f'case_study_{ablation}_{expri}.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[case] wrote {out_path}')


if __name__ == '__main__':
    main()
