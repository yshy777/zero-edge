"""build_kg.py — Assemble the security knowledge graph from official metadata.

Sources (already downloaded to data/security_kg/raw/ by download.py / nvd.py):
  - capec_latest.xml         CAPEC attack patterns (+ ATT&CK taxonomy mappings)
  - cwec_v4.20.xml           CWE weaknesses (+ Related Attack Patterns -> CAPEC)
  - enterprise-attack.json   MITRE ATT&CK techniques/tactics (STIX 2.x)
  - kev.json                 CISA Known-Exploited Vulnerabilities (ground truth)
  - nvd_cve_cpe.jsonl        NVD CVEs (CWE mappings + CVSS + vulnerable CPEs)

Entity/relation vocabulary (string IDs, RMPI `h r t` triple format):
  entities: CVE-xxxx-xxxxx, CWE-xxx, CAPEC-xxx, Txxxx[.xxx], tactic:<phase>,
            product:<vendor>:<product>
  relations:
    has_weakness  CVE      -> CWE        (NVD + KEV)
    exploited_by  CWE      -> CAPEC      (CWE Related_Attack_Patterns)
    maps_to       CAPEC    -> technique  (CAPEC Taxonomy_Mappings @ ATTACK)
    has_tactic    technique-> tactic     (ATT&CK kill_chain_phases)
    hasVulnerability product -> CVE      (NVD vulnerable CPEs; gives a CVE its
                                          structural identity via product family)

Outputs:
  data/security_kg/{train,valid,test}.txt     transductive split (random 80/10/10)
  data/security_kg_ind/{train,test}.txt       inductive split (disjoint CVEs by time)
  data/security_kg/kev_labels.json            {cve: {exploited, dateAdded}}
  data/security_kg/cvss_scores.json           {cve: float}
"""

import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'security_kg', 'raw'))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'security_kg'))
OUT_IND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'security_kg_ind'))

CAPEC_NS = '{http://capec.mitre.org/capec-3}'
CWE_NS = '{http://cwe.mitre.org/cwe-7}'

REL_WEAKNESS = 'has_weakness'
REL_EXPLOIT = 'exploited_by'
REL_MAP = 'maps_to'
REL_TACTIC = 'has_tactic'
REL_VULN = 'hasVulnerability'


# --------------------------------------------------------------------------- #
# Source parsers -> list of (head, relation, tail) tuples
# --------------------------------------------------------------------------- #

def parse_capec():
    """CAPEC -> ATT&CK (maps_to), from Taxonomy_Mappings where Taxonomy_Name=ATTACK."""
    triples = []
    root = ET.parse(os.path.join(RAW, 'capec_latest.xml')).getroot()
    for ap in root.findall(f'.//{CAPEC_NS}Attack_Pattern'):
        capec_id = f"CAPEC-{ap.attrib['ID']}"
        for tm in ap.findall(f'.//{CAPEC_NS}Taxonomy_Mappings/{CAPEC_NS}Taxonomy_Mapping'):
            if tm.attrib.get('Taxonomy_Name') != 'ATTACK':
                continue
            for e in tm.findall(f'{CAPEC_NS}Entry_ID'):
                if e.text and e.text.strip():
                    triples.append((capec_id, REL_MAP, f"T{e.text.strip()}"))
    return triples


def parse_cwe():
    """CWE -> CAPEC (exploited_by), from Related_Attack_Patterns."""
    triples = []
    root = ET.parse(os.path.join(RAW, 'cwec_v4.20.xml')).getroot()
    for w in root.findall(f'.//{CWE_NS}Weakness'):
        cwe_id = f"CWE-{w.attrib['ID']}"
        for rap in w.findall(f'.//{CWE_NS}Related_Attack_Patterns/{CWE_NS}Related_Attack_Pattern'):
            cap_id = rap.attrib.get('CAPEC_ID')
            if cap_id:
                triples.append((cwe_id, REL_EXPLOIT, f"CAPEC-{cap_id}"))
    return triples


def parse_attack():
    """technique -> tactic (has_tactic). Sub-techniques inherit the parent's tactic."""
    triples = []
    a = json.load(open(os.path.join(RAW, 'enterprise-attack.json'), encoding='utf-8'))
    tech_to_tactics = {}
    for o in a['objects']:
        if o['type'] != 'attack-pattern':
            continue
        if o.get('revoked') or o.get('x_mitre_deprecated'):
            continue
        ext_id = None
        for r in o.get('external_references', []):
            if r.get('source_name') == 'mitre-attack':
                ext_id = r.get('external_id')
                break
        if not ext_id:
            continue
        phases = [kp.get('phase_name') for kp in o.get('kill_chain_phases', [])
                  if kp.get('kill_chain_name') == 'mitre-attack']
        if phases:
            tech_to_tactics[ext_id] = phases

    for tid, tactics in tech_to_tactics.items():
        for tac in tactics:
            triples.append((tid, REL_TACTIC, f"tactic:{tac}"))
    # Sub-techniques: no kill_chain_phases in STIX; inherit from parent T-code.
    # The parser above only emits top-level; sub-techniques carry no tactic of
    # their own, so CAPEC -> sub-technique -> parent tactic is resolved by the
    # consumer. We emit the parent tactic once per top-level technique here.
    return triples


def valid_technique_ids():
    """Set of non-deprecated ATT&CK technique IDs (T-codes), top-level + sub."""
    a = json.load(open(os.path.join(RAW, 'enterprise-attack.json'), encoding='utf-8'))
    ids = set()
    for o in a['objects']:
        if o['type'] != 'attack-pattern':
            continue
        if o.get('revoked') or o.get('x_mitre_deprecated'):
            continue
        for r in o.get('external_references', []):
            if r.get('source_name') == 'mitre-attack':
                ids.add(r.get('external_id'))
    return ids


def resolve_techniques(triples, valid_ids):
    """Drop `maps_to` links whose technique was deprecated in ATT&CK, falling
    back to the parent technique when it still exists."""
    out = []
    dropped = 0
    for h, r, t in triples:
        if r == REL_MAP and t not in valid_ids:
            parent = t.split('.')[0]
            if parent in valid_ids:
                out.append((h, r, parent))
            else:
                dropped += 1
        else:
            out.append((h, r, t))
    return out, dropped


def subsample_cves(triples, max_cves, seed=41504):
    """Deterministically cap the number of CVEs (heads of has_weakness links),
    keeping the full CWE/CAPEC/ATT&CK backbone intact. Used for fast iteration.
    `hasVulnerability` edges (CVE is the tail) are filtered by the same keep set."""
    if max_cves is None:
        return triples
    rng = random.Random(seed)
    weakness = [t for t in triples if t[1] == REL_WEAKNESS]
    backbone = [t for t in triples if t[1] != REL_WEAKNESS]
    cves = sorted({t[0] for t in weakness})
    if len(cves) <= max_cves:
        return triples
    keep = set(rng.sample(cves, max_cves))
    return backbone + [t for t in weakness if t[0] in keep]


def _nvd_path():
    """New canonical NVD source includes vulnerable CPEs; fall back to the
    older product-less dump if the re-pull has not produced one yet."""
    for name in ('nvd_cve_cpe.jsonl', 'nvd_cve.jsonl'):
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            return path
    return os.path.join(RAW, 'nvd_cve_cpe.jsonl')


def parse_nvd():
    """CVE -> CWE (has_weakness) + CVSS scores + vulnerable products.

    Returns (triples, cvss, products) where `products` maps CVE -> sorted
    [vendor:product] strings; each becomes a `hasVulnerability` head, giving the
    CVE structural identity through its product family (see REL_VULN)."""
    triples = []
    cvss = {}
    products = {}
    path = _nvd_path()
    if not os.path.exists(path):
        return triples, cvss, products
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cve = rec.get('cve')
            if not cve:
                continue
            if rec.get('cvss') is not None:
                cvss[cve] = rec['cvss']
            for cwe in rec.get('cwe', []):
                triples.append((cve, REL_WEAKNESS, cwe))
            prods = rec.get('products', [])
            if prods:
                products[cve] = prods
    return triples, cvss, products


def parse_kev():
    """KEV: supplement CVE -> CWE (has_weakness) + exploited labels."""
    triples = []
    labels = {}
    kev = json.load(open(os.path.join(RAW, 'kev.json'), encoding='utf-8'))
    for v in kev['vulnerabilities']:
        cve = v.get('cveID')
        if not cve:
            continue
        labels[cve] = {
            'exploited': True,
            'dateAdded': v.get('dateAdded'),
            'product': v.get('product'),
            'vendor': v.get('vendorProject'),
        }
        for cwe in v.get('cwes', []) or []:
            triples.append((cve, REL_WEAKNESS, cwe))
    return triples, labels


FP_DIM = 32  # hashing buckets for the product-name fingerprint


def product_fingerprint(products):
    """CVE product-family fingerprint: [log1p(#products)] + log1p(hash counts).

    A fixed 33-dim vector encoding *which* vulnerable products a CVE affects
    (CRC32 hashing of product names into FP_DIM buckets) and *how many*. Used as
    the CVE node input feature (scheme 3) instead of materialising product
    entities as graph edges, so GNN subgraphs stay small while an unseen CVE
    still gets a distinct, product-driven identity."""
    n = len(products)
    counts = [0] * FP_DIM
    for p in products:
        counts[zlib.crc32(p.encode('utf-8')) % FP_DIM] += 1
    return [math.log1p(n)] + [math.log1p(c) for c in counts]


# --------------------------------------------------------------------------- #
# Splitting + writing
# --------------------------------------------------------------------------- #

def dedupe(triples):
    return sorted(set(triples))


def write_triples(path, triples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for h, r, t in triples:
            f.write(f'{h}\t{r}\t{t}\n')


def split_transductive(all_triples, seed=41504, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    triples = list(all_triples)
    rng.shuffle(triples)
    n = len(triples)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    return triples[:n_train], triples[n_train:n_train + n_valid], triples[n_train + n_valid:]


def split_inductive(all_triples, cve_published, ratios=(0.7, 0.1, 0.2), seed=41504):
    """Temporal split by CVE publication date: train = oldest 70% of CVEs,
    valid = middle 10%, test = newest 20% (unseen). Backbone is always in train.
    The valid/test CVEs are disjoint from train CVEs -> fully-inductive."""
    has_weakness = [t for t in all_triples if t[1] == REL_WEAKNESS]
    backbone = [t for t in all_triples if t[1] != REL_WEAKNESS]

    cves = sorted(cve_published.keys(), key=lambda c: cve_published[c])
    n = len(cves)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_cves = set(cves[:n_train])
    valid_cves = set(cves[n_train:n_train + n_valid])
    test_cves = set(cves[n_train + n_valid:])

    train = list(backbone)
    valid = []
    test = []
    for h, r, t in has_weakness:
        if h in test_cves:
            test.append((h, r, t))
        elif h in valid_cves:
            valid.append((h, r, t))
        else:
            train.append((h, r, t))
    return train, valid, test


def main():
    max_cves = None
    if '--max-cves' in sys.argv:
        max_cves = int(sys.argv[sys.argv.index('--max-cves') + 1])

    # Optional: redirect outputs to data/<name>/ and data/<name>_ind/ so a
    # full-scale build can live alongside the 15k validation KG without
    # overwriting it.
    global OUT, OUT_IND
    if '--out-name' in sys.argv:
        name = sys.argv[sys.argv.index('--out-name') + 1]
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        OUT = os.path.join(data_dir, name)
        OUT_IND = os.path.join(data_dir, name + '_ind')

    triples = dedupe(
        parse_capec() + parse_cwe() + parse_attack()
    )
    nvd_triples, cvss, nvd_products = parse_nvd()
    kev_triples, kev_labels = parse_kev()

    # Scheme 3: product identity is injected as a per-CVE node feature
    # (product_fingerprint below), NOT as `product -> CVE` graph edges.
    # Materialising products as entities exploded the 2-hop enclosing subgraphs
    # (~40 product neighbours per CVE), so they are dropped from the graph here.
    triples = dedupe(triples + nvd_triples + kev_triples)

    # Resolve CAPEC->technique links whose sub-technique was deprecated.
    triples, dropped = resolve_techniques(triples, valid_technique_ids())
    print(f'[build_kg] dropped {dropped} stale CAPEC->technique links')

    if max_cves is not None:
        triples = subsample_cves(triples, max_cves)
        print(f'[build_kg] subsampled to <= {max_cves} CVEs -> {len(triples)} triples')

    # Scheme 3: per-CVE product-family fingerprint (node feature). Only CVEs kept
    # in the KG get one; CVEs with no NVD CPE data have no entry (zero vector at
    # runtime). Written to both split dirs so train/valid/test share one source.
    fingerprint = {}
    for cve in sorted({t[0] for t in triples if t[1] == REL_WEAKNESS}):
        if cve in nvd_products:
            fingerprint[cve] = product_fingerprint(nvd_products[cve])
    for d in (OUT, OUT_IND):
        json.dump(fingerprint, open(os.path.join(d, 'product_fingerprint.json'), 'w', encoding='utf-8'))

    # Published dates for inductive CVE split (from NVD `published`).
    cve_published = {}
    nvd_path = _nvd_path()
    if os.path.exists(nvd_path):
        with open(nvd_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get('cve') and rec.get('published'):
                    cve_published[rec['cve']] = rec['published']

    # Transductive split.
    train, valid, test = split_transductive(triples)
    for name, part in (('train', train), ('valid', valid), ('test', test)):
        write_triples(os.path.join(OUT, f'{name}.txt'), part)

    # Inductive split (CVE-disjoint, temporal).
    train_ind, valid_ind, test_ind = split_inductive(triples, cve_published)
    for name, part in (('train', train_ind), ('valid', valid_ind), ('test', test_ind)):
        write_triples(os.path.join(OUT_IND, f'{name}.txt'), part)

    # Sidecars for the risk-ranking benchmark (Stage 3).
    json.dump(kev_labels, open(os.path.join(OUT, 'kev_labels.json'), 'w', encoding='utf-8'))
    json.dump(cvss, open(os.path.join(OUT, 'cvss_scores.json'), 'w', encoding='utf-8'))

    print(f'[build_kg] total triples: {len(triples)}')
    print(f'  product fingerprints: {len(fingerprint)} CVEs (dim {FP_DIM + 1})')
    print(f'  transductive  train={len(train)} valid={len(valid)} test={len(test)}')
    print(f'  inductive     train={len(train_ind)} valid={len(valid_ind)} test={len(test_ind)}')
    print(f'  KEV labels: {len(kev_labels)} | CVSS scores: {len(cvss)} | '
          f'CVE dates: {len(cve_published)}')


if __name__ == '__main__':
    main()
