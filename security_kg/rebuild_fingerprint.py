"""rebuild_fingerprint.py — regenerate product_fingerprint.json for the full-scale KG.

The security_kg_full / security_kg_full_ind datasets were built before the
Scheme-3 product fingerprint was emitted, so their models ran with prod_dim=0
(no head-specific side channel) and the completion dump collapsed to a single
CWE-79 answer for every unseen CVE.  This script regenerates the fingerprint
WITHOUT rebuilding the KG: it re-reads the same raw NVD CPE dump that build_kg
uses and covers every has_weakness head in the existing splits.

Output (same semantics as build_kg main()):
  data/security_kg_full/product_fingerprint.json
  data/security_kg_full_ind/product_fingerprint.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kg import parse_nvd, product_fingerprint, REL_WEAKNESS  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
TARGETS = ('security_kg_full', 'security_kg_full_ind')
SPLITS = ('train', 'valid', 'test')


def collect_cves():
    """Every CVE that is a has_weakness head anywhere in the full-scale splits."""
    cves = set()
    for name in TARGETS:
        for part in SPLITS:
            path = os.path.join(BASE, name, f'{part}.txt')
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) == 3 and parts[1] == REL_WEAKNESS:
                        cves.add(parts[0])
    return cves


def main():
    _, _, nvd_products = parse_nvd()
    cves = collect_cves()
    fingerprint = {}
    for cve in sorted(cves):
        if cve in nvd_products:
            fingerprint[cve] = product_fingerprint(nvd_products[cve])
    covered = len(fingerprint)
    missing = len(cves) - covered
    print(f'[rebuild_fingerprint] total has_weakness heads = {len(cves)}')
    print(f'[rebuild_fingerprint] NVD CPE products map     = {len(nvd_products)} CVEs')
    print(f'[rebuild_fingerprint] covered (non-zero fp)    = {covered}')
    print(f'[rebuild_fingerprint] missing (zero fp)        = {missing} '
          f'({100.0 * missing / max(len(cves), 1):.2f}%)')
    for name in TARGETS:
        path = os.path.join(BASE, name, 'product_fingerprint.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f)
        print(f'[rebuild_fingerprint] wrote {path} ({len(fingerprint)} CVEs)')


if __name__ == '__main__':
    main()
