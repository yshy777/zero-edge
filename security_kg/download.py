"""download.py — Pull official structured metadata for the security KG.

Sources (all official, metadata only):
  - CISA KEV            (known-exploited vulnerabilities, ground truth for risk)
  - MITRE ATT&CK        (enterprise techniques/tactics, STIX 2.x)
  - MITRE CAPEC         (attack patterns, XML)
  - MITRE CWE           (weakness enumeration, XML zip)

NVD is handled separately (API 2.0, rate-limited) in nvd.py.

Everything lands in data/security_kg/raw/. Downloads are idempotent: an
existing file is skipped.
"""

import os
import shutil
import urllib.request

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'security_kg', 'raw'))

SOURCES = {
    'kev.json': 'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json',
    'enterprise-attack.json': 'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
    'capec_latest.xml': 'https://capec.mitre.org/data/xml/capec_latest.xml',
    'cwec_latest.xml.zip': 'https://cwe.mitre.org/data/xml/cwec_latest.xml.zip',
}

UA = 'RMPI-SecKG/1.0 (academic research; contact: local)'


def download(name, url):
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'[skip] {name} already present ({os.path.getsize(dest)} bytes)')
        return dest
    print(f'[get ] {name}\n      <- {url}')
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)
    print(f'[ok  ] {name} ({os.path.getsize(dest)} bytes)')
    return dest


if __name__ == '__main__':
    for name, url in SOURCES.items():
        try:
            download(name, url)
        except Exception as e:  # noqa: BLE001 — report and keep going
            print(f'[fail] {name}: {e}')
