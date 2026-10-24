"""nvd.py — Pull CVE records from NVD API 2.0 (no API key: 5 req / 30 s).

NVD API 2.0 caps any single date-range query at 120 consecutive days, so we
slice the requested window into 120-day chunks and paginate each chunk at
resultsPerPage=2000. We extract a compact JSONL of
    {cve, published, cwe: [..], cvss: float|None}
rather than the full record, to keep the raw footprint small.

Usage:  python security_kg/nvd.py [START_DATE] [END_DATE]
        dates in ISO yyyy-mm-dd; defaults 2021-01-01 .. today.
"""

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'security_kg', 'raw'))
API = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
UA = 'RMPI-SecKG/1.0 (academic research)'
PAGE = 2000
CHUNK_DAYS = 120
SLEEP = 6.5      # seconds between requests (5 req / 30 s budget)
MAX_RETRIES = 5


def _retry(attempt, reason):
    wait = SLEEP * (attempt + 1)
    print(f'[nvd] {reason}, retry in {wait:.0f}s', flush=True)
    time.sleep(wait)


def _get(params):
    url = API + '?' + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 403/429 = rate limit, 503 = server busy -> back off and retry.
            if e.code in (403, 429, 503):
                _retry(attempt, f'HTTP {e.code}')
                continue
            raise
        except json.JSONDecodeError:
            _retry(attempt, 'JSONDecodeError')
            continue
        except http.client.HTTPException as e:
            # IncompleteRead / BadStatusLine: response body truncated mid-read.
            _retry(attempt, type(e).__name__)
            continue
        except OSError as e:
            # urllib.error.URLError, ConnectionError, RemoteDisconnected and
            # TimeoutError are all OSError subclasses -> transient network.
            _retry(attempt, type(e).__name__)
            continue
    raise RuntimeError(f'nvd: gave up after {MAX_RETRIES} retries on {url}')


def _extract_products(cve):
    """Vulnerable CPEs -> vendor:product (CPE 2.3 fields 3:4), deduped.

    Normalising to vendor:product (dropping version) is what gives a CVE its
    *structural* identity in the KG: CVEs that share a product family are
    reachable from one another through the `hasVulnerability` relation, so an
    unseen CVE stops being an isolated node. Full CPE (with version) would
    over-fragment (every version is a distinct entity -> CVE stays isolated).
    """
    products = set()
    for cfg in (cve.get('configurations') or []):
        for node in (cfg.get('nodes') or []):
            for cm in (node.get('cpeMatch') or []):
                if not cm.get('vulnerable'):
                    continue
                crit = cm.get('criteria') or ''
                parts = crit.split(':')
                # cpe:2.3:<part>:<vendor>:<product>:<version>:...
                if len(parts) >= 5 and parts[0] == 'cpe' and parts[1] == '2.3':
                    vendor, product = parts[3], parts[4]
                    if vendor and product and vendor != '*' and product != '*':
                        products.add(f'{vendor}:{product}')
    return sorted(products)


def _extract(cve):
    cwe_ids = []
    for w in (cve.get('weaknesses') or []):
        for d in (w.get('description') or []):
            v = d.get('value', '')
            if v.startswith('CWE-'):
                cwe_ids.append(v)
    cwe_ids = sorted(set(cwe_ids))

    score = None
    metrics = cve.get('metrics') or {}
    for key in ('cvssMetricV40', 'cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2'):
        for m in (metrics.get(key) or []):
            s = ((m.get('cvssData') or {}).get('baseScore'))
            if s is not None:
                score = s
                break
        if score is not None:
            break

    return {'cve': cve.get('id'), 'published': cve.get('published'),
            'cwe': cwe_ids, 'cvss': score, 'products': _extract_products(cve)}


def pull(start, end, out_path, mode='w'):
    out = open(out_path, mode, encoding='utf-8')
    n = 0
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        start_idx = 0
        while True:
            params = {
                'pubStartDate': cur.strftime('%Y-%m-%dT%H:%M:%S.000'),
                'pubEndDate': chunk_end.strftime('%Y-%m-%dT%H:%M:%S.000'),
                'resultsPerPage': PAGE,
                'startIndex': start_idx,
            }
            d = _get(params)
            vulns = d.get('vulnerabilities', [])
            total = d.get('totalResults', 0)
            for v in vulns:
                out.write(json.dumps(_extract(v.get('cve', {}))) + '\n')
                n += 1
            start_idx += len(vulns)
            print(f'[nvd] {cur.date()}..{chunk_end.date()} '
                  f'page_start={start_idx}/{total} written={n}', flush=True)
            if not vulns or start_idx >= total:
                break
            time.sleep(SLEEP)
        cur = chunk_end
        time.sleep(SLEEP)
    out.close()
    print(f'[nvd] DONE: {n} records -> {out_path}')


if __name__ == '__main__':
    args = sys.argv[1:]
    append = '--append' in args
    args = [a for a in args if a != '--append']
    out_name = 'nvd_cve.jsonl'
    if '--out' in args:
        i = args.index('--out')
        out_name = args[i + 1]
        args = args[:i] + args[i + 2:]
    today = datetime.utcnow().date()
    start = datetime.fromisoformat(args[0]) if len(args) > 0 else datetime(2021, 1, 1)
    end = datetime.fromisoformat(args[1]) if len(args) > 1 else datetime(today.year, today.month, today.day)
    end = min(end, datetime(today.year, today.month, today.day))
    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, out_name)
    print(f'[nvd] window {start.date()} .. {end.date()} mode={"append" if append else "write"} -> {out}')
    pull(start, end, out, mode='a' if append else 'w')
