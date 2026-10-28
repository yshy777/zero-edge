#!/usr/bin/env bash
# parallel_repull.sh — re-pull NVD CVE records (with CPE products) across 6
# disjoint date ranges in parallel, then merge into nvd_cve_cpe.jsonl.
#
# The sequential nvd.py run is bandwidth-bound (~90s per 2000-record page,
# ~2.5-3h for 210k CVEs). NVD's 5-req/30s rate limit is on request *rate*, not
# concurrency, and each puller issues ~1 req/90s, so 6 concurrent pullers stay
# well under the limit while overlapping the download time (~6x faster).
set -u
cd /c/RMPI_Code
RAW=data/security_kg/raw
mkdir -p "$RAW"

pull() {
  local tag="$1" s="$2" e="$3"
  python security_kg/nvd.py "$s" "$e" --out "nvd_cpe_$tag.jsonl" \
      > "security_kg/repull_$tag.log" 2>&1
  echo "[repull] $tag rc=$? lines=$(wc -l < "$RAW/nvd_cpe_$tag.jsonl" 2>/dev/null || echo 0)"
}

# Disjoint ranges; heavy years (2024-2025) sliced finer to balance wall-clock.
pull A 2021-01-01 2022-01-01 &
pull B 2022-01-01 2023-01-01 &
pull C 2023-01-01 2024-01-01 &
pull D 2024-01-01 2024-09-01 &
pull E 2024-09-01 2025-07-01 &
pull F 2025-07-01 2026-09-16 &
wait

echo "[repull] all pullers finished; merging"
python - <<'PY'
import json, os
raw = "data/security_kg/raw"
tags = ["A", "B", "C", "D", "E", "F"]
seen = {}
order = []
for t in tags:
    p = os.path.join(raw, f"nvd_cpe_{t}.jsonl")
    if not os.path.exists(p):
        print(f"[merge] WARNING missing {p}")
        continue
    n = 0
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cve = rec.get("cve")
        if cve and cve not in seen:
            seen[cve] = line
            order.append(cve)
        n += 1
    print(f"[merge] {t}: {n} raw records")
out = os.path.join(raw, "nvd_cve_cpe.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for cve in order:
        f.write(seen[cve] + "\n")
print(f"[merge] DONE: {len(order)} unique CVEs -> nvd_cve_cpe.jsonl")
PY
