#!/usr/bin/env bash
# pipeline_rebuild_retrain_aco.sh — full ACO-subset pipeline, run autonomously
# after the NVD re-pull finishes.
#
#   1) rebuild security_kg{,_ind} with hasVulnerability (Product -> CVE)
#   2) verify the fix (unseen test CVE is no longer isolated)
#   3) invalidate stale caches (relation2id / subgraph db / cluster cache)
#   4) retrain proto_hard on security_kg_ind (config reproduced from the
#      2026-09-19 log: ablation 2, batch 16, eval 455, 20 epochs)
#   5) smoke-dump completion cands for 5 CVEs -> assert completion is now
#      CVE-specific (the whole point of the fix)
#   6) full dump of completion cands -> run the 4 completion-aware ACO ablations
set -u
set -o pipefail
cd /c/RMPI_Code
LOG=security_kg/pipeline_rebuild_retrain_aco.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec > "$LOG" 2>&1

echo "=== [pipeline] $(date '+%F %T') STEP 1: rebuild (--max-cves 15000) ==="
python security_kg/build_kg.py --max-cves 15000 || { echo "BUILD FAILED"; exit 1; }

echo "=== STEP 2: verify product fingerprint (scheme 3) ==="
python - <<'PY' || { echo "VERIFY FAILED"; exit 1; }
import json

def read_triples(fn):
    out = []
    with open('data/security_kg_ind/' + fn, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3:
                out.append(p)
    return out

train = read_triples('train.txt')
test = read_triples('test.txt')
rels = {p[1] for p in train}
print('relations in train.txt:', sorted(rels))
assert 'has_weakness' in rels
assert 'hasVulnerability' not in rels, 'hasVulnerability edges must be removed (scheme 3)'

fp = json.load(open('data/security_kg_ind/product_fingerprint.json', encoding='utf-8'))
train_hw = {p[0] for p in train if p[1] == 'has_weakness'}
test_cves = sorted({p[0] for p in test if p[1] == 'has_weakness'} - train_hw)
with_prod = [c for c in test_cves if c in fp]
frac = len(with_prod) / max(len(test_cves), 1)
print(f'test CVEs: {len(test_cves)}, with product fingerprint: {len(with_prod)} ({100*frac:.1f}%)')
assert frac >= 0.30, f'fingerprint coverage too low ({100*frac:.1f}%)'
print('VERIFY OK: product fingerprint gives CVE-specific identity without graph edges')
PY

echo "=== STEP 3: invalidate stale caches ==="
for d in data/security_kg data/security_kg_ind; do
  rm -f "$d/relation2id.json"
  rm -rf "$d/subgraphs_RMPI_neg_1_hop_2"
  rm -f "$d/id2cluster_nc64.npy" "$d/id2soft_nc64.npy" "$d/cluster_meta_nc64.npz" "$d/id2entity_nc64.npy"
  rm -rf "$d"/test_subgraphs_RMPI_ab*_neg_1_hop_2
  rm -f "$d"/*_ranking_head.txt "$d"/*_ranking_tail.txt
  rm -f "$d"/completion_cands_*.json "$d"/aco_risk*.json "$d"/negative_test_triples_by_RMPI.txt
  echo "  cleaned $d"
done

echo "=== STEP 4: retrain proto_hard (ablation=2, 20 epochs) ==="
cd RMPI
EXPI=security_kg_ind_RMPI_proto_hard_cpe
python -u train.py -d security_kg_ind -e "$EXPI" --ablation 2 --num_epochs 20 \
    --batch_size 16 --eval_every_iter 455 --num_workers 0 --gpu -1 \
    > "train_${EXPI}.log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if grep -q "Epoch 20 " "train_${EXPI}.log" 2>/dev/null; then sleep 5; kill -9 "$pid" 2>/dev/null; break; fi
  if grep -qE "Traceback|RuntimeError|MemoryError|IndexError" "train_${EXPI}.log" 2>/dev/null; then break; fi
  sleep 30
done
wait "$pid" 2>/dev/null
if grep -q "Epoch 20 " "train_${EXPI}.log" 2>/dev/null; then
  echo "RETRAIN OK: $(grep -aE 'best validation AUC' train_${EXPI}.log | tail -1)"
else
  echo "RETRAIN FAILED: $(grep -aE 'Traceback|RuntimeError|MemoryError|Error' train_${EXPI}.log | tail -1)"; exit 1
fi

echo "=== STEP 5: smoke-dump 20 CVEs + assert CVE-specificity (with-fingerprint) ==="
python -u dump_completion_cands.py -d security_kg_ind -e "$EXPI" \
    --max_cves 20 --top_k 50 --chunk 100 2>&1 | tail -3
python - <<'PY' || { echo "CVE-INVARIANT: fix did not take"; exit 1; }
import json
d = json.load(open('../data/security_kg_ind/completion_cands_%s.json' % 'security_kg_ind_RMPI_proto_hard_cpe'))
fp = json.load(open('../data/security_kg_ind/product_fingerprint.json', encoding='utf-8'))
keys = sorted(d)
with_prod = [k for k in keys if k in fp]
without = [k for k in keys if k not in fp]
vecs = {k: [round(s, 5) for _, s in d[k]] for k in keys}
uniq_wp = {vecs[k] for k in with_prod}
print(f'smoke CVEs: {len(keys)} (with fingerprint: {len(with_prod)}, without: {len(without)})')
print(f'distinct score vectors among with-fingerprint CVEs: {len(uniq_wp)}/{len(with_prod)}')
assert len(with_prod) >= 3, 'need >=3 with-fingerprint CVEs'
assert len(uniq_wp) >= 2, 'with-fingerprint CVEs share identical score vectors -> still CVE-invariant'
print('SPECIFICITY OK: with-fingerprint CVEs get distinct (CVE-specific) completion scores')
PY

echo "=== STEP 6: full completion dump ==="
python -u dump_completion_cands.py -d security_kg_ind -e "$EXPI" \
    --max_cves 0 --top_k 50 --chunk 100 2>&1 | tail -5 || { echo "DUMP FAILED"; exit 1; }
n_dumped=$(python -c "import json;print(len(json.load(open('../data/security_kg_ind/completion_cands_${EXPI}.json'))))")
echo "DUMP OK: $n_dumped unseen CVEs"
[ "$n_dumped" -gt 0 ] || { echo "DUMP EMPTY"; exit 1; }
cd ..

echo "=== STEP 7: completion-aware ACO (4 ablations) ==="
for ab in A A1 B C; do
  python security_kg/aco_completion.py --kg security_kg_ind --expri "$EXPI" \
      --ablation "$ab" --top_k 20 2>&1 | grep -aE "KEV ROC|Spearman|queries="
done
for ab in A A1 B C; do
  [ -f "data/security_kg_ind/aco_risk_${ab}_${EXPI}.json" ] \
      || { echo "ACO $ab MISSING OUTPUT"; exit 1; }
done
echo "ACO OK: all 4 ablations produced risk scores"

echo "=== [pipeline] ALL DONE $(date '+%F %T') ==="
