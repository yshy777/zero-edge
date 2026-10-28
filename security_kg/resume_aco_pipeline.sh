#!/usr/bin/env bash
# resume_aco_pipeline.sh — STEP 6 (full completion dump) + STEP 7 (4 ACO ablations),
# reusing the completed retrain checkpoint (best_graph_classifier.pth, val AUC 0.8255).
# Run after the STEP 5 smoke dump already verified CVE-specificity.
set -u
set -o pipefail
cd /c/RMPI_Code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
EXPI=security_kg_ind_RMPI_proto_hard_cpe
LOG=security_kg/resume_aco_pipeline.log
exec > "$LOG" 2>&1

echo "=== [resume] $(date '+%F %T') STEP 6: full completion dump ==="
cd RMPI
python -u dump_completion_cands.py -d security_kg_ind -e "$EXPI" \
    --max_cves 0 --top_k 50 --chunk 100 2>&1 | tail -5 || { echo "DUMP FAILED"; exit 1; }
n_dumped=$(python -c "import json;print(len(json.load(open('../data/security_kg_ind/completion_cands_${EXPI}.json'))))")
echo "DUMP OK: $n_dumped unseen CVEs"
[ "$n_dumped" -gt 0 ] || { echo "DUMP EMPTY"; exit 1; }
cd ..

echo "=== [resume] $(date '+%F %T') STEP 7: completion-aware ACO (4 ablations) ==="
for ab in A A1 B C; do
  echo "--- ablation $ab ---"
  python security_kg/aco_completion.py --kg security_kg_ind --expri "$EXPI" \
      --ablation "$ab" --top_k 20 2>&1 | grep -aE "KEV ROC|Spearman|queries="
done
for ab in A A1 B C; do
  [ -f "data/security_kg_ind/aco_risk_${ab}_${EXPI}.json" ] \
      || { echo "ACO $ab MISSING OUTPUT"; exit 1; }
done
echo "ACO OK: all 4 ablations produced risk scores"
echo "=== [resume] ALL DONE $(date '+%F %T') ==="
