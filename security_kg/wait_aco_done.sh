#!/usr/bin/env bash
# wait_aco_done.sh — lightweight notifier for STEP 6 (dump) + STEP 7 (ACO).
# Replaces the tracking lost to BG_SHELL_PRESSURE_REAP (underlying processes survived).
# Exits 0 with results when all 4 ACO ablations finish; 1 on error; 2 on timeout.
cd /c/RMPI_Code
EXPI=security_kg_ind_RMPI_proto_hard_cpe
LOG=security_kg/resume_aco_pipeline.log
DIR=data/security_kg_ind
CAND="$DIR/completion_cands_${EXPI}.json"
START=$(date +%s)
dump_reported=0

for i in $(seq 1 480); do   # up to 4h (480 x 30s)
  # dump-completion signal (completion_cands rewritten after the waiter started)
  if [ "$dump_reported" -eq 0 ] && [ -f "$CAND" ]; then
    m=$(stat -c %Y "$CAND" 2>/dev/null)
    if [ -n "$m" ] && [ "$m" -gt "$START" ]; then
      echo "=== DUMP DONE $(date '+%F %T') (completion_cands rewritten) ==="
      dump_reported=1
    fi
  fi

  # error detection (dump/ACO failure signatures)
  if grep -qaE "DUMP FAILED|DUMP EMPTY|MISSING OUTPUT|Traceback|RuntimeError|MemoryError|bad allocation" "$LOG" 2>/dev/null; then
    echo "=== ACO pipeline ERROR $(date '+%F %T') ==="
    tail -40 "$LOG"
    exit 1
  fi

  # success: all 4 ablations produced risk scores
  cnt=0
  for ab in A A1 B C; do
    [ -f "$DIR/aco_risk_${ab}_${EXPI}.json" ] && cnt=$((cnt+1))
  done
  if [ "$cnt" -ge 4 ]; then
    echo "=== ACO 4 ablations DONE $(date '+%F %T') ==="
    grep -aE "ablation|KEV ROC|Spearman|queries=|MISSING" "$LOG" | tail -40
    exit 0
  fi

  sleep 30
done

echo "=== waiter 4h TIMEOUT $(date '+%F %T') — dump still running? ==="
powershell.exe -NoProfile -Command "(Get-Process -Id 9764 -ErrorAction SilentlyContinue).CPU" 2>/dev/null
exit 2
