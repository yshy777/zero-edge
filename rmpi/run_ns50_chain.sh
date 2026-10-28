#!/usr/bin/env bash
# run_ns50_chain.sh — 2026-10-25 用户拍板：投稿前补 ns50 正式口径（num_samples=50，默认值）
#   + seed 已接线（test_ranking_P.py L648-651，C6）→ 本次即 bit-exact 复现口径。
#   串行链：base → NE → proto_hard → proto_hard_sc（轻→重，每步 49,535 links, runs=1, seed 41504）。
#   纪律（rmpi-oom-serial-execution-rule）：一次一个 job，起前 FreeGB>=2 + 无残留 python，绝不并发。
#   关收割：CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1
mkdir -p eval_logs_evalmode

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[driver $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[driver $(date '+%F %T')] 内存不足(<2GB)，中止，避免 OOM"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[driver $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[driver $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

run_rank() {
  local model="$1"
  chk_leftover
  chk_mem
  echo "[driver $(date '+%F %T')] ${model} (ns50, runs=1, workers=1, cache=3000, seed=41504)"
  python -u test_ranking_P.py -e "security_kg_full_ind_RMPI_${model}" -d security_kg_full_ind -m sample \
    --runs 1 --workers 1 --cache_size 3000 --num_samples 50 --seed 41504 \
    > "eval_logs_evalmode/rank_security_kg_full_ind_RMPI_${model}_ns50.log" 2>&1
  if grep -qaE "Avg test Set Performance" "eval_logs_evalmode/rank_security_kg_full_ind_RMPI_${model}_ns50.log"; then
    echo "[driver $(date '+%F %T')] ${model}(ns50) OK: $(grep -aE '^(MRR|Hits@1|Hits@5|Hits@10)' "eval_logs_evalmode/rank_security_kg_full_ind_RMPI_${model}_ns50.log" | tail -4 | tr '\n' ' ')"
  else
    echo "[driver $(date '+%F %T')] ${model}(ns50) FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/rank_security_kg_full_ind_RMPI_${model}_ns50.log" | tail -1)"
  fi
}

echo "=== ns50 串行排名链启动 $(date '+%F %T') ==="
run_rank base
run_rank NE
run_rank proto_hard
run_rank proto_hard_sc
echo "[driver $(date '+%F %T')] ALL DONE (ns50 chain: base/NE/proto_hard/proto_hard_sc)"
