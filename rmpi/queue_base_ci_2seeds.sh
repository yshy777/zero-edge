#!/usr/bin/env bash
# queue_base_ci_2seeds.sh — base ns50 子集双 seed 重抽 CI（GPT 意见#1：单 seed 稳健性）。
#   口径：同协议 --max_links 4954 --subset_shuffle，仅换 seed（41505/41506 ≠ 主 seed 41504）
#   → 每次是独立的 10% 随机重抽（链接组成 + 负样本一起变，bootstrap 式稳定性证据）。
#   论文披露口径：固定子集 41504（0.0514）+ 两次重抽的 MRR 区间。
#   10-28 子集表齐后、全量链启动前跑（~2.5-3h/seed，串行）。
#   纪律：串行、起前两查；FAILED 即 exit 1。
set -u
cd /c/RMPI_Code/RMPI
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGDIR=eval_logs_evalmode
mkdir -p "$LOGDIR"

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[queue $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[queue $(date '+%F %T')] 内存不足(<2GB)，中止，避免 OOM"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[queue $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[queue $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

run_ci() {
  local seed="$1"
  chk_leftover
  chk_mem
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  echo "[queue $(date '+%F %T')] base (ns50 子集 CI, seed=${seed}: 4954 links 重抽)"
  python -u test_ranking_P.py -e "security_kg_full_ind_RMPI_base" -d security_kg_full_ind -m sample \
    --runs 1 --workers 1 --cache_size 3000 --num_samples 50 --seed ${seed} \
    --max_links 4954 --subset_shuffle \
    > "$LOGDIR/rank_security_kg_full_ind_RMPI_base_ns50_ci_seed${seed}.log" 2>&1
  if grep -qaE "Avg test Set Performance" "$LOGDIR/rank_security_kg_full_ind_RMPI_base_ns50_ci_seed${seed}.log"; then
    echo "[queue $(date '+%F %T')] base CI seed=${seed} OK: $(grep -aE '^(MRR|Hits@1|Hits@5|Hits@10)' "$LOGDIR/rank_security_kg_full_ind_RMPI_base_ns50_ci_seed${seed}.log" | tail -4 | tr '\n' ' ')"
  else
    echo "[queue $(date '+%F %T')] base CI seed=${seed} FAILED: $(grep -aE 'Error|Traceback|MemoryError' "$LOGDIR/rank_security_kg_full_ind_RMPI_base_ns50_ci_seed${seed}.log" | tail -1)"
    exit 1
  fi
}

echo "=== queue_base_ci_2seeds 启动 $(date '+%F %T') ==="

# 闸门 0：子集表必须已齐
if ! grep -qaE "Avg test Set Performance" "$LOGDIR/rank_security_kg_full_ind_RMPI_proto_hard_sc_ns50_subset.log"; then
  echo "[queue $(date '+%F %T')] proto_hard_sc 子集未完成，中止（先等子集表齐）"
  exit 1
fi
echo "[queue $(date '+%F %T')] 闸门0 通过：子集表已齐"

run_ci 41505
run_ci 41506

echo "[queue $(date '+%F %T')] ALL DONE（base 双 seed CI）"
