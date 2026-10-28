#!/usr/bin/env bash
# queue_zec_standard_kg_ns50.sh — 2026-10-25：WN18RR ZEC 表 ns12→ns50 连坐（论文换 ns50 口径时执行）。
#
# 依据：paper §8.2 tab:wn18rr 为 filtered 协议（caption 明写；NE 0.1636/proto-hard 0.2414/NE+side 0.3563
#       均为 rank_WN18RR_zec_RMPI_*_filtered.log 的数字，ns12）。
# 动作：只重排不重训（4 个 checkpoint 已在 expri_save_models/WN18RR_zec_RMPI_{base,NE,NE_sc,proto_hard}）。
# 口径：-m filtered --num_samples 50 --runs 1 --seed 41504，workers=4（WN18RR 图小，实测 ~6GB 安全，
#       见 queue_zec_standard_kg.sh 注释；与 ns12 filtered 完全同构，仅 num_samples 12→50）。
# 纪律：串行、起前两查、关收割。执行时机：ns50 安全链 ALL DONE 之后（串行，不并发）。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1
mkdir -p eval_logs_evalmode

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[zec50 $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[zec50 $(date '+%F %T')] 内存不足(<2GB)，中止"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[zec50 $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[zec50 $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

run_rank() {  # $1=expri
  local expri="$1"
  chk_leftover; chk_mem
  echo "[zec50 $(date '+%F %T')] rank $expri (filtered, ns50, seed 41504)"
  python -u test_ranking_P.py -d WN18RR_zec -e "$expri" -m filtered --runs 1 --seed 41504 \
    --num_samples 50 --workers 4 --cache_size 3000 \
    > "eval_logs_evalmode/rank_${expri}_filtered_ns50.log" 2>&1
  if grep -qaE "Avg test Set Performance" "eval_logs_evalmode/rank_${expri}_filtered_ns50.log"; then
    echo "[zec50 $(date '+%F %T')] rank $expri OK: $(grep -aE '^(MRR|Hits@10)' "eval_logs_evalmode/rank_${expri}_filtered_ns50.log" | tail -2 | tr '\n' ' ')"
  else
    echo "[zec50 $(date '+%F %T')] rank $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/rank_${expri}_filtered_ns50.log" | tail -1)"
    return 1
  fi
}

echo "=== WN18RR ZEC ns50 filtered 连坐链启动 $(date '+%F %T') ==="
run_rank WN18RR_zec_RMPI_base;       [ $? -eq 0 ] || exit 1
run_rank WN18RR_zec_RMPI_NE;         [ $? -eq 0 ] || exit 1
run_rank WN18RR_zec_RMPI_proto_hard; [ $? -eq 0 ] || exit 1
run_rank WN18RR_zec_RMPI_NE_sc;      [ $? -eq 0 ] || exit 1
echo "[zec50 $(date '+%F %T')] ALL DONE (WN18RR ns50 filtered: base/NE/proto_hard/NE_sc)"
