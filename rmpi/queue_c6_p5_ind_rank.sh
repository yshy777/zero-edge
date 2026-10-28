#!/usr/bin/env bash
# queue_c6_p5_ind_rank.sh — ind 15K 子集 CWE-pool 重排 ×4（P0② 口径修正补漏）。
# 主链 P3 只重跑了 full_ind；draft 里 ind 旧阶梯（base 0.027 → NE 0.445 → proto_hard 0.605，
# 及 proto_soft 温度行）同样是全实体池数字，同款类型通胀，须同一口径重排。
# 时机：主链 ALL DONE 后、P4 AUC 链之前（ind 小：3,679 test links，~2.5-3.5h 含 db 生成）。
# 口径：mode=sample（CWE 池）+ ns12 + seed 41504，与 P3 主表一致；db 生成于 C 盘
#       data/security_kg_ind/test_subgraphs_RMPI_neg_1_hop_2（当前无 RMPI 版，只有旧 TACT）。
# 注意：ind split 自带产品指纹侧信道（draft §2 口径已披露），此处数字仍含 SC——口径不变。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p eval_logs_evalmode

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[p5 $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[p5 $(date '+%F %T')] 内存不足(<2GB)，中止"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[p5 $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[p5 $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

run_rank() {  # $1=expri
  local expri="$1"
  chk_leftover; chk_mem
  echo "[p5 $(date '+%F %T')] rank $expri (mode=sample, CWE-pool)"
  python -u test_ranking_P.py -d security_kg_ind -e "$expri" -m sample --runs 1 --seed 41504 --num_samples 12 \
    > "eval_logs_evalmode/rank_${expri}_sample.log" 2>&1
  if grep -qaE "Avg test Set Performance" "eval_logs_evalmode/rank_${expri}_sample.log"; then
    echo "[p5 $(date '+%F %T')] rank $expri OK: $(grep -aE '^(MRR|Hits@10)' "eval_logs_evalmode/rank_${expri}_sample.log" | tail -2 | tr '\n' ' ')"
  else
    echo "[p5 $(date '+%F %T')] rank $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/rank_${expri}_sample.log" | tail -1)"
    return 1
  fi
}

echo "[p5 $(date '+%F %T')] === P5: security_kg_ind CWE-pool ×4 ==="
run_rank security_kg_ind_RMPI_base;        [ $? -eq 0 ] || exit 1
run_rank security_kg_ind_RMPI_NE;          [ $? -eq 0 ] || exit 1
run_rank security_kg_ind_RMPI_proto_hard;  [ $? -eq 0 ] || exit 1
run_rank security_kg_ind_RMPI_proto_soft;  [ $? -eq 0 ] || exit 1

echo "[p5 $(date '+%F %T')] P5 ALL DONE (rank_security_kg_ind_*_sample.log 已落盘)"
