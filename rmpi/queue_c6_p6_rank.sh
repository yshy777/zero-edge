#!/usr/bin/env bash
# queue_c6_p6_rank.sh — P6：转导 CWE-pool 重排 ×4 + 2+SC mini-probe 交叉验证。
# 目的：(1) draft §8.2 转导段还引着全实体池旧数字，须与 P3/P5 同口径（CWE 池 ns12 seed 41504）重排；
#       (2) §8.2 分层分解 M_cov≈0.473 的 mini 交叉验证——旧 mini（0.644/0.6514）是全实体池，已撤。
# 时机：P4 全链 ALL DONE 之后（proto 变体依赖 P4d 重建的转导 cluster 缓存，同 seed 41504）。
# 规模：转导 test 仅 1,951 links（P5 半归纳 3,679 links 全链 ~1.5h 的一半），mini-probe 2,000 links ~30min。
# 口径：mode=sample（CWE 池）+ ns12 + seed 41504 + runs=1，与 P3 主表/P5 子集表一致。
# 日志：rank_security_kg_RMPI_{base,NE,proto_hard,proto_soft}_sample.log（新名，旧 pool 时代无 _sample 后缀不冲突）；
#       mini 写 rank_sc_mini_cwepool.log（旧 rank_sc_mini.log 为 pool 口径，保留不覆盖）。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p eval_logs_evalmode

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[p6 $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[p6 $(date '+%F %T')] 内存不足(<2GB)，中止"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[p6 $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[p6 $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

run_rank() {  # $1=dataset $2=expri $3=logname
  local ds="$1" expri="$2" logname="$3"
  chk_leftover; chk_mem
  echo "[p6 $(date '+%F %T')] rank $expri (mode=sample, CWE-pool, ns12)"
  python -u test_ranking_P.py -d "$ds" -e "$expri" -m sample --runs 1 --seed 41504 --num_samples 12 --workers 1 \
    > "eval_logs_evalmode/${logname}" 2>&1
  if grep -qaE "Avg test Set Performance" "eval_logs_evalmode/${logname}"; then
    echo "[p6 $(date '+%F %T')] rank $expri OK: $(grep -aE '^(MRR|Hits@10)' "eval_logs_evalmode/${logname}" | tail -2 | tr '\n' ' ')"
  else
    echo "[p6 $(date '+%F %T')] rank $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/${logname}" | tail -1)"
    return 1
  fi
}

echo "[p6 $(date '+%F %T')] === P6a: security_kg 转导 CWE-pool ×4（P4d cluster 缓存须已重建）==="
run_rank security_kg security_kg_RMPI_base        rank_security_kg_RMPI_base_sample.log;        [ $? -eq 0 ] || exit 1
run_rank security_kg security_kg_RMPI_NE          rank_security_kg_RMPI_NE_sample.log;          [ $? -eq 0 ] || exit 1
run_rank security_kg security_kg_RMPI_proto_hard  rank_security_kg_RMPI_proto_hard_sample.log;  [ $? -eq 0 ] || exit 1
run_rank security_kg security_kg_RMPI_proto_soft  rank_security_kg_RMPI_proto_soft_sample.log;  [ $? -eq 0 ] || exit 1

echo "[p6 $(date '+%F %T')] === P6b: 2+SC mini-probe（2,000 links，交叉验证 §8.2 分层分解）==="
chk_leftover; chk_mem
python -u test_ranking_P.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard_sc -m sample \
  --runs 1 --seed 41504 --num_samples 12 --workers 1 --cache_size 3000 --max_links 2000 \
  > eval_logs_evalmode/rank_sc_mini_cwepool.log 2>&1
if grep -qaE "Avg test Set Performance" eval_logs_evalmode/rank_sc_mini_cwepool.log; then
  echo "[p6 $(date '+%F %T')] mini OK: $(grep -aE '^(MRR|Hits@10)' eval_logs_evalmode/rank_sc_mini_cwepool.log | tail -2 | tr '\n' ' ')"
else
  echo "[p6 $(date '+%F %T')] mini FAILED: $(grep -aE 'Error|Traceback|MemoryError' eval_logs_evalmode/rank_sc_mini_cwepool.log | tail -1)"
  exit 1
fi

echo "[p6 $(date '+%F %T')] P6 ALL DONE (rank_security_kg_RMPI_*_sample.log + rank_sc_mini_cwepool.log 已落盘)"
