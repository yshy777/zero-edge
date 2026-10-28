#!/usr/bin/env bash
# queue_full_chain_rebuttal.sh — 全量 ns50 链 + WN18RR ns50（rebuttal 材料，无死线）。
#   与 queue_predeadline.sh 的步骤 4-5 相同；在 10-28 子集表完成 + base 双 seed CI 之后
#   单独启动（queue_predeadline 会在子集链末被停掉，勿让它自动滚进本链）。
#   已核实（10-27）：C1 数据路径 + 原型 head 砍 + P0② 已在 10-20 落地、v1.0 同款，
#   无需任何改码——本链在修复代码上直跑。
#   纪律：串行、起前两查（残留 python + 内存≥2GB）；任一模型 FAILED 即 exit 1。
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

run_rank() {
  local model="$1"
  chk_leftover
  chk_mem
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  echo "[queue $(date '+%F %T')] ${model} (ns50 全量, runs=1, workers=1, cache=3000, seed=41504)"
  python -u test_ranking_P.py -e "security_kg_full_ind_RMPI_${model}" -d security_kg_full_ind -m sample \
    --runs 1 --workers 1 --cache_size 3000 --num_samples 50 --seed 41504 \
    > "$LOGDIR/rank_security_kg_full_ind_RMPI_${model}_ns50.log" 2>&1
  if grep -qaE "Avg test Set Performance" "$LOGDIR/rank_security_kg_full_ind_RMPI_${model}_ns50.log"; then
    echo "[queue $(date '+%F %T')] ${model}(ns50 全量) OK: $(grep -aE '^(MRR|Hits@1|Hits@5|Hits@10)' "$LOGDIR/rank_security_kg_full_ind_RMPI_${model}_ns50.log" | tail -4 | tr '\n' ' ')"
  else
    echo "[queue $(date '+%F %T')] ${model}(ns50 全量) FAILED: $(grep -aE 'Error|Traceback|MemoryError' "$LOGDIR/rank_security_kg_full_ind_RMPI_${model}_ns50.log" | tail -1)"
    exit 1
  fi
}

echo "=== queue_full_chain_rebuttal 启动 $(date '+%F %T') ==="

# 闸门 0：子集表必须已齐（proto_hard_sc 子集 OK 是最后一行）——防误启动
if ! grep -qaE "Avg test Set Performance" "$LOGDIR/rank_security_kg_full_ind_RMPI_proto_hard_sc_ns50_subset.log"; then
  echo "[queue $(date '+%F %T')] proto_hard_sc 子集未完成，中止（先等子集表齐）"
  exit 1
fi
echo "[queue $(date '+%F %T')] 闸门0 通过：子集表已齐（proto_hard_sc OK）"

# ── 全量 ns50 链（rebuttal 材料，无死线；确定性协议，重启不损失数字）──
run_rank NE
run_rank proto_hard
run_rank proto_hard_sc

# ── WN18RR ZEC ns50 filtered 连坐（rebuttal 材料）──
chk_leftover; chk_mem
cd /c/RMPI_Code/RMPI
bash /c/RMPI_Code/RMPI/queue_zec_standard_kg_ns50.sh || { echo "[queue $(date '+%F %T')] WN18RR ns50 失败"; exit 1; }

echo "[queue $(date '+%F %T')] ALL DONE（全量 ns50 链 + WN18RR ns50）"
