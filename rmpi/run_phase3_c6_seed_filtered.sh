#!/usr/bin/env bash
# run_phase3_c6_seed_filtered.sh — 链C：C6① 主表复现 + filtered 口径 合并一次重跑。
#
# 目标（A2_PLAN §3.B C6①）：把主表 T3 的排名口径从「sampled/raw」升级为「filtered」，
#   并显式固定 seed（当前 --seed 是死参数），使数字 bit-exact 可复现、与文献（Bordes filtered）可比。
#
# ⚠️ 执行前置（Phase 0 改码，我执行时先做，本脚本不内含）：
#   (a) test_ranking_P.py 修 seed：加 random.seed(params.seed) / np.random.seed(params.seed)。
#   (b) test_ranking_P.py 加 filtered mode：mode=all 时过滤掉其他真实 (h,r,t') 三元组，
#       （现 choices 只有 sample/all/ruleN，filtered 需新增），供本脚本用 -m filtered。
#   这两处改码等「现在这条排名链全部跑完、无 python 进程占用 .py」后再落地（rmpi-oom-serial-execution-rule）。
#
# 纪律：串行、一次一个 job、起前两查（FreeRAM>=2GB + 无残留 python）、关收割。
#   沿用 security_kg 排名实测配置：workers=1（mp.Pool 串行，内存安全）、cache_size 3000。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
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

# 排名口径：filtered（Phase 0 改码后启用）；seed 固定 42（与 ZEC split 一致）。
MODE=filtered
SEED=42
# 主表要重排名的模型（full_ind 纯结构阶梯 + ind 含侧信道阶梯）。
#   ablation: 0=base / 1=NE / 2=proto_hard / 3=proto_soft
PAIRS="
security_kg_full_ind_RMPI_base|security_kg_full_ind
security_kg_full_ind_RMPI_NE|security_kg_full_ind
security_kg_full_ind_RMPI_proto_hard|security_kg_full_ind
security_kg_full_ind_RMPI_proto_soft|security_kg_full_ind
security_kg_ind_RMPI_base|security_kg_ind
security_kg_ind_RMPI_NE|security_kg_ind
security_kg_ind_RMPI_proto_hard|security_kg_ind
security_kg_ind_RMPI_proto_soft|security_kg_ind
"

echo "=== 启动 C6① seed+filtered 主表重排（mode=$MODE seed=$SEED）$(date '+%F %T') ==="

for pair in $PAIRS; do
  expri="${pair%%|*}"; ds="${pair#*|}"
  model="expri_save_models/${expri}/best_graph_classifier.pth"
  if [ ! -f "$model" ]; then
    echo "[driver $(date '+%F %T')] skip $expri（模型不存在）"
    continue
  fi
  chk_leftover; chk_mem
  log="eval_logs_evalmode/rank_${expri}_filtered.log"
  echo "[driver $(date '+%F %T')] rank $expri ($ds, filtered, seed=$SEED)"
  python -u test_ranking_P.py -e "$expri" -d "$ds" -m "$MODE" --runs 1 --workers 1 --cache_size 3000 --seed "$SEED" > "$log" 2>&1
  if grep -qaE "Avg test Set Performance" "$log"; then
    echo "[driver $(date '+%F %T')] $expri OK: $(grep -aE '^(MRR|Hits@10)' "$log" | tail -2 | tr '\n' ' ')"
  else
    echo "[driver $(date '+%F %T')] $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "$log" | tail -1)"
  fi
done

echo "[driver $(date '+%F %T')] C6① seed+filtered 主表重排 DONE"
