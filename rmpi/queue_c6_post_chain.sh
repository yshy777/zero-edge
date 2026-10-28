#!/usr/bin/env bash
# queue_c6_post_chain.sh — WN18RR 链完成后的串行实验队列（C6 修复包 + C1 + 侧信道变体）。
#
# 前置（脚本启动前由人工完成，本脚本只负责跑）：
#   [编辑1] test_ranking_P.py —— ① --seed 接线（:802 死参数 → np.random.seed + 抽样前固定）
#                               ② 新增 -m filtered 模式（负候选=tail 实体空间：WN18RR 用关系一致的
#                                  train tail；security KG 用 CWE 实体池），P0②（:128/:717）
#                               ③ security KG sampled 模式的负采样池从全实体 → CWE 实体池
#   [编辑2] graph_classifier.py —— 原型 head 砍（head_proto 常数、排序抵消；砍后 proto_hard≈NE+tail 原型）
#   [编辑3] datasets.py —— cwe2id remap 错位修复（P1，AUC 路径）
#   [编辑4] C1 五点接线（datasets.py:128-130 / graph_classifier.py:16-17 / test_ranking_P.py:644-650
#           / train.py:58 / 评测侧注入，见 paper/C1_data_path_generalization.md）
#   启动前自检：grep 下方 PHASE0 检查项，全过才放行。
#
# 阶段（串行、fail-fast、每步前两查 FreeRAM≥2GB + 无残留 python）：
#   P1  WN18RR_zec filtered 重跑 ×3 消融（~5-6h）  → §8.2 第二数据集主表口径（文献可比，ZEC 第 5 条）
#   P2  WN18RR_zec NE+侧信道 变体（训 ~5h + 排 ~2h）→ ZEC 第 6 条侧信道消融行
#   P3  security_kg_full_ind CWE-pool 采样重跑 ×4（~60-70h）→ 主表口径修正（P0②）+ seed 可复现（P0①）
#   P4  AUC 复核（test_auc_F，cut 后模型，strict=False 载旧权重；ind + full_ind 两表）
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
    echo "[driver $(date '+%F %T')] 内存不足(<2GB)，中止"
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

# ---- PHASE0: 编辑完成度自检（防带病上阵）----
echo "[driver $(date '+%F %T')] PHASE0 编辑自检"
fail=0
grep -qa 'filtered' <(python -c "import re;print(open('test_ranking_P.py',encoding='utf-8').read())" | grep -o 'choices=\[[^]]*\]') && echo "  [ok] -m filtered 已入 choices" || { echo "  [FAIL] test_ranking_P.py 无 filtered 模式"; fail=1; }
grep -qa "np.random.seed" test_ranking_P.py && echo "  [ok] seed 接线存在" || { echo "  [FAIL] seed 未接线"; fail=1; }
grep -qa "head_proto" model/dgl/graph_classifier.py && echo "  [FAIL] head_proto 还在（head 砍未做）" && fail=1 || echo "  [ok] head_proto 已砍"
grep -qa "load_side_channel" subgraph_extraction/datasets.py && echo "  [ok] C1 数据路径已接" || { echo "  [FAIL] load_side_channel 未接"; fail=1; }
[ "$fail" != "0" ] && { echo "[driver] PHASE0 未通过，拒绝启动（先把前置编辑做完）"; exit 1; }
echo "[driver] PHASE0 通过"

run_rank() {  # $1=dataset $2=expri $3=mode
  local ds="$1" expri="$2" mode="$3"
  chk_leftover; chk_mem
  echo "[driver $(date '+%F %T')] rank $expri (mode=$mode)"
  python -u test_ranking_P.py -d "$ds" -e "$expri" -m "$mode" --runs 1 --seed 41504 --num_samples 12 \
    > "eval_logs_evalmode/rank_${expri}_${mode}.log" 2>&1
  if grep -qaE "Avg test Set Performance" "eval_logs_evalmode/rank_${expri}_${mode}.log"; then
    echo "[driver $(date '+%F %T')] rank $expri OK: $(grep -aE '^(MRR|Hits@10)' "eval_logs_evalmode/rank_${expri}_${mode}.log" | tail -2 | tr '\n' ' ')"
  else
    echo "[driver $(date '+%F %T')] rank $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/rank_${expri}_${mode}.log" | tail -1)"
    return 1
  fi
}

# ---- P1: WN18RR_zec filtered ×3（第二数据集主表口径）----
echo "[driver $(date '+%F %T')] === P1: WN18RR_zec filtered ×3 ==="
run_rank WN18RR_zec WN18RR_zec_RMPI_base filtered;      [ $? -eq 0 ] || exit 1
run_rank WN18RR_zec WN18RR_zec_RMPI_NE filtered;        [ $? -eq 0 ] || exit 1
run_rank WN18RR_zec WN18RR_zec_RMPI_proto_hard filtered; [ $? -eq 0 ] || exit 1

# ---- P2: WN18RR_zec NE + 侧信道（ZEC 第 6 条消融行；side_channel.npz 已生成）----
echo "[driver $(date '+%F %T')] === P2: WN18RR_zec NE+SC ==="
chk_leftover; chk_mem
python -u train.py -d WN18RR_zec -e WN18RR_zec_RMPI_NE_sc --ablation 1 \
  --num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1 \
  > train_WN18RR_zec_RMPI_NE_sc.log 2>&1
if grep -qaE "best validation AUC" train_WN18RR_zec_RMPI_NE_sc.log; then
  echo "[driver $(date '+%F %T')] train NE_sc OK: $(grep -aE 'best validation AUC' train_WN18RR_zec_RMPI_NE_sc.log | tail -1)"
else
  echo "[driver $(date '+%F %T')] train NE_sc FAILED"; exit 1
fi
run_rank WN18RR_zec WN18RR_zec_RMPI_NE_sc filtered;     [ $? -eq 0 ] || exit 1

# ---- P3: security_kg_full_ind CWE-pool 采样重跑 ×4（主表口径修正+可复现）----
echo "[driver $(date '+%F %T')] === P3: security_kg_full_ind CWE-pool ×4 ==="
run_rank security_kg_full_ind security_kg_full_ind_RMPI_base sample;       [ $? -eq 0 ] || exit 1
run_rank security_kg_full_ind security_kg_full_ind_RMPI_NE sample;         [ $? -eq 0 ] || exit 1
run_rank security_kg_full_ind security_kg_full_ind_RMPI_proto_hard sample; [ $? -eq 0 ] || exit 1
run_rank security_kg_full_ind security_kg_full_ind_RMPI_proto_hard_sc sample; [ $? -eq 0 ] || exit 1

# ---- P4: AUC 复核（route-B/cut 后，test_auc_F；旧权重 strict=False 载入）----
echo "[driver $(date '+%F %T')] === P4: AUC 复核 ==="
# TODO: 确认 test_auc_F 的 expri/数据集参数后填具体命令（ind 表 base/NE/proto_hard + full_ind 表）
echo "[driver $(date '+%F %T')] P4 命令待编辑完成后确认（test_auc_F 参数面）"

echo "[driver $(date '+%F %T')] ALL DONE (C6+C1 队列：P1-P4)"
