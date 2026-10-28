#!/usr/bin/env bash
# queue_zec_standard_kg.sh — C5: 标准 KG（WN18RR 全量）零边冷启动全流程（A2_PLAN §3.C Q2）。
#
# 目的：在「零边 ZEC split」上跑我方 3 消融（base/NE/proto_hard，proto_soft 后补），
#       证明 Lemma 1（零边→空 enclosing→塌缩）+ disclosing/原型恢复，在标准 KG 上同样成立，
#       作为 security_kg 之外的「第二数据集泛化证据」。
#
# 数据集：data/WN18RR_zec（全量 WN18RR 40,943 实体 / 11 关系 / 10% 冷实体=4,094）
#         8,157 test / 906 valid，head-zero 率 100%、空 enclosing 率 100%。
#   ⚠️ 不是旧的 WN18RR_v1 / WN18RR_v1_zec —— 那是 9 关系 / 5,410 train 的缩水子集，
#      不是标准 benchmark；本脚本用 baselines/grail/data/WN18RR 的全量 11 关系版重建。
#
# 依赖（写稿/执行前必读）：
#   - 本脚本跑「纯结构」4 消融，与 security_kg 同构，当前代码即可跑（零改码）。
#   - 「+侧信道」变体需等 C1（数据路径通用化：datasets.py 接 load_side_channel）落地后再加；
#     A2_PLAN §2 侧信道定义：WN18RR 用 synset 名 char n-gram，C4 已备 side_channel_text.py
#     （命令见文末注释）。纯结构版先在零边标准 KG 上立住，侧信道是第二张牌。
#   - 排名口径当前是 raw（未 filtered）；C6 会补 filtered 并与 seed 合并一次重跑
#     （A2_PLAN §3.B C6①：ZEC 排名口径同步定为 filtered，与文献可比）。
#
# 纪律：串行、一次一个 job、起前两查（FreeRAM≥2GB + 无残留 python）、关收割。
#       WN18RR 图小（子图稀疏、无 hub），workers=4 内存安全（原 WN18RR 4 并行实测 ~6GB）。
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

DS=WN18RR_zec
# 训练超参对齐原 WN18RR（lr/dropout/layers/rel_dim/l2/margin/neg/hop 走 train.py 默认），
# 仅 batch/eval 间隔按 70k 链接放大（原 v1 5.4k 用 batch16/eval455）。
TRAIN_ARGS="--num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1"
# 排名口径（2026-10-19 定案）：ns12 runs=1，与安全 KG 主表阶梯 + §3.5 ZEC 协议一致；
# 旧 ns50/runs5 弃用，列为 C6/rebuttal 备选（filtered 重跑时同步决定）。
RANK_ARGS="-m sample --runs 1 --workers 4 --num_samples 12"

echo "=== 启动 C5 标准 KG 零边全流程（$DS）$(date '+%F %T') ==="

# ---- step0: 重建 ZEC split（全量 WN18RR，10% 冷实体，seed 42；已存在则跳过）----
if [ ! -f "../data/$DS/zec_meta.json" ]; then
  echo "[driver $(date '+%F %T')] step0 重建 ZEC split（全量 WN18RR）"
  python make_zec_split.py --src /c/RMPI_Code/baselines/grail/data/WN18RR \
    --dst "../data/$DS" --cold_frac 0.1 --valid_frac 0.1 --seed 42
else
  echo "[driver $(date '+%F %T')] step0 skip：ZEC split 已存在（../data/$DS）"
fi

# ---- train 一个消融 ----
train_one() {
  local abl="$1" expri="$2"
  chk_leftover; chk_mem
  echo "[driver $(date '+%F %T')] train $expri (ablation $abl)"
  python -u train.py -d "$DS" -e "$expri" --ablation "$abl" $TRAIN_ARGS \
    > "train_${expri}.log" 2>&1
  if grep -qaE "best validation AUC" "train_${expri}.log"; then
    echo "[driver $(date '+%F %T')] train $expri OK: $(grep -aE 'best validation AUC' "train_${expri}.log" | tail -1)"
  else
    echo "[driver $(date '+%F %T')] train $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "train_${expri}.log" | tail -1)"
    return 1
  fi
}

# ---- 排名一个消融 ----
rank_one() {
  local expri="$1"
  chk_leftover; chk_mem
  echo "[driver $(date '+%F %T')] rank $expri"
  python -u test_ranking_P.py -d "$DS" -e "$expri" $RANK_ARGS \
    > "eval_logs_evalmode/rank_${expri}.log" 2>&1
  if grep -qaE "Avg test Set Performance" "eval_logs_evalmode/rank_${expri}.log"; then
    echo "[driver $(date '+%F %T')] rank $expri OK: $(grep -aE '^(MRR|Hits@10)' "eval_logs_evalmode/rank_${expri}.log" | tail -2 | tr '\n' ' ')"
  else
    echo "[driver $(date '+%F %T')] rank $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/rank_${expri}.log" | tail -1)"
    return 1
  fi
}

# ---- 3 消融（ablation: 0=base / 1=NE / 2=proto_hard；proto_soft 砍出第一轮，
#      温度故事安全 KG §5.4 已覆盖，标准 KG 上富余时间再加）----
# train+rank 交错：每模型完成立即排名，结果流式落盘（避免"全训完再排"把风险堆到最后）。
train_one 0 "${DS}_RMPI_base";     [ $? -eq 0 ] || { echo "[driver] base 失败，中止链"; exit 1; }
rank_one  "${DS}_RMPI_base";       [ $? -eq 0 ] || { echo "[driver] base 排名失败，中止链"; exit 1; }
train_one 1 "${DS}_RMPI_NE";       [ $? -eq 0 ] || { echo "[driver] NE 失败，中止链"; exit 1; }
rank_one  "${DS}_RMPI_NE";         [ $? -eq 0 ] || { echo "[driver] NE 排名失败，中止链"; exit 1; }
train_one 2 "${DS}_RMPI_proto_hard"; [ $? -eq 0 ] || { echo "[driver] proto_hard 失败，中止链"; exit 1; }
rank_one  "${DS}_RMPI_proto_hard"; [ $? -eq 0 ] || { echo "[driver] proto_hard 排名失败，中止链"; exit 1; }

echo "[driver $(date '+%F %T')] ALL DONE (C5 WN18RR 零边：base/NE/proto_hard 训练+排名)"

# =============================================================================
# 待 C1 落地后追加「+侧信道」变体（A2_PLAN §2 侧信道定义，WN18RR 用 synset 名 char n-gram）：
#   1. 生成侧信道（离线，秒级）：
#      python side_channel_text.py --data_dir ../data/WN18RR_zec \
#        --out ../data/WN18RR_zec/side_channel.npz --nmin 3 --nmax 3 --hash_dim 64
#   2. C1 改码（datasets.py:128 / graph_classifier.py:16-17 / test_ranking_P.py:644 接 load_side_channel）后，
#      重训 NE 变体并排名，得「disclosing + 侧信道」行，与纯结构 NE 作对照消融（ZEC 协议第 6 条）。
# =============================================================================
