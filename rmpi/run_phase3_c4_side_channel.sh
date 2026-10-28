#!/usr/bin/env bash
# run_phase3_c4_side_channel.sh — C4：WN18RR 侧信道消融（ZEC 协议第 6 条：有/无侧信道对照）。
#
# 目的：在标准 KG（WN18RR）上验证「disclosing 之外，侧信道（synset 名 char n-gram）
#   是冷实体恢复的第二张牌」。对照对象 = queue_zec_standard_kg.sh 的「纯结构」4 消融
#   （无 side_channel.npz → load_side_channel 返回 (None,0)，模型跳过侧信道注入）。
#
# ⚠️ 执行前置：必须等 C1（数据路径通用化）落地后——datasets.py / graph_classifier.py /
#   test_ranking_P.py 换用 utils.side_channel.load_side_channel（方案见 paper/C1_data_path_generalization.md）。
#   未落地前本脚本跑出的仍是「纯结构」数字，等于白跑。落地方案已拟好 diff，用户过目确认后我改码。
#
# 侧信道定义（A2_PLAN §2）：WN18RR 用 synset 名 char n-gram 袋（3-gram，feature-hash 到 64 维）。
#   生成脚本 side_channel_text.py 已测通（C4 前置）。
#
# 纪律：串行、一次一个 job、起前两查（FreeRAM>=2GB + 无残留 python）、关收割。
#   WN18RR 图小，workers=4 内存安全（与 queue_zec_standard_kg.sh 一致）。
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
TRAIN_ARGS="--num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1"
RANK_ARGS="-m sample --runs 5 --workers 4 --num_samples 50"

echo "=== 启动 C4 WN18RR +侧信道 消融（$DS）$(date '+%F %T') ==="

# ---- step0: 生成侧信道（离线秒级）----
echo "[driver $(date '+%F %T')] step0 生成 side_channel.npz（synset 名 3-gram hash 64 维）"
python side_channel_text.py --data_dir "../data/$DS" --out "../data/$DS/side_channel.npz" --nmin 3 --nmax 3 --hash_dim 64
if [ -f "../data/$DS/side_channel.npz" ]; then
  echo "[driver $(date '+%F %T')] side_channel.npz OK"
else
  echo "[driver $(date '+%F %T')] side_channel.npz FAILED，中止"
  exit 1
fi

# ---- 重训 +侧信道 变体（C1 落地后 load_side_channel 自动读 npz 注入）----
# 侧信道注入走 ablation≥1（disclosing）的路径；NE = disclosing+侧信道，proto_hard = disclosing+原型+侧信道。
train_one() {
  local abl="$1" expri="$2"
  chk_leftover; chk_mem
  echo "[driver $(date '+%F %T')] train $expri (ablation $abl, +side channel)"
  python -u train.py -d "$DS" -e "$expri" --ablation "$abl" $TRAIN_ARGS \
    > "train_${expri}.log" 2>&1
  if grep -qaE "best validation AUC" "train_${expri}.log"; then
    echo "[driver $(date '+%F %T')] train $expri OK: $(grep -aE 'best validation AUC' "train_${expri}.log" | tail -1)"
  else
    echo "[driver $(date '+%F %T')] train $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "train_${expri}.log" | tail -1)"
  fi
}

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
  fi
}

# 侧信道只对 disclosing 及以上有意义（enclosing 空子图侧信道也喂不进）；对照 = queue_zec 的纯结构 NE / proto_hard。
train_one 1 "${DS}_RMPI_NE_sc"
train_one 2 "${DS}_RMPI_proto_hard_sc"

rank_one "${DS}_RMPI_NE_sc"
rank_one "${DS}_RMPI_proto_hard_sc"

echo "[driver $(date '+%F %T')] C4 WN18RR +侧信道 DONE（对照 queue_zec 纯结构 NE/proto_hard 行）"
