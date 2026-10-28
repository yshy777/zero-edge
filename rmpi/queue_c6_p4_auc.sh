#!/usr/bin/env bash
# queue_c6_p4_auc.sh — P4 AUC 复核（route-B：test_auc_F + 修复后 evaluator + head-cut 模型 strict=False 载旧权重）。
# 在 queue_c6_post_chain.sh 全部结束后手动启动（那条链里的 P4 是 TODO 空步骤）。
# 前置：test_auc_F.py 已加 is_big_dataset 注入（安全 KG 走稀疏路径，防 OOM）。
# 口径：--runs 1 --seed 41504 --neg 1 --hop 2 --num_workers 0 --gpu -1；test db 统一放
#       D:/rmpi_subgraphs（RMPI_DB_ROOT）：full_ind ab0/ab2 已存在直接复用，其余现生成。
# 产出（替换草稿 route-A 标记的 AUC）：
#   P4a ind 表 base/NE/proto_hard/proto_soft —— §5.4 阶梯 0.756→0.845→0.661
#   P4b 传导表 base/NE                     —— §5.4 0.746（proto 行见 P4d）
#   P4c full_ind 表 base/NE/proto_hard     —— §8.1 塌缩 0.552 + 全归纳 AUC 行
#   P4d 传导 proto_hard/proto_soft        —— 旧训练缓存已丢失，先用同 seed 重建缓存再跑；
#       结果须与旧值(0.840/0.650)比对合理性后决定是否采用（KMeans 是否复现旧分配存疑）。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RMPI_DB_ROOT=D:/rmpi_subgraphs
mkdir -p eval_logs_evalmode

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[p4 $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[p4 $(date '+%F %T')] 内存不足(<2GB)，中止"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[p4 $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[p4 $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

run_auc() {  # $1=dataset $2=expri $3=ablation
  local ds="$1" expri="$2" abl="$3"
  chk_leftover; chk_mem
  echo "[p4 $(date '+%F %T')] auc $expri (ab=$abl)"
  python -u test_auc_F.py -d "$ds" -e "$expri" --ablation "$abl" --runs 1 --seed 41504 \
    --num_workers 0 --gpu -1 \
    > "eval_logs_evalmode/auc_${expri}.log" 2>&1
  if grep -qaE "Avg test Set Performance -- mean auc_roc" "eval_logs_evalmode/auc_${expri}.log"; then
    echo "[p4 $(date '+%F %T')] auc $expri OK: $(grep -aE 'Avg test Set Performance -- mean auc_roc' "eval_logs_evalmode/auc_${expri}.log" | tail -1)"
  else
    echo "[p4 $(date '+%F %T')] auc $expri FAILED: $(grep -aE 'Error|Traceback|MemoryError' "eval_logs_evalmode/auc_${expri}.log" | tail -1)"
    return 1
  fi
}

# ---- P4a: ind 表（§5.4 阶梯 0.756→0.845→0.661 的 route-B 重核）----
echo "[p4 $(date '+%F %T')] === P4a: security_kg_ind ×4 ==="
run_auc security_kg_ind security_kg_ind_RMPI_base 0;        [ $? -eq 0 ] || exit 1
run_auc security_kg_ind security_kg_ind_RMPI_NE 1;          [ $? -eq 0 ] || exit 1
run_auc security_kg_ind security_kg_ind_RMPI_proto_hard 2;  [ $? -eq 0 ] || exit 1
run_auc security_kg_ind security_kg_ind_RMPI_proto_soft 3;  [ $? -eq 0 ] || exit 1

# ---- P4b: 传导表 base/NE（§5.4 0.746 重核）----
echo "[p4 $(date '+%F %T')] === P4b: security_kg(传导) base/NE ==="
run_auc security_kg security_kg_RMPI_base 0;                [ $? -eq 0 ] || exit 1
run_auc security_kg security_kg_RMPI_NE 1;                  [ $? -eq 0 ] || exit 1

# ---- P4c: full_ind 表（§8.1 塌缩 0.552 + 全归纳 AUC 行；ab0/ab2 db 已存在）----
echo "[p4 $(date '+%F %T')] === P4c: security_kg_full_ind ×3 ==="
run_auc security_kg_full_ind security_kg_full_ind_RMPI_base 0;       [ $? -eq 0 ] || exit 1
run_auc security_kg_full_ind security_kg_full_ind_RMPI_NE 1;         [ $? -eq 0 ] || exit 1
run_auc security_kg_full_ind security_kg_full_ind_RMPI_proto_hard 2; [ $? -eq 0 ] || exit 1

# ---- P4d: 传导 proto（缓存丢失 → 同 seed 重建；结果比对合理性后再采用）----
echo "[p4 $(date '+%F %T')] === P4d: 传导 proto（先重建 cluster 缓存）==="
chk_leftover; chk_mem
python -u - <<'PY' > eval_logs_evalmode/build_trans_cache.log 2>&1
import sys, os
sys.path.insert(0, os.path.abspath('.'))
class _P: pass
params = _P()
params.dataset = 'security_kg'
params.seed = 41504
params.main_dir = 'utils/..'
# 与 train.py 一致：entity2id = train+valid 实体空间（缓存按此空间落盘，评测侧按名 remap）
params.file_paths = {
    'train': os.path.abspath('../data/security_kg/train.txt'),
    'valid': os.path.abspath('../data/security_kg/valid.txt'),
}
from utils.cluster_assign import build_cluster_assignments
id2cluster, id2soft, n = build_cluster_assignments(params, n_clusters=64)
print(f'cache built: n_clusters={n}, entities={len(id2cluster)}, seen={(id2cluster >= 0).sum()}')
PY
if grep -qa "cache built" eval_logs_evalmode/build_trans_cache.log; then
  echo "[p4 $(date '+%F %T')] trans cache OK: $(grep -a 'cache built' eval_logs_evalmode/build_trans_cache.log | tail -1)"
else
  echo "[p4 $(date '+%F %T')] trans cache build FAILED: $(grep -aE 'Error|Traceback' eval_logs_evalmode/build_trans_cache.log | tail -1)"
  exit 1
fi
run_auc security_kg security_kg_RMPI_proto_hard 2;  [ $? -eq 0 ] || exit 1
run_auc security_kg security_kg_RMPI_proto_soft 3;  [ $? -eq 0 ] || exit 1

echo "[p4 $(date '+%F %T')] P4 ALL DONE (auc_*.log 已落盘 eval_logs_evalmode/)"
