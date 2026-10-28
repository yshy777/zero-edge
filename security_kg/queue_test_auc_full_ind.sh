#!/usr/bin/env bash
# queue_test_auc_full_ind.sh — full_ind 两个模型的 test_auc_F（只跑 AUC，不跑 ranking）。
# 等 full_ind proto_hard（pid 22328）退出后（base 会先退），依次跑 base(ab0) 再 proto_hard(ab2)。
# test 评测 db 用 RMPI_DB_ROOT 指 D 盘（C 盘仅剩 ~7G，装不下 ~7G 的 test db）。
# 与 queue_cwe_fix.sh 独立：它等同一个 pid，但跑的是 15k（security_kg_ind），这里跑 full_ind。
# full_ind 是祖先进程，用 tasklist 判活（Git Bash kill -0 探测不到 Windows 进程）。
set -u
set -o pipefail
cd /c/RMPI_Code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RMPI_DB_ROOT=D:/rmpi_subgraphs

LOG=security_kg/queue_test_auc_full_ind.log

alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -qi python; }

run_auc() {
  local expri="$1" ds="$2" abl="$3"
  echo "=== test_auc_F $expri (ablation $abl) start $(date '+%F %T') ==="
  python -u test_auc_F.py -e "$expri" -d "$ds" --model RMPI --ablation "$abl" \
      --runs 5 --num_workers 0 --gpu -1 \
      > eval_logs_evalmode/auc_${expri}.log 2>&1 \
      || { echo "test_auc_F $expri FAILED $(date '+%F %T')"; tail -30 eval_logs_evalmode/auc_${expri}.log; exit 1; }
  echo "--- $expri done $(date '+%F %T') ---"
  grep -aE "mean auc_roc|mean auc_pr" eval_logs_evalmode/auc_${expri}.log | tail -2
}

exec > "$LOG" 2>&1
echo "=== [test_auc_F full_ind] queue started $(date '+%F %T') — wait base(16832) + proto_hard(22328) ==="
# 等两个 full_ind 训练都退出（proto_hard 比 base 快，先等 base 再等 proto_hard 都退，避免 test_auc_F 撞上还在跑的 base）
while alive 16832; do sleep 60; done
echo "=== base DONE $(date '+%F %T') ==="
while alive 22328; do sleep 60; done
echo "=== proto_hard DONE $(date '+%F %T') — start test_auc_F (base then proto_hard) ==="

cd RMPI
mkdir -p eval_logs_evalmode
run_auc security_kg_full_ind_RMPI_base        security_kg_full_ind 0
run_auc security_kg_full_ind_RMPI_proto_hard  security_kg_full_ind 2
echo "=== TEST AUC FULL IND ALL DONE $(date '+%F %T') ==="
