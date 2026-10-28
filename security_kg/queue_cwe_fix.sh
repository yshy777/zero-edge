#!/usr/bin/env bash
# queue_cwe_fix.sh — 并行队列（用户已拍板"无风险就并行"）：等 full_ind **base**（pid 16832，快）
#   退出后立即并行起轻量 retrain（~1-2GB，重叠 proto_hard 尾巴）；再等 **proto_hard**（22328）
#   退出；retrain 完成后 dump → ACO。全程写 security_kg/queue_cwe_fix.log。
# full_ind 是祖先进程，用 tasklist 判活（Git Bash kill -0 探测不到 Windows 进程）。
set -u
set -o pipefail
cd /c/RMPI_Code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOG=security_kg/queue_cwe_fix.log
EXPI=security_kg_ind_RMPI_proto_hard_cwe
DIR=data/security_kg_ind

alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -qi python; }

exec > "$LOG" 2>&1
echo "=== [parallel] queue started $(date '+%F %T') — wait base(16832) ==="
while alive 16832; do sleep 60; done
echo "=== base DONE $(date '+%F %T') — start retrain (parallel with proto_hard 22328) ==="

cd RMPI
TRAINLOG=train_${EXPI}.log
python -u train.py -d security_kg_ind -e "$EXPI" --ablation 2 --n_clusters 64 \
    --num_epochs 20 --batch_size 16 --eval_every_iter 455 --num_workers 0 --gpu -1 \
    > "$TRAINLOG" 2>&1 &
PID=$!
cd /c/RMPI_Code

echo "=== wait proto_hard(22328) $(date '+%F %T') ==="
while alive 22328; do sleep 60; done
echo "=== proto_hard DONE $(date '+%F %T') — wait retrain ==="

# retrain detect+kill（train.py 跑完 Epoch 20 可能 hang 在非 daemon 线程；正常 exit 则 kill -0 直接退）
cd RMPI
while kill -0 "$PID" 2>/dev/null; do
  if grep -q "Epoch 20 " "$TRAINLOG" 2>/dev/null; then sleep 8; kill -9 "$PID" 2>/dev/null; break; fi
  if grep -qE "Traceback|RuntimeError|MemoryError|WinError|IndexError" "$TRAINLOG" 2>/dev/null; then break; fi
  sleep 30
done
wait "$PID" 2>/dev/null
if grep -q "Epoch 20 " "$TRAINLOG" 2>/dev/null; then
  echo "retrain OK $(date '+%F %T'): $(grep -aE 'best validation AUC' "$TRAINLOG" | tail -1)"
else
  echo "RETRAIN FAILED $(date '+%F %T')"; tail -30 "$TRAINLOG"; exit 1
fi

echo "=== dump completion candidates $(date '+%F %T') ==="
python -u dump_completion_cands.py -d security_kg_ind -e "$EXPI" \
    --max_cves 0 --top_k 50 --chunk 50 \
    || { echo "DUMP FAILED $(date '+%F %T')"; exit 1; }
n_dumped=$(python -c "import json;print(len(json.load(open('../data/security_kg_ind/completion_cands_${EXPI}.json'))))")
echo "DUMP OK: $n_dumped unseen CVEs $(date '+%F %T')"
[ "$n_dumped" -gt 0 ] || { echo "DUMP EMPTY"; exit 1; }
cd ..

echo "=== ACO 4 ablations $(date '+%F %T') ==="
for ab in A A1 B C; do
  echo "--- ablation $ab ---"
  python security_kg/aco_completion.py --kg security_kg_ind --expri "$EXPI" \
      --ablation "$ab" --top_k 20 2>&1 | grep -aE "KEV ROC|Spearman|queries="
done
for ab in A A1 B C; do
  [ -f "$DIR/aco_risk_${ab}_${EXPI}.json" ] || { echo "ACO $ab MISSING OUTPUT"; exit 1; }
done
echo "=== queue ALL DONE $(date '+%F %T') ==="
