#!/usr/bin/env bash
# queue_full_ind_cwe_fix.sh — 等排名(rank_full_ind_serial_A.sh)全部结束后，自动起 full_ind cwe-fix 重训。
#
# 依赖关系：重训与排名结果无关，唯一依赖是「内存 + 不并发」（OOM 纪律）。
# CoMPILE 是排名驱动的最后一步，其 log 的终态标记 = 排名已结束。
# 终态后仍要等「0 残留 python + 内存≥2GB」才起（防 CoMPILE 收尾/孤儿树）。
#
# 只做重训这一步（10 epochs，~2 天）；dump_completion_cands + attack_centrality
# 是重训完成后的下一步（单独排）。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

COMILE_LOG=/c/RMPI_Code/baselines/CoMPILE/CoMPILE_github/test_auc_security_kg_full_ind_CoMPILE.log
TRAIN_LOG=train_security_kg_full_ind_RMPI_proto_hard_cwe.log

echo "[queue $(date '+%F %T')] waiting for ranking (base->proto_hard->CoMPILE) to finish ..."
# CoMPILE log 终态：Avg test Set Performance(成功) 或 Traceback/MemoryError/RuntimeError(失败)。
# 无论成功失败，只要 CoMPILE 这一最后步骤出终态，排名就结束了（重训不依赖排名结果）。
while ! grep -qaE "Avg test Set Performance|Traceback|MemoryError|RuntimeError" "$COMILE_LOG" 2>/dev/null; do
  sleep 300
done
echo "[queue $(date '+%F %T')] CoMPILE terminal marker seen; waiting for python to drain + RAM >= 2GB ..."

# 两查（OOM 纪律）：0 残留 python 且 内存≥2GB，两者同时满足才起。
while true; do
  LEFTOVER=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  LEFTOVER=${LEFTOVER:-0}
  FREEGB=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  if [ "$LEFTOVER" = "0" ] && awk -v g="$FREEGB" 'BEGIN{exit !(g>=2.0)}'; then
    echo "[queue $(date '+%F %T')] 0 leftover python, Free RAM = ${FREEGB} GB -> starting retrain"
    break
  fi
  echo "[queue $(date '+%F %T')] leftover_python=${LEFTOVER} freeGB=${FREEGB} (need 0 & >=2.0)"
  sleep 120
done

echo "[queue $(date '+%F %T')] full_ind cwe-fix retrain start (ablation=2, n_clusters=64, 10 epochs)"
python -u train.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard_cwe \
    --ablation 2 --n_clusters 64 --num_epochs 10 --batch_size 32 \
    --eval_every_iter 6000 --num_workers 0 --gpu -1 > "$TRAIN_LOG" 2>&1 &
PID=$!

# train.py 跑完会 hang 在非 daemon 线程（relaunch_inductive.sh 同款处理）。
while kill -0 "$PID" 2>/dev/null; do
  if grep -q "Epoch 10 " "$TRAIN_LOG" 2>/dev/null; then sleep 8; kill -9 "$PID" 2>/dev/null; break; fi
  if grep -qE "Traceback|RuntimeError|MemoryError|WinError|IndexError" "$TRAIN_LOG" 2>/dev/null; then break; fi
  sleep 30
done
wait "$PID" 2>/dev/null

if grep -q "Epoch 10 " "$TRAIN_LOG" 2>/dev/null; then
  echo "[queue $(date '+%F %T')] retrain OK: $(grep -aE 'best validation AUC' "$TRAIN_LOG" | tail -1)"
else
  echo "[queue $(date '+%F %T')] retrain FAILED: $(grep -aE 'RuntimeError|Traceback|Error|MemoryError' "$TRAIN_LOG" | tail -1)"
fi
echo "[queue $(date '+%F %T')] DONE (retrain only; 下一步 dump_completion_cands + attack_centrality)"
