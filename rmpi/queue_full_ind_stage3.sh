#!/usr/bin/env bash
# queue_full_ind_stage3.sh — 等排名(base→proto_hard→CoMPILE)全部结束后，自动跑 Stage 3 画像管线。
#
# 2026-10-08 重定义：跳过 cwe-fix 重训（cwe_rate 只需 top-1 CWE 身份，现有 proto_hard checkpoint 足够）。
# 排名后两步：dump_completion_cands（全 unseen CVE 补全候选，~1-1.5 天）→ risk_profile cands 版（分钟级）。
# 前置与旧队列一致：CoMPILE 终态 = 排名结束标记；0 残留 python + 内存≥2GB 才起（OOM 纪律）。
#
# ⚠️ 启动前提：queue_full_ind_cwe_fix.sh 的旧进程必须已停（旧队列会起 2 天 cwe-fix 重训，已废弃）。
set -u
cd /c/RMPI_Code/RMPI
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

COMILE_LOG=/c/RMPI_Code/baselines/CoMPILE/CoMPILE_github/test_auc_security_kg_full_ind_CoMPILE.log
CANDS=../data/security_kg_full_ind/completion_cands_security_kg_full_ind_RMPI_proto_hard.json
SMOKE_LOG=eval_logs_evalmode/dump_smoke_full_ind.log
DUMP_LOG=eval_logs_evalmode/dump_full_ind.log

echo "[queue $(date '+%F %T')] waiting for ranking (base->proto_hard->CoMPILE) terminal ..."
# CoMPILE log 终态：Avg test Set Performance(成功) 或 Traceback/MemoryError/RuntimeError(失败)。
while ! grep -qaE "Avg test Set Performance|Traceback|MemoryError|RuntimeError" "$COMILE_LOG" 2>/dev/null; do
  sleep 300
done
echo "[queue $(date '+%F %T')] CoMPILE terminal marker seen; waiting for python to drain + RAM >= 2GB ..."

while true; do
  LEFTOVER=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  LEFTOVER=${LEFTOVER:-0}
  FREEGB=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  if [ "$LEFTOVER" = "0" ] && awk -v g="$FREEGB" 'BEGIN{exit !(g>=2.0)}'; then
    break
  fi
  sleep 300
done

echo "[queue $(date '+%F %T')] smoke dump (200 CVEs) ..."
python -u dump_completion_cands.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard \
  --max_cves 200 --top_k 50 --chunk 50 > "$SMOKE_LOG" 2>&1
grep -aE "\[dump\] wrote|Error|Traceback" "$SMOKE_LOG" | tail -n 3

echo "[queue $(date '+%F %T')] full dump (all unseen CVEs, ~1-1.5 days) ..."
python -u dump_completion_cands.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard \
  --max_cves 0 --top_k 50 --chunk 50 > "$DUMP_LOG" 2>&1
grep -aE "\[dump\] wrote|Error|Traceback" "$DUMP_LOG" | tail -n 3

if [ -f "$CANDS" ]; then
  echo "[queue $(date '+%F %T')] risk_profile (cands mode) ..."
  cd /c/RMPI_Code
  python security_kg/risk_profile.py --kg security_kg_full_ind --cwe_src cands --cands "$CANDS"
else
  echo "[queue $(date '+%F %T')] CANDS NOT FOUND: $CANDS — risk_profile skipped"
fi
echo "[queue $(date '+%F %T')] Stage3 queue done."
