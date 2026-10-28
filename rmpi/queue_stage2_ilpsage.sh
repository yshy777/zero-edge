#!/usr/bin/env bash
# queue_stage2_ilpsage.sh — 阶段二：ns50 安全链全部完成后执行（串行，不并发）。
#   1) WN18RR ZEC ns50 filtered 连坐（4 checkpoint，queue_zec_standard_kg_ns50.sh）
#   2) ILP-SAGE (Neurocomputing 2026) 全量训练 100 epoch + ZEC 候选评分
#      （复用安全链 base run 存盘的 51 行/块候选文件 → 与主表候选 bit 一致）
#   3) DuetGraph (NeurIPS 2025) transductive ZEC 训练 + 协议匹配评分
#      （CPU 适配配置：hidden 16 / B 64 / 10 epochs / limit_val_batches 0.05，
#       论文脚注披露；评分同样复用 base 候选文件）
# 环境：D:\ilpsage_venv（ILP-SAGE）/ D:\duetgraph_venv（DuetGraph）
# 烟雾已通（2026-10-25）：ILP-SAGE toy head MRR 0.0865 / tail 0.1368（玩具数字）。
# 纪律：串行、起前两查（残留 python + 内存≥2GB）。
set -u
cd /c/RMPI_Code/RMPI
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGDIR=eval_logs_evalmode
mkdir -p "$LOGDIR"

chk_mem() {
  local gb
  gb=$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreePhysicalMemory/1MB,1)' 2>/dev/null | tr -d '\r')
  echo "[stage2 $(date '+%F %T')] Free RAM = ${gb} GB"
  if awk -v g="$gb" 'BEGIN{exit !(g<2.0)}'; then
    echo "[stage2 $(date '+%F %T')] 内存不足(<2GB)，中止"
    exit 1
  fi
}

chk_leftover() {
  local n
  n=$(powershell.exe -NoProfile -Command '(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count' 2>/dev/null | tr -d '\r')
  n=${n:-0}
  echo "[stage2 $(date '+%F %T')] 残留 python 进程数 = ${n}"
  if [ "${n}" != "0" ]; then
    echo "[stage2 $(date '+%F %T')] 检测到残留 python，中止！先手动清。"
    exit 1
  fi
}

echo "=== stage2 启动 $(date '+%F %T') ==="

# 闸门 1：安全链 4 模型 ns50 全部完成
for m in base NE proto_hard proto_hard_sc; do
  if ! grep -qaE "Avg test Set Performance" "$LOGDIR/rank_security_kg_full_ind_RMPI_${m}_ns50.log"; then
    echo "[stage2 $(date '+%F %T')] ns50 安全链未完成（缺 $m），中止"
    exit 1
  fi
done
echo "[stage2 $(date '+%F %T')] 闸门1 通过：安全链 4 模型 ns50 全完成"

# 闸门 2：WN18RR ZEC ns50 filtered 连坐
bash /c/RMPI_Code/RMPI/queue_zec_standard_kg_ns50.sh || { echo "[stage2] WN18RR ns50 失败"; exit 1; }

# 闸门 3：ILP-SAGE 候选文件就绪（链 base run 产物，50 行/块 = 正样本第 1 行 + 49 负样本）
cd /c/RMPI_Code/data/security_kg_full_ind
for d in head tail; do
  f="security_kg_full_ind_RMPI_base_ranking_${d}.txt"
  n=$(wc -l < "$f" 2>/dev/null || echo 0)
  want=$((49535*50))
  if [ "$n" != "$want" ]; then
    echo "[stage2 $(date '+%F %T')] 候选文件 $f 行数 $n != $want，中止"
    exit 1
  fi
done
echo "[stage2 $(date '+%F %T')] 闸门3 通过：候选文件 50 行/块就绪"

# ILP-SAGE 全量训练（官方超参：lr 0.01 / 100 epoch / fanouts [15,10,5] / neg_ratio 1 / batch 256）
cd /c/RMPI_Code/baselines/ILP_SAGE_extract/ILP_SAGE-main
chk_leftover; chk_mem
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "[stage2 $(date '+%F %T')] ILP-SAGE train (security_kg_full_ind, 100 epochs)"
"D:/ilpsage_venv/Scripts/python.exe" -u zec/ilpsage_zec.py train \
  > /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_train_full.log 2>&1
if grep -qaE "\[ilpsage_zec\] saved" /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_train_full.log; then
  echo "[stage2 $(date '+%F %T')] ILP-SAGE train OK"
else
  echo "[stage2 $(date '+%F %T')] ILP-SAGE train FAILED: $(grep -aE 'Error|Traceback|MemoryError' /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_train_full.log | tail -1)"
  exit 1
fi

# ILP-SAGE ZEC 评分（候选文件 = base run 51 行/块；head/tail 双向，聚合同 test_ranking_P L830-876）
chk_leftover; chk_mem
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "[stage2 $(date '+%F %T')] ILP-SAGE rank (ZEC candidates)"
"D:/ilpsage_venv/Scripts/python.exe" -u zec/ilpsage_zec.py rank \
  > /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_rank_full.log 2>&1
if grep -qaE "Avg test Set Performance" /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_rank_full.log; then
  echo "[stage2 $(date '+%F %T')] ILP-SAGE rank OK: $(grep -aE '^(MRR|Hits@1|Hits@5|Hits@10)' /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_rank_full.log | tail -4 | tr '\n' ' ')"
else
  echo "[stage2 $(date '+%F %T')] ILP-SAGE rank FAILED: $(grep -aE 'Error|Traceback|MemoryError' /c/RMPI_Code/RMPI/eval_logs_evalmode/ilpsage_rank_full.log | tail -1)"
  exit 1
fi

# DuetGraph (NeurIPS 2025) transductive ZEC 训练（CPU 适配配置：hidden 16 / B 64 /
# 10 epochs / limit_val_batches 0.05 —— 论文脚注披露；rspmm 扩展经 zec/python_vs.bat
# 即 vcvars64 + ninja 编译/加载，torch cpp_extension 要求 cl.exe 在 PATH）。
chk_leftover; chk_mem
cd /c/RMPI_Code/baselines/DuetGraph/dual_pathway
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "[stage2 $(date '+%F %T')] DuetGraph train (security_kg_zec, 10 epochs, d=16 B=64)"
cmd //c "zec\\python_vs.bat -u main.py --accelerator cpu --devices 1 --data_path data/security_kg_zec --num_workers 0 --batch_size 64 --test_batch_size 16 --hidden_dim 16 --num_layer 3 --num_qk_layer 2 --num_v_layer 2 --num_heads 4 --drop 0.1 --loss_fn ce --num_negative_sample 7 --optimizer Adam --learning_rate 1e-4 --weight_decay 1e-4 --adversarial_temperature 1.0 --max_epochs 10 --check_val_every_n_epoch 5 --limit_val_batches 0.05 --seed 2023 --dataset security_kg_zec --checkpoint_save_path ckpt_duetgraph_zec" \
  > /c/RMPI_Code/RMPI/eval_logs_evalmode/duetgraph_train_full.log 2>&1
CKPT=$(ls -t /c/RMPI_Code/baselines/DuetGraph/dual_pathway/ckpt_duetgraph_zec/*/Dual-security_kg_zec-16-3-7-64x1_final.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
  echo "[stage2 $(date '+%F %T')] DuetGraph train OK: $CKPT"
else
  echo "[stage2 $(date '+%F %T')] DuetGraph train FAILED: $(grep -aE 'Error|Traceback|MemoryError|RuntimeError' /c/RMPI_Code/RMPI/eval_logs_evalmode/duetgraph_train_full.log | tail -1)"
  exit 1
fi

# DuetGraph 协议匹配评分（复用 base run 候选文件，51 行/块，同 ILP-SAGE 口径）
chk_leftover; chk_mem
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "[stage2 $(date '+%F %T')] DuetGraph rank (ZEC candidates)"
cmd //c "zec\\python_vs.bat -u zec/zec_score_duetgraph.py --ckpt $CKPT --data data/security_kg_zec --cand-head /c/RMPI_Code/data/security_kg_full_ind/security_kg_full_ind_RMPI_base_ranking_head.txt --cand-tail /c/RMPI_Code/data/security_kg_full_ind/security_kg_full_ind_RMPI_base_ranking_tail.txt --batch-size 32" \
  > /c/RMPI_Code/RMPI/eval_logs_evalmode/duetgraph_rank_full.log 2>&1
if grep -qaE "Avg test Set Performance" /c/RMPI_Code/RMPI/eval_logs_evalmode/duetgraph_rank_full.log; then
  echo "[stage2 $(date '+%F %T')] DuetGraph rank OK: $(grep -aE '^(MRR|Hits@1|Hits@5|Hits@10)' /c/RMPI_Code/RMPI/eval_logs_evalmode/duetgraph_rank_full.log | tail -4 | tr '\n' ' ')"
else
  echo "[stage2 $(date '+%F %T')] DuetGraph rank FAILED: $(grep -aE 'Error|Traceback|MemoryError' /c/RMPI_Code/RMPI/eval_logs_evalmode/duetgraph_rank_full.log | tail -1)"
  exit 1
fi

echo "[stage2 $(date '+%F %T')] ALL DONE (WN18RR ns50 + ILP-SAGE train/rank + DuetGraph train/rank)"
