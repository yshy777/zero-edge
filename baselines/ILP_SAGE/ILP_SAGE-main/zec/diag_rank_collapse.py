# zec/diag_rank_collapse.py — 诊断 ILP-SAGE 0.9978：打印前几块 logits 分布
# 运行: D:\ilpsage_venv\Scripts\python.exe -u zec/diag_rank_collapse.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zec.ilpsage_zec import build_dataset, read_candidate_file, score_pairs  # noqa: E402
from src.model import GraphSAGE  # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ilpsage_zec_model.pt")
DATA_DIR = r"C:\RMPI_Code\data\security_kg_full_ind"

torch.manual_seed(41504)
np.random.seed(41504)
ckpt = torch.load(CKPT, map_location="cpu")
# 必须复用 ckpt 里的节点特征（同 cmd_rank），否则特征重生成 → 嵌入与 predictor
# 不匹配 → 全塌缩，诊断的是伪影而不是真实状态。
dataset = build_dataset(node_feat=ckpt.get("node_feat"))
device = torch.device("cpu")
model = GraphSAGE(dataset.feature.size("node", None, "feat")[0]).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
entity2id = ckpt["entity2id"]

for direction in ("head", "tail"):
    fpath = os.path.join(DATA_DIR, f"security_kg_full_ind_RMPI_base_ranking_{direction}.txt")
    blocks = read_candidate_file(fpath, entity2id)[:3]
    pairs = [p for block in blocks for p in block]
    logits = score_pairs(dataset, model, device, pairs)
    logits = logits.reshape(len(blocks), 50)
    print(f"=== {direction} 前 3 块 logits ===")
    for i in range(3):
        print(f"block{i}: true={logits[i,0]:.6f}  mean={logits[i,1:].mean():.6f} "
              f"min={logits[i].min():.6f} max={logits[i].max():.6f} "
              f"std={logits[i].std():.6f}  #(j>=true)={int((logits[i,1:] >= logits[i,0]).sum())}")
