# zec/ilpsage_zec.py — ILP-SAGE (Neurocomputing 2026) on the ZEC protocol
#
# 运行环境: D:\ilpsage_venv (py3.9, torch 2.4.0+cpu, dgl 2.2.1)
#   D:\ilpsage_venv\Scripts\python.exe zec\ilpsage_zec.py train   # 训练（官方超参）
#   D:\ilpsage_venv\Scripts\python.exe zec\ilpsage_zec.py rank    # ZEC 候选评分
#
# 协议（与 test_ranking_P.py -m sample 对齐，见 rmpi-ns50-baseline-oct25-run memory）：
#   - 训练：官方 create_dataloader 训练分支逐行复用（fanouts=[15,10,5]、neg_ratio=1、
#     batch 1024、lr 0.01、epochs 100、Adam、BCE）。采样图 train-only（ZEC 纪律）。
#   - 评分：候选列表复用 ns50 链存盘的 *_ranking_head.txt / *_ranking_tail.txt
#     （每 link 50 行 = 正样本第 1 行 + 49 负样本，seed 41504 固定 → 与主表候选 bit 一致）。
#     对每个 (h, t) 候选对走官方评分前向（predictor(h_emb * t_emb)，模型无关系输入），
#     按分数降序排名，MRR/Hits 聚合复刻 test_ranking_P.py L830-876。
#   - 种子：torch/np 手动种 41504（官方代码未种；固定 RNG 流不改算法，披露）。
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dgl
from src.model import GraphSAGE
from src.training import train
from zec.ilpsage_zec_dataset import (
    KnowledgeGraphDatasetZEC, create_train_dataloader, create_score_dataloader)

DATA_DIR = r"C:\RMPI_Code\data\security_kg_full_ind"
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ilpsage_zec_model.pt")
SEED = 41504
FANOUTS = [15, 10, 5]  # 官方 train fanouts（模型 2 层，官方前向只消费其中 2 个 block）

_DATA_DIR = DATA_DIR
_CKPT = CKPT


def build_dataset(node_feat=None):
    return KnowledgeGraphDatasetZEC(
        "security_kg_full_ind",
        {"train": os.path.join(_DATA_DIR, "train.txt"),
         "test": os.path.join(_DATA_DIR, "test.txt")},
        node_feat=node_feat)


def cmd_train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset = build_dataset()
    device = torch.device("cpu")
    in_size = dataset.feature.size("node", None, "feat")[0]
    model = GraphSAGE(in_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_dataloader = create_train_dataloader(
        dataset, device, batch_size=1024, fanouts=FANOUTS, negative_ratio=1)
    print(f"[ilpsage_zec] train: {dataset.dgl_graphs['train'].num_edges()} edges, "
          f"{dataset.dgl_graphs['train'].num_nodes()} entities, epochs=100", flush=True)
    train(model, dataset, train_dataloader, device, optimizer, 100)
    # 节点随机特征与 ckpt 一起存盘：rank 必须复用同一份特征（原实现各 randn 一份，
    # rank 的嵌入与 predictor 不匹配 → 分数全塌缩成常数）。
    torch.save({"state_dict": model.state_dict(),
                "entity2id": dataset.entity2id,
                "relation2id": dataset.relation2id,
                "node_feat": dataset.dgl_graphs["train"].ndata["feat"].numpy()}, _CKPT)
    print(f"[ilpsage_zec] saved {_CKPT}", flush=True)


def read_candidate_file(path, entity2id, block_size=50):
    """每 block_size 行一块（正样本第 1 行 + block_size-1 负样本）→ [(h_id, t_id) x block_size]。
    block_size=50 对应 ns50 协议；block_size=12 对应主表 ns12 协议（同 seed 同采样器）。"""
    pairs, blocks, cur = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            h, t = entity2id.get(parts[0]), entity2id.get(parts[2])
            if h is None or t is None:
                raise RuntimeError(f"OOV candidate: {line.strip()}")
            cur.append((int(h), int(t)))
            if len(cur) == block_size:
                blocks.append(cur)
                cur = []
    if cur:
        raise RuntimeError(f"candidate file not aligned: {len(cur)} trailing rows")
    return blocks


def score_pairs(dataset, model, device, pairs, batch_size=256):
    """pairs: [(h_id, t_id)] 全量种子对 → 依序 logits（官方 prediction_model 前向）"""
    dataloader = create_score_dataloader(
        dataset, device, pairs, batch_size=batch_size, fanouts=FANOUTS)
    model.eval()
    logits_list = []
    with torch.no_grad():
        for data in dataloader:
            compacted_seeds = data.compacted_seeds.T.to(device)
            x = data.node_features["feat"].to(device)
            blocks = [b.to(device) for b in data.blocks]
            y = model(blocks, x)
            src = compacted_seeds[0].long()
            dst = compacted_seeds[1].long()
            logit = model.predictor(y[src] * y[dst]).squeeze().detach()
            logits_list.append(logit)
    return torch.cat(logits_list, dim=0).numpy()


def cmd_rank(block_size=50):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ckpt = torch.load(_CKPT, map_location="cpu")
    dataset = build_dataset(node_feat=ckpt.get("node_feat"))
    device = torch.device("cpu")
    model = GraphSAGE(dataset.feature.size("node", None, "feat")[0]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    entity2id = ckpt["entity2id"]

    ranks = []
    for direction in ("head", "tail"):
        fname = f"security_kg_full_ind_RMPI_base_ranking_{direction}.txt"
        fpath = os.path.join(_DATA_DIR, fname)
        blocks = read_candidate_file(fpath, entity2id, block_size=block_size)
        pairs = [p for block in blocks for p in block]
        logits = score_pairs(dataset, model, device, pairs)
        logits = logits.reshape(len(blocks), block_size)
        # 诚实并列口径（同 test_ranking_P L830-876 / DuetGraph scorer）：并列计入，
        # rank = 1 + #(j≠0 且 score_j >= score_true)。塌缩块 → rank 50 → 地板 1/50。
        # （原 argsort 位置口径在并列时把第 0 列真值排第一 → 塌缩块 rank 1 → 0.9978 伪影。）
        pos = (logits[:, 1:] >= logits[:, 0:1]).sum(axis=1) + 1
        ranks.extend(pos.tolist())
        mrr_dir = float(np.mean(1.0 / pos))
        print(f"[ilpsage_zec] {direction}: n_links={len(blocks)} "
              f"block_size={block_size} MRR={mrr_dir:.4f} Hits@1={float(np.mean(pos <= 1)):.4f} "
              f"Hits@5={float(np.mean(pos <= 5)):.4f} Hits@10={float(np.mean(pos <= 10)):.4f}",
              flush=True)

    ranks = np.asarray(ranks, dtype=float)
    print("\nAvg test Set Performance ")
    print(f"MRR: {np.mean(1.0 / ranks): .4f}")
    print(f"Hits@1  : {np.mean(ranks <= 1): .4f}")
    print(f"Hits@5  : {np.mean(ranks <= 5): .4f}")
    print(f"Hits@10 : {np.mean(ranks <= 10): .4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "rank"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--block-size", type=int, default=50,
                    help="候选块行数：50=ns50 协议，12=主表 ns12 协议")
    args = ap.parse_args()
    _DATA_DIR = args.data_dir
    _CKPT = args.ckpt
    if args.cmd == "train":
        cmd_train()
    else:
        cmd_rank(block_size=args.block_size)
