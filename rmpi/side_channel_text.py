"""C4：标准 KG 的文本侧信道模块（实体名 char n-gram -> 向量）。

与安全 KG 的 CPE 产品指纹（Scheme 3 / fc_prod）同构：都是「图外属性 -> 固定维向量」，
供 re-anchoring 用（zero-edge 下唯一 head-specific 信号，break head-independence）。

本模块做两件事：
  1. build_text_side_channel：实体名 -> char n-gram 袋（或 feature-hash 到固定维），
     输出 (n_entities, dim) 稀疏矩阵，可存 npz。
  2. SideChannelText：torch 线性层，把 n-gram 袋投影到 out_dim（对应 fc_prod）。

名字来源（二选一）：
  --entity2text PATH  : 文件每行 `entity_id<TAB>name`（FB15k-237 官方 entity2text.txt 格式）
  否则               : 用实体 ID 字符串本身（inductive：unseen 实体的 ID 串在 inference 也可见）

纯离线、可并行；torch 可选（仅 SideChannelText 需要）。
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# 1. char n-gram 袋构建
# ---------------------------------------------------------------------------
def _char_ngrams(name, nmin, nmax):
    name = name.lower()
    if nmin > 1:
        # 首尾锚定（惯例，帮助捕获词边界）
        name = "<" + name + ">"
    grams = []
    for n in range(nmin, nmax + 1):
        for i in range(len(name) - n + 1):
            grams.append(name[i:i + n])
    return grams


def _stable_hash(s, dim):
    """跨进程稳定的 feature hash（不依赖 python hash 随机盐）。"""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % dim


def build_text_side_channel(entities, names, nmin=3, nmax=3, hash_dim=None):
    """entities: 有序实体名列表；names: {entity -> name_str}。

    返回 (sp.csr_matrix (n_entities, vocab_or_hash_dim), vocab 或 None)。
    hash_dim=None 时返回完整 vocab（一列一个 n-gram）；否则 feature-hash 到 hash_dim 列。
    """
    rows, cols, vals = [], [], []
    vocab = {} if hash_dim is None else None

    for i, e in enumerate(entities):
        name = names.get(e, e)  # 缺名字回退到 ID 字符串
        for g in _char_ngrams(name, nmin, nmax):
            if hash_dim is None:
                j = vocab.setdefault(g, len(vocab))
            else:
                j = _stable_hash(g, hash_dim)
            rows.append(i)
            cols.append(j)
            vals.append(1.0)

    n = len(entities)
    d = len(vocab) if hash_dim is None else hash_dim
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, d))
    if hash_dim is None:
        # 归一化：L2 行归一（缓解高频名主导）
        norm = np.sqrt(np.asarray(mat.power(2).sum(axis=1)).ravel())
        norm[norm == 0] = 1.0
        mat = mat.multiply(1.0 / norm[:, None]).tocsr()
    return mat, vocab


# ---------------------------------------------------------------------------
# 2. torch 侧信道线性层（对应 fc_prod）
# ---------------------------------------------------------------------------
class SideChannelText:
    def __init__(self, in_dim, out_dim):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def __call__(self, bag):  # bag: (batch, in_dim) dense 或 sparse
        if sp.issparse(bag):
            bag = self.torch.from_numpy(bag.toarray()).float()
        return self.fc(bag)


# ---------------------------------------------------------------------------
# 3. CLI：读数据集 + 名字 -> 构建 npz
# ---------------------------------------------------------------------------
def read_entities(data_dir):
    """从 train/valid/test.txt 收集全部实体（保持首次出现顺序）。"""
    seen, order = set(), []
    for fn in ("train.txt", "valid.txt", "test.txt"):
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                for e in (parts[0], parts[2]):
                    if e not in seen:
                        seen.add(e)
                        order.append(e)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--entity2text", default=None, help="`id<TAB>name` 文件；缺省用 ID 串")
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    ap.add_argument("--nmin", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=3)
    ap.add_argument("--hash_dim", type=int, default=None, help="feature-hash 到固定维（省内存）")
    args = ap.parse_args()

    entities = read_entities(args.data_dir)
    names = {}
    if args.entity2text:
        with open(args.entity2text, encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 2:
                    names[parts[0]] = parts[1].strip()

    mat, vocab = build_text_side_channel(entities, names, args.nmin, args.nmax, args.hash_dim)

    sp.save_npz(args.out, mat)
    # 行序 = entities 列表；供 load_side_channel 按名映射回全局 entity id
    with open(os.path.splitext(args.out)[0] + "_entities.txt", "w", encoding="utf-8") as f:
        for e in entities:
            f.write(e + "\n")
    if vocab is not None:
        vpath = os.path.splitext(args.out)[0] + "_vocab.json"
        with open(vpath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    print(f"entities={len(entities)} dim={mat.shape[1]} nnz={mat.nnz} -> {args.out}")


if __name__ == "__main__":
    main()
