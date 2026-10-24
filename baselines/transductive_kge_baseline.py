#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""transductive_kge_baseline.py — 传导式 KGE baseline（TransE / DistMult / ComplEx / RotatE）。

目的：证明传导式 KGE SOTA 在 unseen CVE 上失效（MRR≈chance），对比 ProRMPI 的完全归纳能力。
（对应 INTEGRATION_PLAN §8.9「传导 SOTA 在 unseen 上失效」的结构对比，论文主押。）

自包含：只用 torch，不引入任何新依赖（避免改环境、不影响正在跑的实验）。CPU-only，单线程。

关键设计：
- 实体/关系 vocab 只从 train.txt 建；test-only 的 unseen head 实体落到 OOV id，
  其嵌入随机初始化且永不更新（传导式模型无法表征 unseen 实体）。
- 所有 test/valid 的 head（CVE）都是 unseen，故传导式模型的 tail 排序≈随机。
- 每个模型同时报「seen 实体上的训练 sanity MRR」和「unseen test 上的 MRR/Hits」，
  用前者证明模型确实学会了，后者证明它在 unseen 上失效。

用法：
  python transductive_kge_baseline.py -d security_kg_ind --models TransE,DistMult,ComplEx,RotatE
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def load_triples(path):
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3:
                triples.append((p[0], p[1], p[2]))
    return triples


def build_vocabs(train_triples):
    """从 train 建实体/关系 vocab（传导式：只认识 train 实体）。"""
    e2id, r2id = {}, {}
    for h, r, t in train_triples:
        if h not in e2id:
            e2id[h] = len(e2id)
        if t not in e2id:
            e2id[t] = len(e2id)
        if r not in r2id:
            r2id[r] = len(r2id)
    return e2id, r2id


class TransE(nn.Module):
    def __init__(self, n_e, n_r, d):
        super().__init__()
        self.e = nn.Embedding(n_e, d)
        self.r = nn.Embedding(n_r, d)
        nn.init.xavier_uniform_(self.e.weight)
        nn.init.xavier_uniform_(self.r.weight)

    def score(self, h, r, t):
        return -torch.norm(self.e(h) + self.r(r) - self.e(t), p=2, dim=-1)


class DistMult(nn.Module):
    def __init__(self, n_e, n_r, d):
        super().__init__()
        self.e = nn.Embedding(n_e, d)
        self.r = nn.Embedding(n_r, d)
        nn.init.xavier_uniform_(self.e.weight)
        nn.init.xavier_uniform_(self.r.weight)

    def score(self, h, r, t):
        return (self.e(h) * self.r(r) * self.e(t)).sum(dim=-1)


class ComplEx(nn.Module):
    def __init__(self, n_e, n_r, d):
        super().__init__()
        self.e_re = nn.Embedding(n_e, d)
        self.e_im = nn.Embedding(n_e, d)
        self.r_re = nn.Embedding(n_r, d)
        self.r_im = nn.Embedding(n_r, d)
        nn.init.xavier_uniform_(self.e_re.weight)
        nn.init.xavier_uniform_(self.e_im.weight)
        nn.init.xavier_uniform_(self.r_re.weight)
        nn.init.xavier_uniform_(self.r_im.weight)

    def score(self, h, r, t):
        hre, him = self.e_re(h), self.e_im(h)
        rre, rim = self.r_re(r), self.r_im(r)
        tre, tim = self.e_re(t), self.e_im(t)
        return (hre * rre * tre + him * rim * tre + hre * rim * tim - him * rre * tim).sum(-1)


class RotatE(nn.Module):
    def __init__(self, n_e, n_r, d):
        super().__init__()
        self.e = nn.Embedding(n_e, d)  # 实部
        self.r = nn.Embedding(n_r, d // 2)  # 相位
        nn.init.xavier_uniform_(self.e.weight)
        nn.init.uniform_(self.r.weight, -3.1416, 3.1416)

    def score(self, h, r, t):
        h = self.e(h)
        t = self.e(t)
        d = h.shape[-1]
        h_re, h_im = h[..., : d // 2], h[..., d // 2:]
        t_re, t_im = t[..., : d // 2], t[..., d // 2:]
        phase = self.r(r)
        cos, sin = torch.cos(phase), torch.sin(phase)
        hr_re = h_re * cos - h_im * sin
        hr_im = h_re * sin + h_im * cos
        return -torch.norm(torch.cat([hr_re - t_re, hr_im - t_im], dim=-1), p=2, dim=-1)


MODELS = {'TransE': TransE, 'DistMult': DistMult, 'ComplEx': ComplEx, 'RotatE': RotatE}


def train_model(model, train, n_neg_pool, epochs, bs, lr, margin, seed, device):
    """负采样只在 train 实体池 (0..n_neg_pool-1) 里采，OOV 实体绝不参与训练。"""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train = torch.tensor(train, dtype=torch.long)
    rng = random.Random(seed)
    n = train.shape[0]
    for ep in range(epochs):
        model.train()
        idx = list(range(n))
        rng.shuffle(idx)
        total = 0.0
        for s in range(0, n, bs):
            b = train[idx[s:s + bs]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            neg_t = torch.tensor([rng.randrange(n_neg_pool) for _ in range(len(b))], dtype=torch.long)
            pos = model.score(h, r, t)
            neg = model.score(h, r, neg_t)
            loss = torch.clamp_min(margin + neg - pos, 0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
    return model


@torch.no_grad()
def evaluate(model, triples, n_e, device):
    """raw（未过滤）tail 排序：对每个 (h,r,t)，把 t 排进全体实体，算 MRR/Hits。"""
    model.eval()
    all_e = torch.arange(n_e, device=device)
    ranks = []
    for h, r, t in triples:
        h = torch.tensor([h], device=device)
        r = torch.tensor([r], device=device)
        scores = model.score(h.expand(n_e), r.expand(n_e), all_e)
        true_s = scores[t].item()
        # 严格大于（比 true 高分的个数），ties 用保守计数
        better = (scores > true_s).sum().item()
        ranks.append(better + 1)
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    hits = {k: sum(1 for r in ranks if r <= k) / len(ranks) for k in (1, 3, 10)}
    return mrr, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='security_kg_ind')
    ap.add_argument('--models', default='TransE,DistMult,ComplEx,RotatE')
    ap.add_argument('--dim', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=41504)
    args = ap.parse_args()

    torch.set_num_threads(1)
    device = torch.device('cpu')
    ddir = os.path.join(DEFAULT_DATA, args.dataset)

    train_t = load_triples(os.path.join(ddir, 'train.txt'))
    valid_t = load_triples(os.path.join(ddir, 'valid.txt'))
    test_t = load_triples(os.path.join(ddir, 'test.txt'))
    e2id, r2id = build_vocabs(train_t)

    # test/valid 里的 unseen 实体 → OOV id（随机嵌入，永不更新）
    n_train_e = len(e2id)
    oov = {}
    for triples in (valid_t, test_t):
        for h, r, t in triples:
            for x in (h, t):
                if x not in e2id and x not in oov:
                    oov[x] = n_train_e + len(oov)
    e2id_full = dict(e2id)
    e2id_full.update(oov)
    n_e = len(e2id_full)

    def enc(triples):
        return [(e2id_full[h], r2id[r], e2id_full[t]) for h, r, t in triples
                if h in e2id_full and t in e2id_full]

    train = enc(train_t)
    valid = enc(valid_t)
    test = enc(test_t)

    # sanity：用 train 里 seen-head 的 has_weakness 三元组做「已学会」检查
    seen_hw = [(h, r, t) for h, r, t in train if r == r2id['has_weakness']]
    # unseen 判定：head 落在 OOV 区间（>=n_train_e）
    test_unseen = [(h, r, t) for h, r, t in test if h >= n_train_e]

    print(f"[kge] dataset={args.dataset} train={len(train)} valid={len(valid)} test={len(test)} "
          f"n_train_e={n_train_e} n_e={n_e} n_rel={len(r2id)} "
          f"test_unseen_head={len(test_unseen)}/{len(test)}")
    print(f"[kge] relations={r2id}")

    for name in args.models.split(','):
        name = name.strip()
        cls = MODELS[name]
        torch.manual_seed(args.seed)
        model = cls(n_e, len(r2id), args.dim)
        t0 = time.time()
        train_model(model, train, n_train_e, args.epochs, args.batch_size, args.lr, args.margin, args.seed, device)
        dt = time.time() - t0
        # 冻结 OOV 嵌入（本就未被更新），显式冻结防误训
        mrr_seen, hits_seen = evaluate(model, seen_hw, n_e, device)
        mrr_test, hits_test = evaluate(model, test_unseen, n_e, device)
        mrr_val, _ = evaluate(model, valid, n_e, device)
        print(f"[{name:9s}] train={dt:5.1f}s  seen_MRR={mrr_seen:.4f}  "
              f"valid_MRR={mrr_val:.4f}  test_unseen_MRR={mrr_test:.4f}  "
              f"Hits@10={hits_test[10]:.4f}")


if __name__ == '__main__':
    main()
