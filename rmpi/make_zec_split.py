"""构造 ZEC split（C2）：标准 KG 的冷实体零边划分。

协议（A2_PLAN §2）：
  1. 冷实体集 C = 随机采样实体（默认 10%），inference 时零边。
  2. 从 train 里删掉所有与 C 关联的三元组 -> G_train（C 内实体零边）。
  3. 被删的三元组 = T_held，作为查询（h∈C 的 head-zero 查询为主）。
  4. 导出 train.txt / valid.txt / test.txt 到新目录。

输出新目录结构与原 KG 相同（train/valid/test.txt，行格式 head<TAB>relation<TAB>tail），
外加 cold_entities.txt（每行一个冷实体名）与 zec_meta.json（统计）。

纯离线、不跑训练，可与在跑排名链并行。
"""
import argparse
import json
import os
import random
from collections import defaultdict


def read_triples(path):
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 KG 目录（含 train/valid/test.txt）")
    ap.add_argument("--dst", required=True, help="输出目录")
    ap.add_argument("--cold_frac", type=float, default=0.1, help="冷实体占比")
    ap.add_argument("--valid_frac", type=float, default=0.1, help="T_held 里 valid 占比，其余 test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail_zero", action="store_true", help="额外导出 tail-zero 查询（t∈C）")
    args = ap.parse_args()

    random.seed(args.seed)

    train = read_triples(os.path.join(args.src, "train.txt"))
    valid = read_triples(os.path.join(args.src, "valid.txt"))
    test = read_triples(os.path.join(args.src, "test.txt"))

    # 实体集合（出现在任意三元组里）
    entities = set()
    for h, _, t in train + valid + test:
        entities.add(h)
        entities.add(t)

    # 采样冷实体（限制在有边的实体上，保证能删出查询）
    n_cold = max(1, int(len(entities) * args.cold_frac))
    cold = set(random.sample(sorted(entities), n_cold))

    # 划分：G_train = train 中不碰 C；T_held = 所有碰 C 的三元组
    g_train = []
    t_held = []
    for h, r, t in train:
        (t_held if (h in cold or t in cold) else g_train).append((h, r, t))
    for h, r, t in valid + test:
        if h in cold or t in cold:
            t_held.append((h, r, t))
        # 不碰 C 的 valid/test 三元组丢弃（ZEC 只测冷实体补全）

    # head-zero 查询（h∈C）为主；可选 tail-zero
    head_zero_q = [x for x in t_held if x[0] in cold]
    tail_zero_q = [x for x in t_held if x[2] in cold]

    random.shuffle(head_zero_q)
    n_valid = int(len(head_zero_q) * args.valid_frac)
    valid_q = head_zero_q[:n_valid]
    test_q = head_zero_q[n_valid:]

    os.makedirs(args.dst, exist_ok=True)

    def dump(path, triples):
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

    dump(os.path.join(args.dst, "train.txt"), g_train)
    dump(os.path.join(args.dst, "valid.txt"), valid_q)
    dump(os.path.join(args.dst, "test.txt"), test_q)
    with open(os.path.join(args.dst, "cold_entities.txt"), "w", encoding="utf-8") as f:
        for e in sorted(cold):
            f.write(e + "\n")
    if args.tail_zero:
        random.shuffle(tail_zero_q)
        dump(os.path.join(args.dst, "test_tail_zero.txt"), tail_zero_q)

    meta = {
        "src": args.src,
        "seed": args.seed,
        "cold_frac": args.cold_frac,
        "n_entities": len(entities),
        "n_cold": n_cold,
        "n_train_orig": len(train),
        "n_g_train": len(g_train),
        "n_t_held": len(t_held),
        "n_valid": len(valid_q),
        "n_test": len(test_q),
        "n_tail_zero": len(tail_zero_q) if args.tail_zero else None,
    }
    with open(os.path.join(args.dst, "zec_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
