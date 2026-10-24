"""量化退化率（C7）：零边查询占比 + 空 enclosing 占比，对照排名 MRR。

Lemma 1: 零边 head/tail => enclosing 子图（N(h) ∩ N(t)）确定性为空 => enclosing 类
scorer 塌缩成 tie-break 地板。本脚本对给定数据集统计 test 查询里：
  1. head-zero 率        —— h 在 train 图零边（无任何 incident train 三元组）
  2. tail-zero 率        —— t 在 train 图零边
  3. either-zero 率      —— h 或 t 至少一端零边（Lemma 1 的直接覆盖，即「退化率」）
  4. empty-enclosing 率  —— hop-1 无共同邻居（含零边情形，更精确的实际空 enclosing）

输入：每个数据集目录下 train.txt + test.txt，行格式 `head<TAB>relation<TAB>tail`。
输出：终端表格 + 可选 JSON（--json PATH）。

纯统计、只读、秒级，不与在跑的排名链冲突。
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def load_triples(path):
    """返回 (heads, tails) 字符串列表；空文件返回两个空列表。"""
    heads, tails = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            heads.append(parts[0])
            tails.append(parts[2])
    return heads, tails


def build_neighbors(train_path):
    """无向邻接：entity -> set(neighbor)，仅来自 train（inference 图）。"""
    nb = defaultdict(set)
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            h, t = parts[0], parts[2]
            nb[h].add(t)
            nb[t].add(h)
    return nb


def reach_within(nb, root, k):
    """root 在无向 train 图内 k 跳可达节点集（含 root 自身，dist=0）。"""
    seen = {root}
    frontier = {root}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in nb.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def quantify(data_dir, hop=2):
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    nb = build_neighbors(train_path)
    test_heads, test_tails = load_triples(test_path)

    n = len(test_heads)
    if n == 0:
        return None

    head_zero = tail_zero = either_zero = empty_enc = 0
    for h, t in zip(test_heads, test_tails):
        hz = (len(nb.get(h, ())) == 0)
        tz = (len(nb.get(t, ())) == 0)
        if hz:
            head_zero += 1
        if tz:
            tail_zero += 1
        if hz or tz:
            either_zero += 1
            empty_enc += 1          # 零边 => 交集必空（Lemma 1，任意 hop）
        else:
            # hop-k 无共同可达节点 => enclosing 交集空
            if reach_within(nb, h, hop).isdisjoint(reach_within(nb, t, hop)):
                empty_enc += 1

    return {
        "dataset": os.path.basename(os.path.normpath(data_dir)),
        "n_test": n,
        "n_train_entities": len(nb),
        "hop": hop,
        "head_zero": head_zero,
        "head_zero_rate": head_zero / n,
        "tail_zero": tail_zero,
        "tail_zero_rate": tail_zero / n,
        "either_zero": either_zero,
        "either_zero_rate": either_zero / n,
        "empty_enclosing": empty_enc,
        "empty_enclosing_rate": empty_enc / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dirs", nargs="+", help="含 train.txt/test.txt 的数据集目录")
    ap.add_argument("--hop", type=int, default=2, help="enclosing 子图跳数（匹配模型配置，默认 2）")
    ap.add_argument("--json", default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    rows = []
    for d in args.data_dirs:
        r = quantify(d, hop=args.hop)
        if r is None:
            print(f"[skip] {d}: 缺 train.txt/test.txt", file=sys.stderr)
            continue
        rows.append(r)

    if not rows:
        print("无有效数据集。", file=sys.stderr)
        sys.exit(1)

    # 终端表格
    hdr = (f"{'dataset':<28}{'n_test':>8}{'head0%':>9}{'tail0%':>9}"
           f"{'either0%':>10}{'emptyEnc%':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<28}{r['n_test']:>8}"
              f"{r['head_zero_rate']*100:>8.1f}%"
              f"{r['tail_zero_rate']*100:>8.1f}%"
              f"{r['either_zero_rate']*100:>9.1f}%"
              f"{r['empty_enclosing_rate']*100:>10.1f}%")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\nJSON -> {args.json}")


if __name__ == "__main__":
    main()
