# compute_subset_mrr.py <test_txt> <ranking_head_txt> <ranking_tail_txt> [n_links]
# 从 test_ranking_P 保存的 ranking txt（每 link 12 行 = ns12 候选，按分数降序）
# 重算 MRR：默认全量（验证与官方 MRR 一致），传 n_links 只看前 N 条 link 的子集。
import sys

test_txt, rh, rt = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else None

def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n").split("\t") for l in f]

links = read_lines(test_txt)
if n:
    links = links[:n]

heads = read_lines(rh)  # 12 lines per link
tails = read_lines(rt)

assert len(heads) == 12 * len(links), f"head lines {len(heads)} != {12*len(links)}"
assert len(tails) == 12 * len(links), f"tail lines {len(tails)} != {12*len(links)}"
# 格式自检：第一组候选行与第一条 link 对齐（head 文件首行应等于 links[0][0] 的候选集合之一）
print("first link:", links[0])
# 实测格式：head 文件 = tail 侧排名（col0=真 head 固定，col2=候选 tail 按分数序）；
#           tail 文件 = head 侧排名（col2=真 tail 固定，col0=候选 head 按分数序）。

def mrr(cands, cand_col, true_col, side):
    rr = []
    ties = 0
    for i, l in enumerate(links):
        group = cands[i * 12:(i + 1) * 12]
        names = [g[cand_col] for g in group]
        tgt = l[true_col]
        if tgt not in names:
            rr.append(0.0)
            continue
        # 并列处理：与 test_ranking_P 一致取位置名次（按保存顺序即分数顺序）
        r = names.index(tgt) + 1
        rr.append(1.0 / r)
        if len(set(names)) < len(names):
            ties += 1
    return sum(rr) / len(rr), ties

mh, th = mrr(tails, 0, 0, "head")  # head 侧：候选在 tail 文件 col0
mt, tt = mrr(heads, 2, 2, "tail")  # tail 侧：候选在 head 文件 col2
print(f"n_links={len(links)}  head MRR={mh:.4f} (tie_groups={th})  tail MRR={mt:.4f} (tie_groups={tt})")
print(f"avg MRR={(mh + mt) / 2:.4f}")
