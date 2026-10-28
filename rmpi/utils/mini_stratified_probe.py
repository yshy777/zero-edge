"""mini_stratified_probe.py — 从 mini 排名输出分层复算 covered/uncovered MRR。

输入：security_kg_full_ind_RMPI_proto_hard_sc_ranking_head.txt（新 CWE-pool mini，
2,000 queries × 12 行）；test.txt 前 2,000 行（真尾）；product_fingerprint.json（覆盖）。
输出：mini 整体 MRR（sanity vs 0.3091）+ covered/uncovered 分层 MRR + 分解交叉验证。
"""
import json

DATA = 'C:/RMPI_Code/data/security_kg_full_ind'
RANK = f'{DATA}/security_kg_full_ind_RMPI_proto_hard_sc_ranking_head.txt'
TEST = f'{DATA}/test.txt'
FP = f'{DATA}/product_fingerprint.json'

with open(FP, encoding='utf-8') as f:
    covered_names = set(json.load(f).keys())

with open(TEST, encoding='utf-8') as f:
    test_lines = [l.rstrip('\n') for l in f if l.strip()][:2000]
true_tails = [l.split('\t')[2] for l in test_lines]
heads = [l.split('\t')[0] for l in test_lines]

with open(RANK, encoding='utf-8') as f:
    rank_lines = [l.rstrip('\n') for l in f if l.strip()]
assert len(rank_lines) == 2000 * 12, f"expected 24000 lines, got {len(rank_lines)}"

cov_rr, uncov_rr, all_rr = [], [], []
cov_n = 0
for i in range(2000):
    block = rank_lines[i * 12:(i + 1) * 12]
    tails = [l.split('\t')[2] for l in block]
    true = true_tails[i]
    # 真尾在块中首次出现的位置 = 排名（1-based）
    try:
        rank = tails.index(true) + 1
    except ValueError:
        rank = 12  # 真尾不在 12 候选内（不应发生；保守按最差）
    rr = 1.0 / rank
    all_rr.append(rr)
    if heads[i] in covered_names:
        cov_rr.append(rr)
        cov_n += 1
    else:
        uncov_rr.append(rr)

mrr = sum(all_rr) / len(all_rr)
mrr_cov = sum(cov_rr) / len(cov_rr)
mrr_unc = sum(uncov_rr) / len(uncov_rr)
c = cov_n / 2000
print(f"mini overall MRR (from ranking file): {mrr:.4f}   (log-reported: 0.3091)")
print(f"mini coverage: {cov_n}/2000 = {c:.4f}")
print(f"mini covered   MRR: {mrr_cov:.4f}  (n={len(cov_rr)})")
print(f"mini uncovered MRR: {mrr_unc:.4f}  (n={len(uncov_rr)})")
print()
print("--- decomposition cross-check ---")
# 全量分解分量（§8.2）：M_cov≈0.473, M_unc=0.1394
print(f"full-run components: M_cov=0.473, M_unc=0.1394")
print(f"predicted mini aggregate = c*M_cov + (1-c)*M_unc = {c*0.473 + (1-c)*0.1394:.4f}")
print(f"measured  mini aggregate = {mrr:.4f}")
print(f"mini-measured components: M_cov={mrr_cov:.4f}, M_unc={mrr_unc:.4f}")
print(f"predicted with mini's own c and FULL components: {c*0.473 + (1-c)*0.1394:.4f}")
print(f"predicted with FULL c and MINI components: {0.474*mrr_cov + 0.526*mrr_unc:.4f}")
