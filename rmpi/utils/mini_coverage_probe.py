"""mini_coverage_probe.py — 统计 mini-probe 子集（test.txt 前 2,000 links）的侧信道覆盖率。

交叉验证 §8.2 分层分解（M_cov≈0.473）：用全量分解的 covered/uncovered 分量
预测 mini 子集的聚合 MRR，与实测 mini 聚合（0.3091）比对。
覆盖定义与 test_ranking_P.py 一致：头实体名在 product_fingerprint.json 中（非空向量）。
"""
import json
import os

DATA = 'C:/RMPI_Code/data/security_kg_full_ind'
FP = os.path.join(DATA, 'product_fingerprint.json')
TEST = os.path.join(DATA, 'test.txt')

with open(FP, encoding='utf-8') as f:
    fp = json.load(f)
covered_names = set(fp.keys())
print(f"fingerprint entities: {len(covered_names)}")

with open(TEST, encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]
print(f"test links total: {len(lines)}")

def cov_rate(subset):
    cov = sum(1 for l in subset if l.split('\t')[0] in covered_names)
    return cov, len(subset)

# 全量（sanity：应≈47.4%）
cov_all, n_all = cov_rate(lines)
print(f"FULL  test coverage: {cov_all}/{n_all} = {cov_all/n_all:.4f}")

# mini 子集（max_links=2000 取前 2,000）
sub = lines[:2000]
cov_sub, n_sub = cov_rate(sub)
c_sub = cov_sub / n_sub
print(f"MINI  coverage: {cov_sub}/{n_sub} = {c_sub:.4f}")

# 分解分量（全量阶梯，CWE-pool）：uncovered = proto_hard 0.1394；covered M_cov≈0.473
M_COV, M_UNC = 0.473, 0.1394
predicted = c_sub * M_COV + (1 - c_sub) * M_UNC
print(f"decomposition prediction for mini aggregate: {predicted:.4f}")
print(f"measured mini aggregate: 0.3091")
print(f"delta: {0.3091 - predicted:+.4f}")
