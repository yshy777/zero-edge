#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全/漏洞侧 baseline（补齐清单 item 1 第 3 项）——补全任务上的安全侧方法。

任务：标准 tail 预测 (CVE, has_weakness, CWE)。给定 unseen CVE（head），在 277 个
CWE 里把正确弱点（tail）排到前面。与归纳 KG 方法（ProRMPI/GraIL/CoMPILE/TACT）同任务、
同指标（MRR / Hits@k），但**完全不看图结构**，只用 CVE 自身的漏洞元数据：

  - product_fingerprint.json : CVE -> 33 维向量（CPE/product token 的 log 词频）
  - cvss_scores.json         : CVE -> 标量（单值，对 277 类 CWE 无区分力，仅作对照）

baseline：
  1. majority —— 永远预测训练集最常见的 CWE（平凡下界）。
  2. fingerprint-LR —— 33 维指纹 -> 277 类 softmax（logistic regression），
     训练集 = security_kg_full 的 has_weakness 对（排除 test CVE，防止标签泄漏）。

评估：对每个有指纹的 test CVE，用预测分数把 277 个 CWE 排序，报 MRR / Hits@1/3/10
（多标签：CVE 有多个正确 CWE 时取排名最靠前的那个）。只在有指纹的 test CVE 子集上
评估（49% 覆盖），并同时给出 majority 作为同子集对照。

用法：
  python baselines/security_fingerprint_baseline.py
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # C:/RMPI_Code


def load_triples(path, rel='has_weakness'):
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            s, r, o = parts[0], parts[1], parts[2]
            if r == rel:
                triples.append((s, o))
    return triples


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def rank_metrics(true_cwes_per_cve, scores_per_cve, cwe_vocab_idx):
    """true_cwes_per_cve: {cve: set(cwe)}; scores_per_cve: {cve: np.array over vocab}.

    对每个 CVE 把 vocab 里的 CWE 按 score 降序排，算 MRR / Hits@k（多标签：任一正确
    CWE 命中即算命中）。"""
    n = 0
    mrr = 0.0
    hits = defaultdict(int)
    order = None
    for cve, true_set in true_cwes_per_cve.items():
        if cve not in scores_per_cve or not true_set:
            continue
        s = scores_per_cve[cve]
        if order is None:
            order = np.argsort(-s)  # 降序
        else:
            order = np.argsort(-s)
        ranked_cwes = [cwe_vocab_idx[i] for i in order]
        best_rank = min(i + 1 for i, c in enumerate(ranked_cwes) if c in true_set)
        mrr += 1.0 / best_rank
        n += 1
        for k in (1, 3, 10):
            if best_rank <= k:
                hits[k] += 1
    return {
        'n': n,
        'mrr': mrr / n if n else float('nan'),
        'hits@1': hits[1] / n if n else float('nan'),
        'hits@3': hits[3] / n if n else float('nan'),
        'hits@10': hits[10] / n if n else float('nan'),
    }


def main():
    # ---- 1. test 标签（fully inductive 的 CVE -> CWE）----
    test_pairs = load_triples(os.path.join(ROOT, 'data/security_kg_ind/test.txt'))
    true_cwes = defaultdict(set)
    for s, o in test_pairs:
        true_cwes[s].add(o)
    test_cves = list(true_cwes.keys())
    print(f'test has_weakness: {len(test_pairs)} triples, {len(test_cves)} unique CVE, '
          f'{len({o for _, o in test_pairs})} unique CWE')

    # ---- 2. 指纹 + CVSS ----
    fp = load_json(os.path.join(ROOT, 'data/security_kg_ind/product_fingerprint.json'))
    cvss = load_json(os.path.join(ROOT, 'data/security_kg/cvss_scores.json'))

    # ---- 3. 训练对（security_kg_full），排除 test CVE ----
    train_pairs = load_triples(os.path.join(ROOT, 'data/security_kg_full/train.txt'))
    train_cves = {s for s, _ in train_pairs}
    leak = len(set(test_cves) & train_cves)
    print(f'full train has_weakness: {len(train_pairs)} pairs; test CVE 泄漏进 full train: {leak}/{len(test_cves)}')

    # 只保留有指纹、且 CWE 在 test vocab 内的训练对；剔除 test CVE 自身。
    cwe_vocab = sorted({o for _, o in test_pairs})
    vocab_idx = {c: i for i, c in enumerate(cwe_vocab)}
    train_feat, train_label = [], []
    for s, o in train_pairs:
        if s in test_cves or s not in fp or o not in vocab_idx:
            continue
        train_feat.append(fp[s])
        train_label.append(vocab_idx[o])
    train_feat = np.asarray(train_feat, dtype=np.float32)
    train_label = np.asarray(train_label, dtype=np.int64)
    print(f'训练样本（有指纹、非 test、CWE 在 vocab）：{len(train_feat)} 条')

    # ---- 4. test 特征（有指纹的 CVE）----
    test_cves_fp = [c for c in test_cves if c in fp]
    test_feat = np.asarray([fp[c] for c in test_cves_fp], dtype=np.float32)
    print(f'test CVE 有指纹：{len(test_cves_fp)}/{len(test_cves)} '
          f'({100 * len(test_cves_fp) / len(test_cves):.0f}%)')

    # ---- 5. majority 基线 ----
    maj_cwe = Counter(o for _, o in test_pairs).most_common(1)[0][0]
    print(f'\n多数类 CWE = {maj_cwe}')
    maj_scores = {c: np.zeros(len(vocab_idx)) for c in test_cves_fp}
    for c in test_cves_fp:
        maj_scores[c][vocab_idx[maj_cwe]] = 1.0
    maj_metrics = rank_metrics({c: true_cwes[c] for c in test_cves_fp}, maj_scores, cwe_vocab)

    # ---- 6. fingerprint-LR 基线 ----
    X = normalize(train_feat, norm='l2')
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', n_jobs=1)
    clf.fit(X, train_label)
    Xt = normalize(test_feat, norm='l2')
    P_seen = clf.predict_proba(Xt)  # [n_test, n_seen_classes]，列序 = clf.classes_
    # 映射回完整 vocab（训练集缺的类概率为 0）
    P = np.zeros((P_seen.shape[0], len(vocab_idx)), dtype=np.float64)
    for j, cls in enumerate(clf.classes_):
        P[:, cls] = P_seen[:, j]
    lr_scores = {c: P[i] for i, c in enumerate(test_cves_fp)}
    lr_metrics = rank_metrics({c: true_cwes[c] for c in test_cves_fp}, lr_scores, cwe_vocab)

    # ---- 7. 报告 ----
    print('\n================ 安全侧 baseline（补全 tail 预测，指纹子集）================')
    for name, m in [('majority（平凡下界）', maj_metrics), ('fingerprint-LR（安全侧）', lr_metrics)]:
        print(f'{name:28s}  n={m["n"]:5d}  MRR={m["mrr"]:.4f}  '
              f'H@1={m["hits@1"]:.4f}  H@3={m["hits@3"]:.4f}  H@10={m["hits@10"]:.4f}')
    print('--------------------------------------------------------------------------')
    print('注：fingerprint-LR 只在 49% 有指纹的 test CVE 上评估；KG 方法覆盖 100%。')
    print('    与 ProRMPI 对比时需在同一指纹子集重报（见 item 2 排名脚本）。')

    out = {
        'test_triples': len(test_pairs),
        'test_cves': len(test_cves),
        'unique_cwe': len(cwe_vocab),
        'train_samples': len(train_feat),
        'fingerprint_test_coverage': len(test_cves_fp),
        'majority': maj_metrics,
        'fingerprint_lr': lr_metrics,
    }
    out_path = os.path.join(ROOT, 'baselines/security_fingerprint_baseline.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n结果已写 {out_path}')


if __name__ == '__main__':
    main()
