# 传导式 KGE Baseline 结果（TransE/DistMult/ComplEx/RotatE）

数据集：`security_kg_ind`（train 13969 / valid 1851 / test 3679，test 3679/3679 为 unseen head CVE，完全归纳）
设置：dim=100，epochs=100，batch=1024，lr=0.01，margin=1.0，seed=41504，CPU 单线程。
评估：raw（未过滤）tail 排序，MRR + Hits@k。

| 模型 | seen_MRR | valid_MRR | test_unseen_MRR | Hits@10 (unseen) |
|---|---|---|---|---|
| TransE | 0.4529 | 0.1914 | **0.1377** | 0.2566 |
| DistMult | 0.9393 | 0.0198 | 0.0206 | 0.0427 |
| ComplEx | 0.9381 | 0.0148 | 0.0139 | 0.0285 |
| RotatE | 0.9173 | 0.0002 | 0.0002 | 0.0000 |

（随机 tail 排序的 MRR 期望 ≈ ln(n_e)/n_e ≈ 0.0006，n_e=16760。）

## 解读（论文核心结构论证 §8.9）

1. **传导式模型「学会了」但「传不了」**：seen_MRR 0.45–0.94（在见过 head 的三元组上排序很好）→ unseen 上 MRR 崩到 0.0002–0.14（≈随机）。传导式模型靠实体嵌入，对 unseen head（OOV，嵌入随机且冻结）完全无法表征，尾巴排序退化。

2. **TransE 例外有 head-independent 的尾巴先验**：`score = -‖e(h)+e(r)-e(t)‖` 分解出与 head 无关的 `‖e(r)-e(t)‖` 项，使其在 unseen 上仍能按「哪些 CWE 离 has_weakness 关系向量最近」排序，故 0.138 高于其余（DistMult/ComplEx/RotatE 的 bilinear 分数无法这样分解，直接随机）。这是传导式模型的**上限**，仍远低于归纳式。

3. **对比 ProRMPI（归纳式）**：同 unseen CVE 上 test AUC 0.8469（full_ind）/ 0.8304（15k cwe-fix retrain）。

⚠️ **指标口径待对齐（item 2）**：KGE 报 MRR/Hits（排名），ProRMPI 目前报 AUC（分类）。要图里做 head-to-head 柱状图，需要补 ProRMPI 的 MRR/Hits@10（= 补齐清单 item 2「排名 MRR/Hits@10」）。定性差距已成立（传导 MRR≈0.02 vs 归纳 AUC≈0.83），补上排名指标即可完全对齐。
