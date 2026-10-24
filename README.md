# Zero-Edge Inductive Link Prediction: Re-Anchoring Cold Entities for Day-Zero Vulnerability Completion

Official code release for the paper *"Zero-Edge Inductive Link Prediction: Re-Anchoring Cold Entities for Day-Zero Vulnerability Completion"* (Zifan Yin, Jianhua Song, Chen Huang; submitted to ICDE 2027, Round 2).

> **Zero-edge collapse → ZEC protocol → re-anchoring recovery.** Subgraph-based link predictors score a query from its enclosing subgraph, so an entity that arrives with zero inference-time edges (a CVE on the day it is published) receives a deterministically empty subgraph and an identical score for every candidate tail. We formalize this collapse (Lemma 1), prove a ceiling over the graph-based family on cold-head queries (Proposition 1), introduce the **Z**ero-shot **E**ntity **C**ompletion (ZEC) protocol to make the regime measurable, and recover per-entity ranking by re-anchoring the query to the cold entity through a graph-external side channel (CPE product fingerprints on the security KG; entity-name character n-grams on standard KGs). Downstream, the recovered scores drive a backtrackable threat profile validated against CISA-KEV exploitation ground truth.

---

## Repository layout

```
rmpi/                 main method (RMPI-D, ProRMPI prototype prior, 2+SC production variant)
  train.py            training entry (variants via --ablation: 0=base, 1=NE/disclosing, 2=proto-hard)
  test_ranking_P.py   CWE-pool ranking (the paper's MRR/Hits protocol)
  test_auc_F.py       route-B pair discrimination (the paper's AUC protocol)
  make_zec_split.py   builds a ZEC split (cold-entity zero-edge cut) from any triple dataset
  side_channel_text.py  builds the text side channel (entity-name char n-grams) for standard KGs
  quantify_degeneration.py / inspect_cwe2id.py   paper-support tools (degeneration rates; cwe2id table)
  managers/ model/ subgraph_extraction/ utils/   method internals
security_kg/          security-KG data pipeline (NVD → KG → splits → fingerprints → threat profile)
  download.py nvd.py build_kg.py rebuild_fingerprint.py risk_profile.py aco*.py attack_centrality*.py ...
cluster_manager.py    shared clustering module (ProtoRMPI prototype prior)
baselines/
  CoMPILE/CoMPILE_github/   CoMPILE (Mai et al., AAAI 2021) + our security-split conversion
  grail/data/WN18RR/        WN18RR source data (for the ZEC split; benchmark from Dettmers et al.)
  transductive_kge_baseline.py   TransE/DistMult/ComplEx/RotatE baseline
  security_fingerprint_baseline.py  CPE-fingerprint popularity baseline
  kge_results.md            transductive-KGE numbers reproduced by this repository
```

The code is derived from [RMPI](https://github.com/zjukg/RMPI) (Geng et al., *RMPI: Reducing Memory and Path Information for inductive KG completion*, ICDE 2023). `CoMPILE/` retains its upstream license.

---

## Environment

- Python 3.9, CPU-only (no GPU needed — all paper runs were CPU-only; DGL has no Windows CUDA wheels).
- Windows / Linux / macOS (the code is plain PyTorch + DGL; development machine was Windows 11).

```bash
pip install -r requirements.txt
```

All training/evaluation commands below are run **from `rmpi/`** with the datasets under `../data/<dataset>/` (mirroring the layout produced by the data scripts).

---

## Data preparation

### Security KG (main results)

The security KG is built from public sources — NVD CVE/CPE records, MITRE CWE/CAPEC, MITRE ATT&CK, and CISA KEV. Total ~24 GB after processing (the fully-inductive split); the build is fully scripted:

```bash
# 1. Pull NVD CVE/CPE records (paged API pull; several hours) + static feeds (KEV, ATT&CK, CAPEC, CWE)
python ../security_kg/nvd.py          # → ../data/security_kg/raw/nvd_cve*.jsonl, nvd_cpe*.jsonl
python ../security_kg/download.py     # → ../data/security_kg/raw/{kev.json, enterprise-attack.json, capec_latest.xml, cwec_latest.xml.zip}

# 2. Build the KG and both splits (transductive random 80/10/10 + inductive temporal CVE-disjoint 70/10/20, seed 41504)
python ../security_kg/build_kg.py --out-name security_kg_full
#   → ../data/security_kg_full/           transductive split (1,951 test links)
#   → ../data/security_kg_full_ind/       fully-inductive split (42,207 unseen 2026 CVEs, 49,535 test links)

# 3. CPE product fingerprints (the graph-external side channel; auto-loaded when present)
python ../security_kg/rebuild_fingerprint.py   # → product_fingerprint.json in both split dirs
```

The fully-inductive split is produced by the temporal CVE split: all 2026 CVEs land in test with zero training edges (100% head-zero rate). `cvss_scores.json` and `kev_labels.json` are produced by the same build chain.

### WN18RR ZEC split (standard-KG results)

```bash
# from rmpi/ — the source WN18RR triple files ship with this repository
python make_zec_split.py --src ../baselines/grail/data/WN18RR --dst ../data/WN18RR_zec \
    --cold_frac 0.1 --valid_frac 0.1 --seed 42
#   → ../data/WN18RR_zec/  (8,157 test links, 100% head-zero by construction)

# text side channel (entity-name char 3-grams; the floor-crossing step on WN18RR)
python side_channel_text.py --data_dir ../data/WN18RR_zec --out ../data/WN18RR_zec/side_channel.npz
```

---

## Reproduction — the paper's tables

Protocol (fixed throughout the paper): **12-candidate CWE-pool ranking** (`test_ranking_P.py -m sample --num_samples 12 --runs 1 --seed 41504`), and **route-B pair discrimination** (`test_auc_F.py`, uniform head/tail corruption). The negative-sampler seed is wired inside the scripts, so any re-run under the stated protocol reproduces the reported numbers (not bit-exact across platforms, but protocol-exact).

### Table II — fully-inductive security split (42,207 unseen 2026 CVEs)

```bash
# train (one per variant; --ablation 0=base, 1=NE, 2=proto-hard; _sc = 2+SC production variant)
python train.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_base      --ablation 0 --num_epochs 10 --batch_size 32 --eval_every_iter 6000 --num_workers 0 --gpu -1
python train.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_NE        --ablation 1 --num_epochs 10 --batch_size 32 --eval_every_iter 6000 --num_workers 0 --gpu -1
python train.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard --ablation 2 --num_epochs 10 --batch_size 32 --eval_every_iter 6000 --num_workers 0 --gpu -1
python train.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard_sc --ablation 2 --num_epochs 10 --batch_size 32 --eval_every_iter 6000 --num_workers 0 --gpu -1

# rank (MRR / Hits@1/5/10)
python -u test_ranking_P.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_base         -m sample --runs 1 --seed 41504 --num_samples 12 --workers 1
python -u test_ranking_P.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_NE           -m sample --runs 1 --seed 41504 --num_samples 12 --workers 1
python -u test_ranking_P.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard   -m sample --runs 1 --seed 41504 --num_samples 12 --workers 1
python -u test_ranking_P.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard_sc -m sample --runs 1 --seed 41504 --num_samples 12 --workers 1

# route-B AUC
python -u test_auc_F.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_base           --ablation 0 --runs 1 --seed 41504 --num_workers 0 --gpu -1
python -u test_auc_F.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_NE             --ablation 1 --runs 1 --seed 41504 --num_workers 0 --gpu -1
python -u test_auc_F.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard     --ablation 2 --runs 1 --seed 41504 --num_workers 0 --gpu -1
python -u test_auc_F.py -d security_kg_full_ind -e security_kg_full_ind_RMPI_proto_hard_sc  --ablation 2 --runs 1 --seed 41504 --num_workers 0 --gpu -1
```

| Variant | MRR (12-cand.) | What it is |
|---|---|---|
| RMPI-base (enclosing) | 0.1068 | tie-break floor of Lemma 1 (random expectation $H_{12}/12\approx 0.259$) |
| NE (disclosing) | 0.3155 | disclosing promotion — the floor-crossing step |
| proto-hard | 0.1394 | prototype prior, degenerate cluster ties on this KG |
| 2+SC (production) | 0.2974 | proto-hard + CPE-fingerprint side channel + per-CWE tail features |

The 2+SC variant loads the side channel automatically: `product_fingerprint.json` (step 3 above) is detected in the dataset directory.

### Table III — inductive subset (3,679 links) / transductive (1,951 links)

Same commands with `-d security_kg_ind` (inductive subset) and `-d security_kg` (transductive split).

### Table IV — WN18RR ZEC split (8,157 test links)

```bash
python train.py -d WN18RR_zec -e WN18RR_zec_RMPI_base      --ablation 0 --num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1
python train.py -d WN18RR_zec -e WN18RR_zec_RMPI_NE        --ablation 1 --num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1
python train.py -d WN18RR_zec -e WN18RR_zec_RMPI_proto_hard --ablation 2 --num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1
python train.py -d WN18RR_zec -e WN18RR_zec_RMPI_NE_sc     --ablation 1 --num_epochs 20 --batch_size 32 --eval_every_iter 2000 --num_workers 0 --gpu -1   # with side_channel.npz present

python -u test_ranking_P.py -d WN18RR_zec -e WN18RR_zec_RMPI_base      -m sample --runs 1 --workers 4 --num_samples 12
python -u test_ranking_P.py -d WN18RR_zec -e WN18RR_zec_RMPI_NE        -m sample --runs 1 --workers 4 --num_samples 12
python -u test_ranking_P.py -d WN18RR_zec -e WN18RR_zec_RMPI_proto_hard -m sample --runs 1 --workers 4 --num_samples 12
python -u test_ranking_P.py -d WN18RR_zec -e WN18RR_zec_RMPI_NE_sc     -m sample --runs 1 --workers 4 --num_samples 12
```

| Variant | MRR (12-cand.) |
|---|---|
| RMPI-base (enclosing) | 0.0834 — exactly $1/12$, the deterministic tie-break floor |
| NE (disclosing) | 0.1636 |
| proto-hard | 0.2414 |
| NE + text side channel | 0.3563 — the floor-crossing step |

### External baselines

```bash
# CoMPILE (below random on the fully-inductive split — AUC 0.179 ± 0.0086, 5 runs)
cd ../baselines/CoMPILE/CoMPILE_github
python -u test_auc.py -d security_kg_full_ind -e security_kg_full_ind_CoMPILE --runs 5 --hop 2 --disable_cuda

# Transductive KGEs (category error, not a tuning failure — MRR 0.0002–0.1377 on unseen heads)
python ../baselines/transductive_kge_baseline.py -d security_kg_ind     # TransE/DistMult/ComplEx/RotatE; full numbers in ../baselines/kge_results.md

# CPE-fingerprint popularity baseline (the popularity-prior head representation)
python ../baselines/security_fingerprint_baseline.py
```

GraIL's enclosing branch is replicated by RMPI-base (`--ablation 0`), so the GraIL rows are produced by the RMPI-base commands above. NBFNet is cited from its published results — its CUDA extension does not build in a CPU-only environment and its full-graph boundary tensor exceeds our memory budget.

### Stage 3 — threat profile (CISA-KEV)

```bash
python ../security_kg/risk_profile.py            # threat profile + oracle ceiling (ROC-AUC 0.886)
python ../security_kg/probe_cands_confound.py    # completion-predicted term dissects to a coverage marker (0.4783 covered-only)
python ../security_kg/verify_product_artifact.py # product-surface artifact re-derivation from raw NVD
```

---

## Protocol and reproducibility notes

- **Candidate pool**: 12 candidates per query — 11 sampled negatives + the true tail, per-CWE-class (head side) and per-CVE (tail side); seed 41504, single run.
- **Floors are sample-count dependent**: the tie-break floor reads 0.1068 at 12 candidates and 0.0496–0.0498 at a larger count; all comparisons in the paper are within the 12-candidate protocol.
- **The negative-sampler seed is wired** (`test_ranking_P.py --seed`, default 41504): re-running the shipped scripts is itself reproducible. Re-runs reproduce the reported numbers under the fixed protocol; the paper does not claim bit-exactness across platforms.
- The security-KG splits are disclosed as semi-inductive on the tail side (the CWE taxonomy is closed — 787 classes at training time; 23 test-only classes fall to a shared OOV slot, affecting 34 of 49,535 test links).
- The CPE-fingerprint side channel exists only where NVD carries product records: 48.89% of unseen CVEs carry none and keep the popularity-prior fallback.

## Attribution and license

Built on [RMPI](https://github.com/zjukg/RMPI) (Geng et al., ICDE 2023) — the RMPI backbone (enclosing/disclosing subgraph extraction, relational message passing) is theirs; the ZEC protocol, re-anchoring recovery, side channels, prototype extension, and the security-KG pipeline are ours. If you use this code, please cite both this paper and RMPI. Released under the MIT license (see `LICENSE`); `baselines/CoMPILE/` retains its upstream license.

## Data availability

The security KG (24 GB processed) is regenerable end-to-end from public NVD/MITRE/CISA sources with the scripts above (see `security_kg/`). A frozen archive of `security_kg_full_ind` (the main-table split, 2026-CVE test cut, seed 41504) is available from the authors on request and will be deposited to a public archive (Zenodo) with a DOI at publication time.
