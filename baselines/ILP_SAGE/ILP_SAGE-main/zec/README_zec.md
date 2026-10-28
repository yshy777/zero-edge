# ILP-SAGE ZEC adaptation — reproducing the paper's ILP-SAGE rows

Upstream: **ILP-SAGE** (Rahimi et al., *Inductive Link Prediction with Sequential Attentive Graph Embeddings*, Neurocomputing 2026) — https://github.com/mohammadRahimi1993/ILP_SAGE.git.
This directory ships the upstream `src/` (model/training, unmodified) plus our `zec/` adaptation that trains and scores ILP-SAGE under the paper's ZEC protocol. No checkpoints ship in-repo (see the main README's data availability).

## Environment (the one used for the paper's rows)

- Python 3.9.13, **CPU-only** (DGL has no Windows CUDA wheels; the paper's footnote discloses the CPU adaptation)
- `torch==2.3.0+cpu`, `dgl==2.2.1`, `numpy==1.26.4`
- Upstream `requirements.txt` targets torch 2.2 + DGL wheels from data.dgl.ai; the paper runs used the CPU equivalents above. Creating the environment used for the paper:

```bash
python -m venv ilpsage_venv
ilpsage_venv/Scripts/pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
ilpsage_venv/Scripts/pip install dgl==2.2.1 numpy==1.26.4
```

Run everything from `baselines/ILP_SAGE/ILP_SAGE-main/` (the script's `from src.model import GraphSAGE` and `from zec.ilpsage_zec_dataset import ...` assume this cwd).

## Protocol and disclosure

- **Same candidate blocks as the main table.** The rank step reads `security_kg_full_ind_RMPI_base_ranking_{head,tail}.txt` from the split directory — the candidate blocks stored by the RMPI-base run of `test_ranking_P.py` (positive first, `block_size` rows per block). ILP-SAGE therefore scores exactly the same negatives as every main-table row (bit-identical candidates, same seed 41504).
- **Direction naming (matches the files, not the paper's query vocabulary).** `ranking_head.txt` = corrupt-**head** blocks, candidates drawn from the CVE pool = the paper's tail-cold queries; `ranking_tail.txt` = corrupt-**tail** blocks, CWE-pool candidates = the paper's head-cold queries.
- **Ties are counted honestly.** `rank = 1 + #{candidates scoring >= the true tail}`, the same tie convention as `test_ranking_P.py` (ties included, not broken in the positive's favour — an argsort-based reading inflates collapsed blocks to rank 1, the 0.9978 artifact `diag_rank_collapse.py` documents).
- **Random node features, saved with the checkpoint.** The upstream model has no entity features on this split; the adaptation feeds seeded random 64-dim features (`torch.randn`, seed fixed) and saves them inside the checkpoint so `rank` reuses the exact same features as `train` (otherwise rank embeddings and the predictor disagree and every score collapses to a constant).

## Commands

```bash
# train (100 epochs, official hyperparameters, CPU)
python -u zec/ilpsage_zec.py train --data-dir ../data/security_kg_full_ind
#   → zec/ilpsage_zec_model.pt

# rank under the 12-candidate main-table protocol
python -u zec/ilpsage_zec.py rank --data-dir ../data/security_kg_full_ind --ckpt zec/ilpsage_zec_model.pt --block-size 12

# rank under the 50-candidate robustness protocol
python -u zec/ilpsage_zec.py rank --data-dir ../data/security_kg_full_ind --ckpt zec/ilpsage_zec_model.pt --block-size 50
```

## Expected rows (the paper's numbers, single run, seed 41504)

| protocol | MRR | head-cold (corrupt-tail) | tail-cold (corrupt-head) |
|---|---|---|---|
| 12 candidates | **0.0936** | 0.0838 (deterministic tie-break floor, Hits@1 0.0001) | 0.1034 (head-independent CWE prior) |
| 50 candidates | **0.0226** (full-split 0.0225) | ≈ floor 1/50 | — |

The dev-machine chain that ran these rows is `rmpi/queue_stage2_ilpsage.sh` (verbatim; it hardcodes the `D:/ilpsage_venv` path — adjust before re-running).
