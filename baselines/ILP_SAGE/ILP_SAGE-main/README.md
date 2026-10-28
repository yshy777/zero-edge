# 📚 ILP-SAGE: A Sampling-Based Framework for Scalable Inductive Link Prediction

> **Published in Neurocomputing (2026)**

## Abstract

<p align="justify">
Link prediction is a fundamental technique for inferring missing facts in knowledge graphs.
Under an inductive setting, it aims to predict relationships involving nodes not present during
training, making it especially valuable for dynamic and large-scale applications. While graph
neural network-based methods represent a common approach, they often suffer from two key
shortcomings: over smoothing and high computational cost with increased neighborhood
depth, limiting their scalability, and a lack of rigorous evaluation on non-English datasets,
leaving their cross-lingual generalizability in question. To address these challenges, we propose
ILP-SAGE, a novel framework for inductive link prediction. ILP-SAGE employs a columnar
data storage scheme to enable fast subgraph sampling and efficient processing by graph
networks. A controlled random sampling mechanism extracts relevant subgraphs, which are
then encoded into node embeddings using a GraphSAGE-based model. This design facilitates
controlled graph expansion and produces embeddings that effectively combine structural and
feature information, balancing accuracy with computational efficiency. We conduct a
comprehensive evaluation of ILP-SAGE using a suite of 27 benchmarks (13 transductive and
14 inductive) derived from prominent English-language datasets WN18RR, FB15K-237, and
NELL-995, and the Persian-language dataset PersianILP. Results demonstrate that ILP-SAGE
achieves competitive predictive performance while significantly reducing runtime. For
instance, on version 4 of the NELL-995 dataset, the proposed framework achieves a 3.52-point
improvement in AUC-PR over recent state-of-the-art methods and reduces the training time
from 1705.5 minutes to 2.28 minutes compared to GraIL. These results confirm the suitability
of the proposed framework for accurate and scalable link prediction in large, heterogeneous,
and multilingual knowledge graphs.
</p>

---

<p>
<strong> You can watch the presentation video here: </strong>
<a href="https://www.aparat.com/v/vss779n" target="_blank">ILP-SAGE Presentation on Aparat</a>
</p>

# 🚀 Installation

## Prerequisites

- Python 3.12
- Git
- Jupyter Notebook (optional, for interactive experiments)

---

## 1. Clone the Repository

```bash
git clone https://github.com/mohammadRahimi1993/ILP_SAGE.git
cd ILP_SAGE
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
pip install \
    numpy==1.26.4 \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    torcheval \
    dgl -f https://data.dgl.ai/wheels/torch-2.2/repo.html \
    torchmetrics==1.2.1 \
    transformers==4.38.0 \
    scikit-learn==1.6.1
```

---

## 5. (Optional) Register a Jupyter Kernel

```bash
pip install ipykernel

python -m ipykernel install \
    --user \
    --name ilp_sage \
    --display-name "Python (ILP-SAGE)"
```

---

# ▶️ Running Experiments

The repository provides implementations of both **ILP-SAGE** and the **GraphSAGE-based baseline**.

Run the following scripts according to your experiment:

```bash
python main_analysis.py
```

```bash
python run_experiments.py
```

```bash
python GraphSAGE.py
```

---

# 📦 Dependencies

- PyTorch 2.2
- DGL
- NumPy
- Transformers
- TorchMetrics
- TorchEval
- Scikit-learn

---

# 📄 Citation

If you use this repository in your research, please cite the accompanying Neurocomputing paper.
## Citation

If you use this repository, please cite:
```bibtex
@article    {rahimi2026ilpsage,
  title   = {ILP-SAGE: A Sampling-Based Framework for Scalable Inductive Link Prediction},
  author  = {Rahimi, Mohammad and Fatemi, Afsaneh and Baraani, Ahmad},
  journal = {Neurocomputing},
  volume  = {699},
  pages   = {134429},
  year    = {2026},
  month   = jul
}

