# BinaryClassification – Tackling Imbalanced Binary Datasets from Scratch

> **Machine Learning I (CC2008) – University of Porto**
> Academic Year 2024/25 · Guilherme Kotchergenko Batista & Yan Coelho

---

## 🚀 Project at a Glance

|                 | Details                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Goal**        | Build and evaluate a **binary classifier implemented 100 % in pure Python** (no `scikit‑learn`) and make it *imbalance‑aware*. |
| **Datasets**    | > 51 public benchmark datasets with class skew (UCI, KEEL).                                                                   |
| **Metrics**     | F‑score, ROC‑AUC, G‑mean & PR‑AUC under repeated stratified CV.                                                                |
| **Outcome**     | Outcome – Weighted New Sigmoid improves minority-class recall by +23 pp (0.69 → 0.85) and balanced-accuracy by +9 pp (0.71 → 0.80) while keeping F1 unchanged. Gains are significant (paired t, p < 0.001 for balanced-acc).                                                   |
| **Inspiration** | [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms) for Logistic Regression logic.            |

---

## 📂 Repository Layout

```
BinaryClassification/
├── data/                      # Raw datasets
│   └── class_imbalance/
├── notebooks/                 # Jupyter notebooks – exploration & experiment tracking
│   ├── 01_test_models.ipynb   # Quick tests
│   └── 02_global_execution.ipynb # Run every dataset → aggregate global metrics
├── src/                       # Pure-Python logistic-regression variants (baseline & imbalance-aware)
│   ├── BCE_Logistic_Regression.py
│   ├── BCE_NewSigmoid_Logistic_Regression.py
│   ├── Focal_Loss_Logistic_Regression.py
│   ├── Focal_Loss_DynamicAlpha_Logistic_Regression.py
│   ├── Weighted_BCE_Logistic_Regression.py
│   └── Weighted_BCE_NewSigmoid_Logistic_Regression.py
├── requirements.txt           # Reproducible Python env (tested on ≥ 3.10)
├── README.md                  # ← you are here
└── PracticalAssignment_ML1.pdf# Assignment brief (PDF)

```
---

## 🏁 Quick Start

```bash
# 1) Clone
$ git clone https://github.com/GuilhermeKotchergenko/BinaryClassification.git
$ cd BinaryClassification

# 2) Set up an isolated environment (recommended)
$ python3 -m venv .venv && source .venv/bin/activate  # Linux/macOS

# 3) Install requirements
$ pip install -r requirements.txt

# 4) Explore the experiments
$ jupyter lab notebooks/02_global_execution.ipynb
```

## 📈 Key Results

| Model                    |      Accuracy | **Balanced Acc.** |     Precision | **Recall (minority)** |            F1 |       ROC-AUC |
| ------------------------ | ------------: | ----------------: | ------------: | --------------------: | ------------: | ------------: |
| **BCE (baseline)**       | 0.929 ± 0.056 | 0.709 ± 0.183 | 0.734 ± 0.307 |     0.688 ± 0.375 |0.688 ± 0.340| 0.874 ± 0.139 |
| BCE New Sigmoid          | 0.926 ± 0.070 |     0.744 ± 0.196 | 0.702 ± 0.348 |         0.730 ± 0.367 | 0.703 ± 0.356 | 0.868 ± 0.152 |
| Focal                    | 0.918 ± 0.067 |     0.724 ± 0.191 | 0.691 ± 0.300 |         0.749 ± 0.350 | 0.701 ± 0.318 | 0.874 ± 0.140 |
| Focal Dynamic Alpha      | 0.641 ± 0.284 |     0.712 ± 0.149 | 0.453 ± 0.355 |     0.969 ± 0.074     | 0.538 ± 0.324 | 0.876 ± 0.137 |
| Weighted                 | 0.701 ± 0.221 |     0.764 ± 0.142 | 0.490 ± 0.350 |         0.921 ± 0.125 | 0.560 ± 0.298 | 0.872 ± 0.140 |
| **Weighted New Sigmoid** | 0.843 ± 0.178 | **0.798 ± 0.174** | 0.654 ± 0.336 |     **0.846 ± 0.213** | **0.696 ± 0.300** | 0.866 ± 0.153 |


Black bars in the slide deck illustrate the entry that achieves the highest (and statistically-significant) value (paired Wilcoxon, *p* < 0.05).

---

## 🛠️ Inside the Classifier

* **Base algorithm** – *(e.g.)* Logistic Regression solved with **batch gradient descent**.
* **Adjustment** – Class‑dependent **cost term** added to the loss; automatic cost scaling driven by the inverse class frequency.
* **From‑scratch code** – No `numpy.linalg` black‑boxes 🚫; every derivative is spelled out.
* **Speed** – Numpy vectorisation + optional Cython (see `setup.cfg`).

For a full derivation see `notebooks/02_custom_adjustment.ipynb` (§ “Deriving the cost‑sensitive loss”).

---

## 📜 Licence

This work is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---
