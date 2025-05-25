# BinaryClassification – Tackling Imbalanced Binary Datasets from Scratch

> **Machine Learning I (CC2008) – University of Porto**
> Academic Year 2024/25 · Guilherme Kotchergenko Batista & Yan Coelho

---

## 🚀 Project at a Glance

|                 | Details                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Goal**        | Build and evaluate a **binary classifier implemented 100 % in pure Python** (no `scikit‑learn`) and make it *imbalance‑aware*. |
| **Datasets**    | 50 public benchmark datasets with imbalanced classes                                                                   |
| **Metrics**     | Accuracy, Balanced Accuracy, Precision, Recall, F‑score, ROC‑AUC, G‑mean                                                              |
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

Values: Mean ± sd

| Model                | accuracy          | balanced_accuracy   | precision         | recall            | f1                | auc               | gmean             |
|:---------------------|:------------------|:--------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| BCE                  | 0.929 ± 0.056 | 0.709 ± 0.183           | 0.734 ± 0.307     | 0.688 ± 0.375     | 0.688 ± 0.341     | 0.874 ± 0.139     | 0.531 ± 0.376     |
| BCE New Sigmoid      | 0.925 ± 0.070     | 0.742 ± 0.199       | 0.702 ± 0.351     | 0.725 ± 0.373     | 0.700 ± 0.361     | 0.861 ± 0.163     | 0.579 ± 0.399     |
| Focal                | 0.918 ± 0.067     | 0.724 ± 0.191       | 0.691 ± 0.300     | 0.749 ± 0.350     | 0.701 ± 0.317     | 0.874 ± 0.140     | 0.561 ± 0.380     |
| Focal Dynamic Alpha  | 0.641 ± 0.284     | 0.712 ± 0.149       | 0.453 ± 0.355     | **0.969 ± 0.074** | 0.538 ± 0.324     | 0.876 ± 0.137     | 0.584 ± 0.313     |
| Weighted             | 0.701 ± 0.221     | 0.764 ± 0.142       | 0.490 ± 0.350     | 0.921 ± 0.125     | 0.560 ± 0.298     | 0.872 ± 0.140     | 0.704 ± 0.240     |
| Weighted_New_Sigmoid | 0.837 ± 0.178     | **0.796 ± 0.175**   | 0.651 ± 0.341     | 0.837 ± 0.214     | 0.690 ± 0.302     | 0.865 ± 0.152     | **0.760 ± 0.234** |


Black bars in the slide deck illustrate the entries that achieves the highest (and statistically-significant) value (paired Wilcoxon, *p* < 0.05).


## 📜 Licence

This work is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---
