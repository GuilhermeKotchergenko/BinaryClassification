# BinaryClassification – Tackling Imbalanced Binary Datasets from Scratch

> **Machine Learning I (CC2008) – University of Porto**
> Academic Year 2024/25 · Guilherme Kotchergenko Batista & Yan Coelho

---

## 🚀 Project at a Glance

|                 | Details                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Goal**        | Build and evaluate a **binary classifier implemented 100 % in pure Python** (no `scikit‑learn`) and make it *imbalance‑aware*. |
| **Datasets**    | 50 public benchmark datasets with imbalanced classes                                                                   |
| **Metrics**     | Balanced Accuracy, Precision, Recall, F‑score, ROC‑AUC, G‑mean                                                              |
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
├── README.md                  #
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

| Model                | balanced_accuracy   | precision         | recall            | f1                | auc               | gmean             |
|:---------------------|:--------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| BCE                  | 0.771 ± 0.209       | 0.905 ± 0.099     | 0.792 ± 0.342     | 0.791 ± 0.300     | 0.912 ± 0.110     | 0.623 ± 0.394     |
| BCE New Sigmoid      | 0.826 ± 0.176       | **0.906 ± 0.065** | 0.861 ± 0.241     | **0.865 ± 0.179** | 0.935 ± 0.092     | 0.747 ± 0.308     |
| Focal                | 0.765 ± 0.207       | 0.902 ± 0.098     | 0.787 ± 0.341     | 0.787 ± 0.298     | 0.914 ± 0.111     | 0.616 ± 0.391     |
| Focal Dynamic Alpha  | 0.835 ± 0.161       | 0.631 ± 0.277     | 0.856 ± 0.215     | 0.674 ± 0.269     | 0.914 ± 0.113     | 0.828 ± 0.169     |
| Weighted             | 0.843 ± 0.146       | 0.594 ± 0.285     | 0.860 ± 0.209     | 0.646 ± 0.259     | 0.911 ± 0.116     | 0.838 ± 0.151     |
| Weighted_New_Sigmoid | **0.873 ± 0.146**   | 0.767 ± 0.276     | **0.940 ± 0.077** | 0.806 ± 0.268     | **0.939 ± 0.091** | **0.857 ± 0.176** |


Black bars in the slide deck illustrate the entries that achieves the highest (and statistically-significant) value (paired Wilcoxon, *p* < 0.05).


## 📜 Licence

This work is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---
