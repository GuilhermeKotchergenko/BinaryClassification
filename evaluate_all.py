import os
import pandas as pd
import numpy as np
import importlib
from glob import glob
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, balanced_accuracy_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import t, ttest_rel

sys.path.append(os.path.abspath("."))

from src.BCE_Logistic_regression import LogisticRegression as BCE_Logistic
from src.BCE_Logistic_Sigmoid import BCE_Logistic_Sigmoid
from src.Weighted_BCE_Logistic_regression import LogisticRegression as Weighted_Logistic
from src.Weighted_BCE_Logistic_regression_New_Sigmoid import LogisticRegression as WeightedNewSigmoid_Logistic
from src.Focal_Loss_Logistic_Regression import LogisticRegression as Focal_Logistic
from src.Focal_Loss_DynamicAlpha_Logistic_Regression import LogisticRegression as Focal_Loss_DynamicAlpha_Logistic

def robust_preprocessing(df):
    df = df.copy()
    df.replace({"t": 1, "f": 0, "M": 1, "F": 0}, inplace=True)
    df.replace("?", np.nan, inplace=True)
    for col in df.select_dtypes(include=["object", "bool"]).columns:
        df[col] = df[col].replace("nan", np.nan)
        df[col] = df[col].fillna("missing")
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    return df

def binarise_target(y):
    uniques = pd.Series(y).dropna().unique()
    if len(uniques) != 2:
        raise ValueError(f"Target não é binária: {uniques}")

    if set(uniques) == {0, 1}:
        return pd.Series(y)
    elif set(uniques) == {True, False}:
        return pd.Series(y).astype(int)
    elif set(uniques) == {-1, 1}:
        return (pd.Series(y) == 1).astype(int)
    else:
        mapping = {val: i for i, val in enumerate(sorted(uniques))}
        return pd.Series(y).map(mapping)


models = {
    "BCE": BCE_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=1000),
    "BCE New Sigmoid": BCE_Logistic_Sigmoid(lr=0.05, penalty='l2', max_iters=1000),
    "Focal": Focal_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=1000),
    "Focal Dynamic Alpha": Focal_Loss_DynamicAlpha_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=1000),
    "Weighted": Weighted_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=1000),
    "Weighted_New_Sigmoid": WeightedNewSigmoid_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=1000),
}

metric_names = [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auc", 
    "gmean",
]
metrics = {name: {m: [] for m in metric_names} for name in models}

data_dir = "data/class_imbalance/"
csv_files = glob(os.path.join(data_dir, "*.csv"))

print(f"Number of files: {len(csv_files)}\n")

for file in csv_files[:10]: # Running on 10 files for faster feedback first
    try:
        df = pd.read_csv(file)
        df = robust_preprocessing(df)

        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1]
        try:
            y = binarise_target(y_raw).values
        except ValueError as e:
            continue

        if np.isnan(X).any() or np.isnan(y).any():
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        imputer = SimpleImputer(strategy="mean")

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            bal_acc = balanced_accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            eps = 1e-12
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            den = tn + fp
            if den == 0:
                specificity = 0.0
            else:
                specificity = tn / den

            gmean = np.sqrt(rec * specificity)

            metrics[name]["balanced_accuracy"].append(bal_acc)
            metrics[name]["precision"].append(prec)
            metrics[name]["recall"].append(rec)
            metrics[name]["f1"].append(f1)
            metrics[name]["auc"].append(roc_auc)
            metrics[name]["gmean"].append(gmean)
            
    except Exception as e:
        print(f"Error processing {file}: {e}")

def bold_best(s):
    numbers = s.str.extract(r'([0-9.]+)').astype(float)[0]
    best = numbers.max()
    return s.where(numbers < best, "**" + s + "**")

rows = []
for name in models:
    row = {"Model": name}
    for metric in metric_names:
        vals = metrics[name][metric]
        if len(vals) > 0:
            mean, std = np.mean(vals), np.std(vals)
            row[metric] = f"{mean:.3f} ± {std:.3f}"
        else:
            row[metric] = "0.000 ± 0.000"
    rows.append(row)

df_results = pd.DataFrame(rows).set_index("Model")
df_bold = df_results.apply(bold_best)
print("\n=== Aggregated Results on first 10 datasets ===")
print(df_bold.to_markdown())
