import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("."))
import src.BCE_Logistic_regression
import src.BCE_Logistic_Sigmoid
import src.Weighted_BCE_Logistic_regression
from src.BCE_Logistic_regression import LogisticRegression as BCE_Logistic
from src.BCE_Logistic_Sigmoid import BCE_Logistic_Sigmoid
from src.Weighted_BCE_Logistic_regression import LogisticRegression as Weighted_Logistic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/class_imbalance/dataset_100_spambase.csv")
df.replace({"t": 1, "f": 0, "M": 1, "F": 0}, inplace=True)
df.replace("?", np.nan, inplace=True)

for col in df.select_dtypes(include=["object", "bool"]).columns:
    df[col] = df[col].replace("nan", np.nan)
    df[col] = df[col].fillna("missing")
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df.dropna(axis=1, how='all', inplace=True)
df.dropna(inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
imputer = SimpleImputer(strategy="mean")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "BCE": BCE_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=200),
    "BCE New Sigmoid": BCE_Logistic_Sigmoid(lr=0.05, penalty='l2', max_iters=200),
    "Weighted": Weighted_Logistic(lr=0.05, penalty='l2', tolerance=1e-6, max_iters=200)
}

for name, model in models.items():
    print(f"Treinando {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print(f"{name} - F1: {f1:.4f}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

