import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def kfold_cross_validate(model_cls, X, y, k=5, shuffle=True, random_state=42, **model_kwargs):
    """Perform stratified k-fold cross validation for a custom model.

    Parameters
    ----------
    model_cls : class
        Class of the model to instantiate for each fold.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Binary target vector.
    k : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting the data.
    random_state : int, default=42
        Random state for reproducibility when ``shuffle`` is True.
    **model_kwargs : dict
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    dict
        Dictionary mapping metric names to lists of scores across folds.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
        "gmean": [],
    }

    for train_idx, test_idx in skf.split(X, y):
        model = model_cls(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])

        y_pred = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])

        acc = accuracy_score(y[test_idx], y_pred)
        bal = balanced_accuracy_score(y[test_idx], y_pred)
        prec = precision_score(y[test_idx], y_pred, zero_division=0)
        rec = recall_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred)
        auc = roc_auc_score(y[test_idx], y_proba)

        tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
        specificity = tn / (tn + fp + 1e-12)
        gmean = np.sqrt(rec * specificity)

        metrics["accuracy"].append(acc)
        metrics["balanced_accuracy"].append(bal)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)
        metrics["gmean"].append(gmean)

    return metrics


def summarise_metrics(metric_dict):
    """Return mean ± std as formatted strings for metric dictionary."""
    summary = {}
    for name, values in metric_dict.items():
        values = np.asarray(values)
        summary[name] = f"{values.mean():.3f} ± {values.std():.3f}"
    return summary

