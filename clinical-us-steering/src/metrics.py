"""Classification metrics, bootstrap CIs, representation similarity."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)


def attribute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = y_true >= 0
    y_true, y_pred = y_true[keep], y_pred[keep]
    if len(y_true) == 0:
        return {"macro_f1": float("nan"), "balanced_acc": float("nan"), "n": 0}
    labels = np.unique(y_true)  # classes absent from y_true would add spurious zeros
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                       zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)), "n": int(len(y_true))}


def youden_threshold(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return 0.5
    best, thr = -1.0, 0.5
    for t in np.unique(p):
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        if sens + spec - 1 > best:
            best, thr = sens + spec - 1, float(t)
    return thr


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "sensitivity": sens, "specificity": spec, "balanced_acc": (sens + spec) / 2,
            "f1": float(f1_score(y, pred, zero_division=0)), "n": int(len(y))}


def bootstrap(metric_fn, *arrays, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of metric_fn(*arrays) over resampled sample indices."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    m, vals = len(arrays[0]), []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals.append(metric_fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_bootstrap_diff(metric_fn, y, pred_a, pred_b, n: int = 2000, seed: int = 0) -> dict:
    """metric(b) - metric(a) with a paired-resampling 95% CI and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    obs = metric_fn(y, pred_b) - metric_fn(y, pred_a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            diffs.append(metric_fn(y[idx], pred_b[idx]) - metric_fn(y[idx], pred_a[idx]))
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_value": float(min(p, 1.0))}


def macro_f1(y, pred) -> float:
    return attribute_metrics(y, pred)["macro_f1"]


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
    return (a * b).sum(1)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(x.T @ y, "fro") ** 2
    return float(hsic / (np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")))
