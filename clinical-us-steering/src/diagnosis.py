"""Benign/malignant classifiers on frozen features: 2-layer MLP and logistic regression."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DiagnosisMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(x_tr, y_tr, x_val, y_val, cfg: dict, seed: int):
    """Standardise on train, BCE with positive re-weighting, early stopping on val AUROC."""
    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa: E731
    xt, xv = f(x_tr), f(x_val)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    model = DiagnosisMLP(x_tr.shape[1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pos_w = ((len(yt) - yt.sum()) / yt.sum().clamp(min=1)).detach()
    best, best_ep, best_state, hist = -1.0, 0, None, []
    for ep in range(cfg["epochs"]):
        model.train()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xt), yt, pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).numpy()
        auc = roc_auc_score(y_val, pv)
        hist.append({"epoch": ep, "loss": loss.item(), "val_auroc": auc})
        if auc > best:
            best, best_ep, best_state = auc, ep, copy.deepcopy(model.state_dict())
        if ep - best_ep >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()

    def predict(a):
        with torch.no_grad():
            return torch.sigmoid(model(f(a))).numpy()

    return predict, {"state": best_state, "mu": mu, "sd": sd, "best_epoch": best_ep,
                     "best_val_auroc": best}, hist


def fit_logreg(x_tr, y_tr):
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-3, 1e-2, 1e-1, 1.0, 10.0], cv=3, class_weight="balanced", max_iter=5000,
        scoring="roc_auc"))
    clf.fit(x_tr, y_tr)
    return lambda a: clf.predict_proba(a)[:, 1]


def one_hot_attributes(df, attributes: dict) -> np.ndarray:
    """Oracle features: one-hot of each GT descriptor (all-zero block when missing)."""
    cols = []
    for a, classes in attributes.items():
        if not df[a].notna().any():
            continue
        cols.append(np.stack([(df[a] == c).to_numpy(float) for c in classes], 1))
    return np.concatenate(cols, 1)
