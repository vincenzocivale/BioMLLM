"""Diagnosis metrics with bootstrap CIs and paired comparisons against the base embedding.

    python scripts/evaluate_diagnosis.py
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import _path  # noqa: F401
from metrics import binary_metrics, bootstrap, paired_bootstrap_diff
from utils import ROOT, load_yaml, save_json


def fold_thresholded(pred, col, thr_col):
    return (pred[col] >= pred[thr_col]).astype(int)


def summarise(pred, prob, thr, n_boot):
    y = pred.malignancy_gt.to_numpy()
    p = pred[prob].to_numpy()
    # per-fold val-selected thresholds -> pass the binarised prediction through a 0.5 cut
    hard = fold_thresholded(pred, prob, thr).to_numpy()
    m = binary_metrics(y, p, 0.5)
    hm = binary_metrics(y, hard, 0.5)
    for k in ("sensitivity", "specificity", "balanced_acc", "f1"):
        m[k] = hm[k]
    m["auroc_ci95"] = bootstrap(roc_auc_score, y, p, n=n_boot)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    cfg = load_yaml("configs/train_diagnosis.yaml")
    sfx = "" if args.seed == 17 else f"_seed{args.seed}"
    res, preds = {}, {}
    for run in cfg["runs"]:
        f = ROOT / cfg["output_root"] / f"{run}{sfx}" / "predictions.csv"
        if not f.exists():
            continue
        pred = pd.read_csv(f).sort_values("sample_id").reset_index(drop=True)
        preds[run] = pred
        res[run] = {"mlp": summarise(pred, "malignancy_probability", "threshold", cfg["bootstrap"]),
                    "logreg": summarise(pred, "logreg_probability", "logreg_threshold", cfg["bootstrap"]),
                    "feature_dim": int(pred.feature_dim.iloc[0])}
        save_json(res[run], ROOT / cfg["output_root"] / f"{run}{sfx}" / "metrics.json")
    comps = {}
    base = "diagnosis_base_usfmae"
    for run in preds:
        if run == base or base not in preds:
            continue
        a, b = preds[base], preds[run]
        assert (a.sample_id == b.sample_id).all()
        for col in ("malignancy_probability", "logreg_probability"):
            comps[f"{run} - {base} [{col}]"] = paired_bootstrap_diff(
                roc_auc_score, a.malignancy_gt.to_numpy(), a[col].to_numpy(), b[col].to_numpy(),
                n=cfg["bootstrap"])
    summary = {"results": res, "auroc_differences_vs_base": comps}
    save_json(summary, ROOT / cfg["output_root"] / f"diagnosis_comparison{sfx}.json")
    rows = [{"run": r, "classifier": c, **{k: v for k, v in res[r][c].items() if k != "auroc_ci95"},
             "auroc_ci_low": res[r][c]["auroc_ci95"][0], "auroc_ci_high": res[r][c]["auroc_ci95"][1],
             "feature_dim": res[r]["feature_dim"]} for r in res for c in ("mlp", "logreg")]
    pd.DataFrame(rows).to_csv(ROOT / cfg["output_root"] / f"diagnosis_comparison{sfx}.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    print(json.dumps(comps, indent=2))


if __name__ == "__main__":
    main()
