"""STEP 9-10: malignancy classifiers on frozen embeddings (per CV fold, same folds as steering).

    python scripts/train_diagnosis.py --run diagnosis_clinical_steering
"""
import argparse

import numpy as np
import pandas as pd
import torch

import _path  # noqa: F401
from common import load_configs, load_manifest
from diagnosis import fit_logreg, fit_mlp, one_hot_attributes
from metrics import youden_threshold
from utils import ROOT, load_yaml, run_dir, save_json, save_yaml


def features(run_cfg, df, data_cfg, root, fold, seed):
    if run_cfg["features"] == ["oracle"]:
        return one_hot_attributes(df, data_cfg["attributes"])
    src = run_cfg["source_run"] + ("" if seed == 17 else f"_seed{seed}")
    emb = np.load(ROOT / root / src / f"fold{fold}" / "embeddings.npz")
    assert (emb["sample_id"] == df.sample_id.to_numpy()).all()
    return np.concatenate([emb["z_none" if f == "base" else f"z_{f}"] for f in run_cfg["features"]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    data_cfg, _, _ = load_configs()
    cfg = load_yaml("configs/train_diagnosis.yaml")
    run_cfg = cfg["runs"][args.run]
    name = args.run if args.seed == 17 else f"{args.run}_seed{args.seed}"
    out = run_dir(cfg["output_root"], name)
    save_yaml({"run": args.run, "seed": args.seed, "run_cfg": run_cfg, "train": cfg}, out / "config.yaml")
    df = load_manifest(data_cfg)
    n_folds = data_cfg["split"]["n_folds"]
    lab = df.malignancy.notna().to_numpy()
    y = df.malignancy.fillna(-1).astype(int).to_numpy()
    rows, hist = [], []
    for k in range(n_folds):
        x = features(run_cfg, df, data_cfg, cfg["output_root"], k, args.seed)
        part = np.where(df.fold == k, "test", np.where(df.fold == (k + 1) % n_folds, "val", "train"))
        tr, va, te = [(part == s) & lab for s in ("train", "val", "test")]
        mlp, state, h = fit_mlp(x[tr], y[tr], x[va], y[va], cfg, args.seed + k)
        (out / f"fold{k}").mkdir(exist_ok=True)
        torch.save(state, out / f"fold{k}" / "best.pt")
        hist += [{"fold": k, **r} for r in h]
        thr = youden_threshold(y[va], mlp(x[va]))
        lr = fit_logreg(x[tr | va], y[tr | va])  # LR tunes C by inner CV: no separate val needed
        lr_thr = youden_threshold(y[tr | va], lr(x[tr | va]))
        p_mlp, p_lr = mlp(x[te]), lr(x[te])
        for j, i in enumerate(np.where(te)[0]):
            rows.append({"sample_id": df.sample_id[i], "patient_id": df.patient_id[i],
                         "malignancy_gt": int(y[i]), "malignancy_probability": float(p_mlp[j]),
                         "threshold": thr, "logreg_probability": float(p_lr[j]),
                         "logreg_threshold": lr_thr, "birads": df.birads[i],
                         "split": f"test_fold{k}", "fold": k, "feature_dim": x.shape[1]})
        print(f"[{name}] fold {k}: dim {x.shape[1]} val AUROC {state['best_val_auroc']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    save_json({"run": name, "n": len(rows)}, out / "train_summary.json")


if __name__ == "__main__":
    main()
