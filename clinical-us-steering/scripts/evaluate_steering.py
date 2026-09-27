"""STEP 7: attribute metrics, query controls, steering selectivity matrix, representation
similarity and gate analysis for a trained run (all CV folds, out-of-fold aggregation).

    python scripts/evaluate_steering.py --run clinical_steering_usfmae
"""
import argparse
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _path  # noqa: F401,E402
from common import (build_model, device, fold_datasets, label_index, load_configs,  # noqa: E402
                    load_manifest)
from data import partition  # noqa: E402
from metrics import attribute_metrics, bootstrap, cosine_rows, linear_cka, macro_f1  # noqa: E402
from utils import ROOT, count_params, save_json  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@torch.no_grad()
def extract(model, ds, texts: dict, dev, amp):
    """-> {name: (N, D) embedding}, {name: {attr: (N, C) probs}}, {name: {layer: rel_update}}"""
    model.eval()
    loader = torch.utils.data.DataLoader(ds, 32)
    Z = {n: [] for n in texts}
    P = {n: {a: [] for a in model.heads.attributes} for n in texts}
    upd = {n: {} for n in texts}
    for b in loader:
        records = {}
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            z = model.encode(b["image"].to(dev), list(texts.values()), records)
        for n, t in texts.items():
            zt = z[t].float()
            Z[n].append(zt.cpu())
            for a in model.heads.attributes:
                P[n][a].append(model.heads(a, zt).softmax(-1).float().cpu())
            for layer, rec in records.get(t, {}).items() if t is not None else []:
                upd[n].setdefault(layer, []).append(rec["rel_update"])
    Z = {n: torch.cat(v).numpy() for n, v in Z.items()}
    P = {n: {a: torch.cat(v).numpy() for a, v in d.items()} for n, d in P.items()}
    upd = {n: {l: float(np.mean(v)) for l, v in d.items()} for n, d in upd.items()}
    return Z, P, upd


def probe_predict(z_tr, y_tr, z_te):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = y_tr >= 0
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-3, 1e-2, 1e-1, 1.0], cv=3, class_weight="balanced", max_iter=5000,
        scoring="f1_macro"))
    clf.fit(z_tr[keep], y_tr[keep])
    return clf.predict(z_te)


def heatmap(M: pd.DataFrame, path, title):
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * M.shape[1], 1.0 + 0.55 * M.shape[0]))
    im = ax.imshow(M.values, cmap="viridis", vmin=np.nanmin(M.values), vmax=np.nanmax(M.values))
    ax.set_xticks(range(M.shape[1]), [c.replace("_", "\n") for c in M.columns])
    ax.set_yticks(range(M.shape[0]), [f"{i} query" for i in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xlabel("predicted attribute")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    data_cfg, model_cfg, cfg = load_configs()
    run_cfg = cfg["runs"][args.run]
    name = args.run if args.seed == 17 else f"{args.run}_seed{args.seed}"
    out = ROOT / cfg["output_root"] / name
    df = load_manifest(data_cfg)
    n_folds = data_cfg["split"]["n_folds"]
    folds = sorted(int(p.name[4:]) for p in out.glob("fold*") if (p / "best.pt").exists())
    dev = device()
    amp = torch.bfloat16 if cfg["amp"] == "bf16" and dev.type == "cuda" else None

    model = build_model(data_cfg, model_cfg, run_cfg, df).to(dev)
    attrs = model.heads.attributes
    idx = label_index(data_cfg, attrs)
    texts = {**{a: q for a, q in model_cfg["queries"].items()}, **model_cfg["control_queries"],
             "none": None}
    names = list(texts)
    Y = None
    oof_probs = {n: {a: np.zeros((len(df), len(data_cfg["attributes"][a]))) for a in attrs}
                 for n in names}
    oof_probe = {n: {a: np.full(len(df), -1) for a in attrs} for n in names}
    in_test = np.zeros(len(df), bool)
    sim_rows, gate_rows, pred_rows = [], [], []
    init_state = {k: v.clone() for k, v in model.trainable_state_dict().items()}

    for k in folds:
        model.load_state_dict(init_state, strict=False)
        model.load_state_dict(torch.load(out / f"fold{k}" / "best.pt", map_location=dev,
                                         weights_only=False)["state"], strict=False)
        part, _, ds_all = fold_datasets(df, data_cfg, model_cfg, run_cfg, k, args.seed)
        part = part.to_numpy()
        Y = ds_all.labels
        Z, P, upd = extract(model, ds_all, texts, dev, amp)
        np.savez_compressed(out / f"fold{k}" / "embeddings.npz", sample_id=df.sample_id.to_numpy().astype(str),
                            part=part.astype(str), **{f"z_{n}": Z[n] for n in names})
        te, trv = part == "test", part != "test"
        in_test |= te
        for n in names:
            for a in attrs:
                oof_probs[n][a][te] = P[n][a][te]
                y = Y[:, idx[a]]
                oof_probe[n][a][te] = probe_predict(Z[n][trv], y[trv], Z[n][te])
        # representation similarity on the test fold
        base = Z["none"][te]
        mu = base.mean(0)
        for i, a in enumerate(names):
            for b in names[i:]:
                za, zb = Z[a][te], Z[b][te]
                sim_rows.append({"fold": k, "rep_a": a, "rep_b": b,
                                 "cosine": float(cosine_rows(za, zb).mean()),
                                 "centered_cosine": float(cosine_rows(za - mu, zb - mu).mean()),
                                 "cka": linear_cka(za, zb),
                                 "rel_l2_distance": float((np.linalg.norm(za - zb, axis=1) /
                                                           np.linalg.norm(zb, axis=1)).mean())})
        if model.controller is not None:
            for layer, g in model.controller.gates().items():
                row = {"fold": k, "layer": layer, "gate_tanh_alpha": g}
                for n in names:
                    if n != "none":
                        row[f"rel_update_{n}"] = upd[n].get(layer, float("nan"))
                gate_rows.append(row)
        for i in np.where(te)[0]:
            for n in names:
                for a in attrs:
                    p = P[n][a][i]
                    pred_rows.append({
                        "sample_id": df.sample_id[i], "patient_id": df.patient_id[i],
                        "query": n, "query_text": texts[n], "attribute": a,
                        "attribute_gt": df[a][i], "predicted_class": data_cfg["attributes"][a][int(p.argmax())],
                        "probabilities": json.dumps([round(float(v), 5) for v in p]),
                        "malignancy_gt": None if pd.isna(df.malignancy[i]) else int(df.malignancy[i]),
                        "malignancy_probability": None, "split": f"test_fold{k}", "fold": k})
        print(f"fold {k} done", flush=True)

    assert in_test.all() or len(folds) < n_folds
    sel = np.where(in_test)[0]
    # ---- head-based metrics per query (controls)
    head_M, controls = {}, {}
    for n in names:
        head_M[n] = {}
        for a in attrs:
            y = Y[sel, idx[a]]
            pr = oof_probs[n][a][sel].argmax(1)
            head_M[n][a] = attribute_metrics(y, pr)
    ci = {}
    for a in attrs:
        y = Y[sel, idx[a]]
        pr = oof_probs[a][a][sel].argmax(1) if a in names else None
        keep = y >= 0
        ci[a] = bootstrap(macro_f1, y[keep], pr[keep])
        wrong = [q for q in model_cfg["queries"] if q != a]
        controls[a] = {
            "correct_query": head_M[a][a],
            "wrong_query_mean_macro_f1": float(np.mean([head_M[q][a]["macro_f1"] for q in wrong])),
            "wrong_query_per_query": {q: head_M[q][a]["macro_f1"] for q in wrong},
            "empty_query": head_M["empty"][a], "generic_query": head_M["generic"][a],
            "no_steering_same_heads": head_M["none"][a]}
    # ---- probe-based selectivity matrix
    probe_M = pd.DataFrame({a: {n: attribute_metrics(Y[sel, idx[a]], oof_probe[n][a][sel])["macro_f1"]
                                for n in names} for a in attrs})
    probe_bacc = pd.DataFrame({a: {n: attribute_metrics(Y[sel, idx[a]], oof_probe[n][a][sel])["balanced_acc"]
                                   for n in names} for a in attrs})
    head_df = pd.DataFrame({a: {n: head_M[n][a]["macro_f1"] for n in names} for a in attrs})
    sq = probe_M.loc[attrs, attrs].to_numpy()
    sq_h = head_df.loc[attrs, attrs].to_numpy()
    off = ~np.eye(len(attrs), dtype=bool)
    selectivity = {"probe_macro_f1": float(np.diag(sq).mean() - sq[off].mean()),
                   "head_transfer_macro_f1": float(np.diag(sq_h).mean() - sq_h[off].mean()),
                   "probe_diag_beats_column_rate": float(np.mean(
                       [sq[j, j] > sq[np.arange(len(attrs)) != j, j].max() for j in range(len(attrs))]))}
    probe_M.index.name = head_df.index.name = probe_bacc.index.name = "query"
    probe_M.to_csv(out / "steering_selectivity_matrix.csv")
    probe_bacc.to_csv(out / "steering_selectivity_matrix_balanced_acc.csv")
    head_df.to_csv(out / "steering_selectivity_matrix_heads.csv")
    heatmap(probe_M, out / "steering_selectivity_heatmap.png",
            f"{name}: linear-probe macro-F1 (OOF)\nselectivity={selectivity['probe_macro_f1']:+.3f}")
    heatmap(head_df, out / "steering_selectivity_heatmap_heads.png",
            f"{name}: trained head on query-i embedding (macro-F1)")

    sim = pd.DataFrame(sim_rows)
    sim.to_csv(out / "representation_similarity_per_fold.csv", index=False)
    sim_mean = sim.groupby(["rep_a", "rep_b"], sort=False).mean(numeric_only=True).drop(columns="fold")
    sim_mean.reset_index().to_csv(out / "representation_similarity.csv", index=False)
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(out / "gate_values.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)

    # ---- has the controller learned to use the text?
    flags = []
    if model.controller is not None:
        for a in attrs:
            c = controls[a]
            gap = c["correct_query"]["macro_f1"] - c["wrong_query_mean_macro_f1"]
            if abs(gap) < 0.02:
                flags.append(f"{a}: correct vs wrong query macro-F1 differ by {gap:+.3f} (<0.02)")
        pair = sim_mean.loc[[(x, y) for x, y in sim_mean.index if x in attrs and y in attrs and x != y]]
        if len(pair) and pair["cosine"].min() > 0.99:
            flags.append(f"attribute-query embeddings nearly identical (min cosine {pair['cosine'].min():.4f})")
        if gates["gate_tanh_alpha"].abs().max() < 1e-2:
            flags.append("all gates |tanh(alpha)| < 0.01")
        if flags:
            flags.insert(0, "WARNING: possible absence of real steering")
    metrics = {"run": name, "folds": folds, "n_test_samples": int(in_test.sum()),
               "backbone": model_cfg["backbones"][run_cfg["backbone"]],
               "text_encoder": model_cfg["text_encoder"]["name"], "params": count_params(model),
               "attributes": {a: {**head_M[a][a], "macro_f1_ci95": ci[a]} if a in names else None
                              for a in attrs},
               "mean_macro_f1": float(np.mean([head_M[a][a]["macro_f1"] for a in attrs])),
               "controls": controls, "selectivity": selectivity,
               "probe_matrix_macro_f1": probe_M.to_dict(), "flags": flags}
    if len(gates):
        metrics["gates_mean_by_layer"] = gates.groupby("layer")["gate_tanh_alpha"].mean().to_dict()
        fig, ax = plt.subplots(figsize=(5, 3))
        g = gates.groupby("layer")["gate_tanh_alpha"]
        ax.bar(g.mean().index.astype(str), g.mean().values, yerr=g.std().values)
        ax.set_xlabel("block after which steering is applied")
        ax.set_ylabel("tanh(alpha)")
        fig.tight_layout()
        fig.savefig(out / "gate_values.png", dpi=150)
        plt.close(fig)
    save_json(metrics, out / "metrics.json")
    print(json.dumps({k: metrics[k] for k in ("mean_macro_f1", "selectivity", "flags")}, indent=2))
    print(probe_M.round(3).to_string())


if __name__ == "__main__":
    main()
