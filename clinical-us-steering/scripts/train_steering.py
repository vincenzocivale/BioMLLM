"""Train attribute heads (baseline) or the shared steering controller + heads, per CV fold.

    python scripts/train_steering.py --run clinical_steering_usfmae [--folds 0] [--seed 17]
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch

import _path  # noqa: F401
from attribute_heads import class_weights, masked_ce
from common import (build_model, device, fold_datasets, label_index, load_configs,
                    load_manifest)
from metrics import attribute_metrics
from utils import count_params, run_dir, save_json, save_yaml, seed_everything


@torch.no_grad()
def evaluate(model, loader, idx, dev, amp):
    model.eval()
    attrs = list(idx)
    preds = {a: [] for a in attrs}
    ys = []
    for b in loader:
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            logits = model(b["image"].to(dev))
        for a in attrs:
            preds[a].append(logits[a].argmax(-1).cpu())
        ys.append(b["labels"])
    y = torch.cat(ys).numpy()
    return {a: attribute_metrics(y[:, idx[a]], torch.cat(preds[a]).numpy()) for a in attrs}


def train_fold(args, data_cfg, model_cfg, cfg, run_cfg, df, fold, out):
    seed_everything(args.seed + fold)
    dev = device()
    amp = torch.bfloat16 if cfg["amp"] == "bf16" and dev.type == "cuda" else None
    model = build_model(data_cfg, model_cfg, run_cfg, df).to(dev)
    attrs = model.heads.attributes
    idx = label_index(data_cfg, attrs)
    part, ds, _ = fold_datasets(df, data_cfg, model_cfg, run_cfg, fold, args.seed + fold)
    g = torch.Generator().manual_seed(args.seed + fold)
    train_dl = torch.utils.data.DataLoader(ds["train"], cfg["batch_size"], shuffle=True,
                                           generator=g, num_workers=cfg["num_workers"])
    val_dl = torch.utils.data.DataLoader(ds["val"], 32, num_workers=cfg["num_workers"])
    weights = {a: (class_weights(ds["train"].labels[:, idx[a]],
                                 len(data_cfg["attributes"][a])).to(dev)
                   if cfg["class_weighted"] else None) for a in attrs}

    groups = [{"params": list(model.heads.parameters()), "lr": cfg["lr_heads"], "name": "heads"}]
    if model.controller is not None:
        gates = [p for n, p in model.controller.named_parameters() if n.endswith("alpha")]
        rest = [p for n, p in model.controller.named_parameters() if not n.endswith("alpha")]
        groups.append({"params": rest, "lr": cfg["lr_steering"], "name": "steering"})
        # zero-initialised gates move ~lr per Adam step: they need their own, larger lr
        groups.append({"params": gates, "lr": cfg["lr_gates"], "weight_decay": 0.0,
                       "name": "gates"})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(train_dl)
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    params = count_params(model)
    print(f"[fold {fold}] params {params}  train/val/test "
          f"{len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])}", flush=True)

    best, best_ep, history = -1.0, -1, []
    for ep in range(cfg["epochs"]):
        model.train()
        t0, sums, counts = time.time(), {a: 0.0 for a in attrs}, {a: 0 for a in attrs}
        tot = 0.0
        for b in train_dl:
            x, y = b["image"].to(dev), b["labels"].to(dev)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                logits = model(x)
            losses = {a: masked_ce(logits[a], y[:, idx[a]], weights[a]) for a in attrs}
            losses = {a: l for a, l in losses.items() if l is not None}
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           cfg["grad_clip"])
            opt.step()
            sched.step()
            tot += loss.item()
            for a, l in losses.items():
                sums[a] += l.item()
                counts[a] += 1
        val = evaluate(model, val_dl, idx, dev, amp)
        score = float(np.mean([val[a]["macro_f1"] for a in attrs]))
        row = {"fold": fold, "epoch": ep, "loss": tot / len(train_dl), "val_mean_macro_f1": score,
               "time_s": time.time() - t0}
        for g_ in opt.param_groups:
            row[f"lr_{g_['name']}"] = g_["lr"]
        for a in attrs:
            row[f"loss_{a}"] = sums[a] / max(counts[a], 1)
            row[f"val_macro_f1_{a}"] = val[a]["macro_f1"]
            row[f"val_bacc_{a}"] = val[a]["balanced_acc"]
        if model.controller is not None:
            for l, v in model.controller.gates().items():
                row[f"gate_{l}"] = v
        history.append(row)
        improved = score > best
        if improved:
            best, best_ep = score, ep
            torch.save({"state": model.trainable_state_dict(), "epoch": ep, "val": val,
                        "run": args.run, "fold": fold, "seed": args.seed},
                       out / f"fold{fold}" / "best.pt")
        print(f"[fold {fold}] ep {ep:3d} loss {row['loss']:.3f} val mF1 {score:.3f}"
              f"{' *' if improved else ''}" +
              (f" gates {[round(v, 3) for v in model.controller.gates().values()]}"
               if model.controller is not None else ""), flush=True)
        if ep - best_ep >= cfg["patience"]:
            break
    return history, {"best_val_mean_macro_f1": best, "best_epoch": best_ep, "params": params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    data_cfg, model_cfg, cfg = load_configs()
    run_cfg = cfg["runs"][args.run]
    args.seed = cfg["seed"] if args.seed is None else args.seed
    if args.epochs:
        cfg["epochs"] = args.epochs
    folds = cfg["folds"] if not args.folds else args.folds
    name = args.run if args.seed == 17 else f"{args.run}_seed{args.seed}"
    out = run_dir(cfg["output_root"], name)
    df = load_manifest(data_cfg)
    save_yaml({"run": args.run, "seed": args.seed, "folds": folds, "run_cfg": run_cfg,
               "train": cfg, "model": model_cfg, "data": data_cfg,
               "backbone_checkpoint": model_cfg["backbones"][run_cfg["backbone"]]},
              out / "config.yaml")
    hist, summary = [], {}
    for fold in folds:
        (out / f"fold{fold}").mkdir(exist_ok=True)
        h, s = train_fold(args, data_cfg, model_cfg, cfg, run_cfg, df, fold, out)
        hist += h
        summary[f"fold{fold}"] = s
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        save_json(summary, out / "train_summary.json")


if __name__ == "__main__":
    main()
