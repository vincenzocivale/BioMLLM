"""Experiment 0: linear segmentation probes on frozen encoders, one row per expert.

    python scripts/probe_experts.py                                   # synthetic smoke test
    python scripts/probe_experts.py probe_data=montgomery_lungs \\
        experts=[dinov2,siglip,rad_dino,biomedclip]

Writes results.json and results.md (expert x metric) to the Hydra run directory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from biomllm.data.datasets.segmentation import (MultiLabelSegmentationFolder, SegmentationFolder,
                                                SyntheticShapes)
from biomllm.models.build import build_expert_from_cfg
from biomllm.probing.linear_probe import evaluate_probe, extract_features, train_probe

log = logging.getLogger(__name__)
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def build_split(node: DictConfig, num_classes: int, classes: list[str] | None = None):
    kw = OmegaConf.to_container(node, resolve=True)
    kind = kw.pop("kind")
    if kind == "synthetic":
        return SyntheticShapes(num_classes=num_classes, multilabel=classes is not None, **kw)
    if kind == "folder":
        return SegmentationFolder(num_classes=num_classes, **kw)
    if kind == "multilabel_folder":
        return MultiLabelSegmentationFolder(classes=classes, **kw)
    raise KeyError(f"unknown dataset kind '{kind}'")


def load_expert_cfg(name: str) -> DictConfig:
    path = CONFIGS / "expert" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no expert config {path}")
    return OmegaConf.load(path)


def to_markdown(rows: list[dict], metrics: list[str]) -> str:
    head = "| expert | dim | grid | " + " | ".join(metrics) + " | extract s |"
    sep = "|" + "---|" * (len(metrics) + 4)
    lines = [head, sep]
    for r in rows:
        vals = " | ".join(f"{r['val'][m]:.4f}" for m in metrics)
        lines.append(f"| {r['expert']} | {r['dim']} | {r['grid']} | {vals} | {r['extract_seconds']:.1f} |")
    return "\n".join(lines)


def per_class_markdown(rows: list[dict], classes: list[str], metric: str = "dice_pos") -> str:
    head = "| class | n_pos | " + " | ".join(r["expert"] for r in rows) + " |"
    lines = [head, "|" + "---|" * (len(rows) + 2)]
    for c in classes:
        vals = " | ".join(f"{r['val'][f'{metric}/{c}']:.4f}" for r in rows)
        lines.append(f"| {c} | {rows[0]['val'][f'n_pos/{c}']} | {vals} |")
    return "\n".join(lines)


@hydra.main(config_path="../configs", config_name="probe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    classes = list(cfg.probe_data.classes) if cfg.probe_data.get("classes") else None
    nc = len(classes) if classes else cfg.probe_data.num_classes
    train_set = build_split(cfg.probe_data.train, nc, classes)
    val_set = build_split(cfg.probe_data.val, nc, classes)
    log.info("dataset %s: %d train / %d val", cfg.probe_data.name, len(train_set), len(val_set))

    out = Path(cfg.output_dir)
    rows = []
    for name in cfg.experts:
        expert = build_expert_from_cfg(load_expert_cfg(name))
        t0 = time.perf_counter()
        tr_f, tr_m = extract_features(expert, train_set, cfg.batch_size, device, cfg.num_workers)
        va_f, va_m = extract_features(expert, val_set, cfg.batch_size, device, cfg.num_workers)
        extract_s = time.perf_counter() - t0
        del expert
        if device == "cuda":
            torch.cuda.empty_cache()
        probe = train_probe(tr_f, tr_m, nc, cfg.epochs, cfg.lr, cfg.batch_size, device=device,
                            seed=cfg.seed, multilabel=classes is not None)
        row = {
            "expert": name, "dim": tr_f.shape[1], "grid": f"{tr_f.shape[2]}x{tr_f.shape[3]}",
            "extract_seconds": extract_s,
            "train": evaluate_probe(probe, tr_f, tr_m, nc, cfg.batch_size, device, classes),
            "val": evaluate_probe(probe, va_f, va_m, nc, cfg.batch_size, device, classes),
        }
        del tr_f, va_f
        log.info("%s: val %s", name, {k: round(v, 4) for k, v in row["val"].items()
                                      if isinstance(v, float) and "/" not in k})
        rows.append(row)
        # saved after every expert, so a crash on a later one does not lose finished rows
        (out / "results.json").write_text(json.dumps({"dataset": cfg.probe_data.name, "rows": rows}, indent=2))

    metrics = [m for m in ("dice", "giou", "ciou", "acc_box50") if m in rows[0]["val"]]
    table = to_markdown(rows, metrics)
    if classes:
        table += "\n\nPer-class Dice on images containing the class:\n\n" + per_class_markdown(rows, classes)
    (out / "results.md").write_text(f"# Linear probe — {cfg.probe_data.name}\n\n{table}\n")
    log.info("\n%s", table)


if __name__ == "__main__":
    main()
