"""Capacity-matched control for the C0-vs-C3 gap: an MLP head (768 -> 2560 -> 2560 -> 1,
the same width as C3's MLPProjector) directly on frozen RAD-DINO features, no LLM involved.

Experiment 0 (scripts/probe_experts.py) uses a *linear* head, so if C3 beats it, we don't
know whether that's from routing through the MLLM or just from a head with more non-linear
capacity. This sits between the two: same capacity as C3's projector, but no LLM. If this
already matches C3, the LLM route adds nothing beyond capacity; if it stays near the linear
probe, the MLLM route is doing real work.

Same binary "any lesion" collapse and 256px evaluation as scripts/train_seg_c0_vs_c3.py.

    python scripts/probe_mlp_control.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from biomllm.data.datasets.segmentation import MultiLabelSegmentationFolder
from biomllm.evaluation.metrics import BinarySegMetrics
from biomllm.models.experts.registry import build_expert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CLASSES = ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule",
          "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule",
          "Pleural Thickening", "Pneumothorax"]


class MLPProbe(nn.Module):
    def __init__(self, dim: int, hidden: int = 2560):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [B, N, dim] -> [B, N]
        return self.fc2(self.act(self.fc1(self.norm(tokens)))).squeeze(-1)


def collapse_binary(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch]).any(1).float()
    return {"image": images, "mask": masks}


def seg_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    inter = (probs * target).sum((1, 2))
    denom = probs.sum((1, 2)) + target.sum((1, 2))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


def main(max_steps: int = 400, batch_size: int = 4, lr: float = 3e-4, run_name: str | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = f"{os.environ.get('BIOMLLM_DATA', 'data')}/chestx_det"

    expert = build_expert("hf_vit", model_id="microsoft/rad-dino", pretrained=True, layer=-1).to(device)
    for p in expert.parameters():
        p.requires_grad_(False)
    expert.eval()

    train_ds = MultiLabelSegmentationFolder(root, CLASSES, split_file="train.txt",
                                            image_size=512, mask_size=256)
    val_ds = MultiLabelSegmentationFolder(root, CLASSES, split_file="val.txt",
                                          image_size=512, mask_size=256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collapse_binary, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collapse_binary, num_workers=4)

    probe = MLPProbe(expert.dim, hidden=2560).to(device)
    log.info("probe params: %d", sum(p.numel() for p in probe.parameters()))
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    step = 0
    probe.train()
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            images = batch["image"].to(device)
            target_256 = batch["mask"].to(device)
            with torch.no_grad():
                feats = expert(images)  # FeatureMap [B, h*w, 768], grid (37, 37)
            logits = probe(feats.tokens).view(-1, *feats.grid)  # [B, 37, 37]
            target = F.adaptive_avg_pool2d(target_256[:, None], feats.grid)[:, 0]
            loss = seg_loss(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                log.info("step %d loss %.4f", step, loss.item())

    probe.eval()
    metrics = BinarySegMetrics()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            target = batch["mask"].to(device)
            feats = expert(images)
            logits = probe(feats.tokens).view(-1, *feats.grid)
            logits_256 = F.interpolate(logits[:, None], size=target.shape[-2:], mode="bilinear",
                                       align_corners=False)[:, 0]
            metrics.update(logits_256 > 0, target.bool())
    result = metrics.compute()
    log.info("val: %s", result)

    name = run_name or "rad_dino_mlp_probe"
    out_dir = Path(os.environ.get("BIOMLLM_RUNS", "runs")) / "probe_mlp_control" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"run_name": name, "val": result}, indent=2))


if __name__ == "__main__":
    import sys

    kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    if "max_steps" in kwargs:
        kwargs["max_steps"] = int(kwargs["max_steps"])
    if "batch_size" in kwargs:
        kwargs["batch_size"] = int(kwargs["batch_size"])
    if "lr" in kwargs:
        kwargs["lr"] = float(kwargs["lr"])
    main(**kwargs)
