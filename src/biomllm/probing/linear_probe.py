"""Experiment 0: linear segmentation probes on frozen encoder features.

For each encoder, a single linear layer maps every patch feature to class logits; the logit
map is upsampled to the mask resolution. This measures how much segmentation-relevant
information each encoder exposes linearly, with no MLLM involved. The gap specialist −
generalist on a modality tells where the method has the best chance of helping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from biomllm.evaluation.metrics import BinarySegMetrics, MultiClassSegMetrics, MultiLabelSegMetrics
from biomllm.models.experts.base import FrozenExpert


@torch.no_grad()
def extract_features(expert: FrozenExpert, dataset, batch_size: int = 16, device: str = "cpu",
                     num_workers: int = 0, dtype: torch.dtype = torch.float16):
    """Returns (features [N, C, h, w] on CPU in `dtype`, masks [N, H, W] or [N, K, H, W])."""
    expert.to(device)
    feats, masks = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        fm = expert(batch["image"].to(device))
        feats.append(fm.as_image().to(dtype).cpu())
        masks.append(batch["mask"])
    return torch.cat(feats), torch.cat(masks)


class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim, affine=False)  # per-channel standardisation, still linear
        self.head = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.head(self.norm(feats))
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = logits.softmax(1)[:, 1:]
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].float()
    inter = (probs * onehot).sum((2, 3))
    denom = probs.sum((2, 3)) + onehot.sum((2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def multilabel_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-class sigmoid BCE + soft Dice; Dice only over (image, class) pairs with a lesion, so
    the many empty masks do not dominate."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    inter = (probs * target).sum((2, 3))
    denom = probs.sum((2, 3)) + target.sum((2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    pos = target.sum((2, 3)) > 0
    return bce + (dice[pos].mean() if pos.any() else logits.sum() * 0)


def train_probe(train_feats, train_masks, num_classes: int, epochs: int = 20, lr: float = 1e-3,
                batch_size: int = 16, weight_decay: float = 1e-4, device: str = "cpu",
                seed: int = 0, multilabel: bool = False) -> LinearProbe:
    """multilabel: num_classes independent sigmoid outputs (masks [N, K, H, W] bool)."""
    torch.manual_seed(seed)
    probe = LinearProbe(train_feats.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_feats, train_masks), batch_size=batch_size, shuffle=True)
    out_size = tuple(train_masks.shape[-2:])
    probe.train()
    for _ in range(epochs):
        for f, m in loader:
            f, m = f.to(device).float(), m.to(device)
            logits = probe(f, out_size)
            if multilabel:
                loss = multilabel_loss(logits, m)
            else:
                loss = F.cross_entropy(logits, m) + soft_dice_loss(logits, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def evaluate_probe(probe: LinearProbe, feats, masks, num_classes: int, batch_size: int = 16,
                   device: str = "cpu", classes: list[str] | None = None) -> dict[str, float]:
    """classes given -> multi-label evaluation (sigmoid > 0.5 per class)."""
    if classes is not None:
        metrics = MultiLabelSegMetrics(classes)
    else:
        metrics = BinarySegMetrics() if num_classes == 2 else MultiClassSegMetrics(num_classes)
    out_size = tuple(masks.shape[-2:])
    for f, m in DataLoader(TensorDataset(feats, masks), batch_size=batch_size):
        logits = probe(f.to(device).float(), out_size).cpu()
        if classes is not None:
            metrics.update(logits > 0, m)
            continue
        pred = logits.argmax(1)
        if num_classes == 2:
            metrics.update(pred == 1, m == 1)
        else:
            metrics.update(pred, m)
    return metrics.compute()
