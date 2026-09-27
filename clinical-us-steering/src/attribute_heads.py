"""One linear classification head per attribute (label spaces differ)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeHeads(nn.Module):
    def __init__(self, dim: int, attributes: dict[str, list[str]], dropout: float = 0.1):
        super().__init__()
        self.attributes = list(attributes)
        self.heads = nn.ModuleDict({a: nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout),
                                                     nn.Linear(dim, len(c)))
                                    for a, c in attributes.items()})

    def forward(self, attribute: str, z: torch.Tensor) -> torch.Tensor:
        return self.heads[attribute](z)


def masked_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None):
    """Cross-entropy over labelled samples only (y = -1 means missing). None if nothing labelled."""
    keep = y >= 0
    if not keep.any():
        return None
    return F.cross_entropy(logits[keep].float(), y[keep], weight=weight)


def class_weights(y, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights from the training labels; unseen classes get weight 0."""
    y = torch.as_tensor(y)
    y = y[y >= 0]
    counts = torch.bincount(y, minlength=n_classes).float()
    n_present = int((counts > 0).sum())
    w = torch.where(counts > 0, len(y) / (n_present * counts.clamp(min=1)), torch.zeros_like(counts))
    return w
