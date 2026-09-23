"""Dense-perception metrics.

    Dice / gIoU  mean over samples (sensitive to small targets)
    cIoU         cumulative intersection / cumulative union over the dataset
    Acc@0.5      box derived from the predicted mask vs box derived from the GT mask
"""

from __future__ import annotations

import torch


def mask_to_box(mask: torch.Tensor) -> torch.Tensor | None:
    """[H, W] bool -> (x0, y0, x1, y1) inclusive-exclusive, or None for an empty mask."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    return torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()


def box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ix = (torch.minimum(a[2], b[2]) - torch.maximum(a[0], b[0])).clamp(min=0)
    iy = (torch.minimum(a[3], b[3]) - torch.maximum(a[1], b[1])).clamp(min=0)
    inter = ix * iy
    area = lambda x: (x[2] - x[0]) * (x[3] - x[1])  # noqa: E731
    return float(inter / (area(a) + area(b) - inter))


class BinarySegMetrics:
    """Accumulates binary-mask metrics. Empty-GT / empty-prediction samples count as a perfect
    match (Dice = IoU = 1), following the gRefCOCO no-target convention."""

    def __init__(self) -> None:
        self.dice: list[float] = []
        self.iou: list[float] = []
        self.box_hits: list[float] = []
        self.inter = 0
        self.union = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: [B, H, W] or [H, W], bool / {0,1}."""
        if pred.dim() == 2:
            pred, target = pred[None], target[None]
        for p, t in zip(pred.bool(), target.bool()):
            inter = (p & t).sum().item()
            union = (p | t).sum().item()
            total = p.sum().item() + t.sum().item()
            self.inter += inter
            self.union += union
            self.iou.append(inter / union if union else 1.0)
            self.dice.append(2 * inter / total if total else 1.0)
            pb, tb = mask_to_box(p), mask_to_box(t)
            if tb is not None:
                self.box_hits.append(float(pb is not None and box_iou(pb, tb) >= 0.5))

    def compute(self) -> dict[str, float]:
        n = max(len(self.dice), 1)
        return {
            "dice": sum(self.dice) / n,
            "giou": sum(self.iou) / n,
            "ciou": self.inter / self.union if self.union else 1.0,
            "acc_box50": sum(self.box_hits) / len(self.box_hits) if self.box_hits else float("nan"),
            "n": len(self.dice),
        }


class MultiClassSegMetrics:
    """Per-class binary metrics for label maps with classes 1..C-1 (0 = background)."""

    def __init__(self, num_classes: int) -> None:
        self.per_class = {c: BinarySegMetrics() for c in range(1, num_classes)}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        for c, m in self.per_class.items():
            m.update(pred == c, target == c)

    def compute(self) -> dict[str, float]:
        per = {c: m.compute() for c, m in self.per_class.items()}
        out = {f"{k}_c{c}": v[k] for c, v in per.items() for k in ("dice", "giou")}
        for k in ("dice", "giou", "ciou"):
            out[k] = sum(v[k] for v in per.values()) / len(per)
        return out


class MultiLabelSegMetrics:
    """Overlapping classes (lesions). Per class:
        dice_pos / giou_pos  mean over images where the class is present (lesion quality)
        ciou                 cumulative over all images (false positives on negatives count)
    Macro averages over classes are reported as dice, giou, ciou."""

    def __init__(self, classes: list[str]) -> None:
        self.classes = list(classes)
        self.pos = {c: BinarySegMetrics() for c in self.classes}
        self.inter = {c: 0 for c in self.classes}
        self.union = {c: 0 for c in self.classes}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: [B, C, H, W] bool."""
        for k, c in enumerate(self.classes):
            p, t = pred[:, k].bool(), target[:, k].bool()
            self.inter[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()
            has = t.flatten(1).any(1)
            if has.any():
                self.pos[c].update(p[has], t[has])

    def compute(self) -> dict[str, float]:
        out = {}
        for c in self.classes:
            r = self.pos[c].compute()
            out[f"dice_pos/{c}"] = r["dice"] if r["n"] else float("nan")
            out[f"giou_pos/{c}"] = r["giou"] if r["n"] else float("nan")
            out[f"ciou/{c}"] = self.inter[c] / self.union[c] if self.union[c] else float("nan")
            out[f"n_pos/{c}"] = r["n"]
        valid = [c for c in self.classes if self.pos[c].dice]
        for k in ("dice_pos", "giou_pos", "ciou"):
            vals = [out[f"{k}/{c}"] for c in valid]
            out[k.replace("_pos", "")] = sum(vals) / len(vals) if vals else float("nan")
        return out
