"""DETR-style set prediction loss for the `det` task tokens.

Each of the Q [DET] queries predicts per-class logits (sigmoid, no "no-object" class) and a
normalised cxcywh box. Queries are matched one-to-one to ground-truth boxes (Hungarian);
matched queries get focal + L1 + GIoU losses, unmatched ones are pushed to background by
the focal term. Weights default to configs/task/det.yaml (cls 2, l1 5, giou 2), the usual
Deformable-DETR / RF-DETR choice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def pairwise_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [N, 4], b: [M, 4] xyxy -> [N, M] generalised IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(a[:, None, :2], b[None, :, :2])
    rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb_c - lt_c).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-7)


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25,
                       gamma: float = 2.0) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    return (alpha * targets + (1 - alpha) * (1 - targets)) * loss


class HungarianMatcher(nn.Module):
    def __init__(self, cost_cls: float = 2.0, cost_l1: float = 5.0, cost_giou: float = 2.0,
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.cost_cls, self.cost_l1, self.cost_giou = cost_cls, cost_l1, cost_giou
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, boxes: torch.Tensor,
                targets: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """logits [B, Q, C], boxes [B, Q, 4] cxcywh; targets[b] = {"labels": [N], "boxes": [N, 4]}.
        Returns per image (query indices, target indices)."""
        from scipy.optimize import linear_sum_assignment

        out = []
        for b, t in enumerate(targets):
            if t["labels"].numel() == 0:
                empty = torch.empty(0, dtype=torch.long)
                out.append((empty, empty))
                continue
            p = logits[b].float().sigmoid()[:, t["labels"]]                    # [Q, N]
            pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
            neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
            cost = (self.cost_cls * (pos - neg)
                    + self.cost_l1 * torch.cdist(boxes[b].float(), t["boxes"].float(), p=1)
                    - self.cost_giou * pairwise_giou(box_cxcywh_to_xyxy(boxes[b].float()),
                                                     box_cxcywh_to_xyxy(t["boxes"].float())))
            qi, ti = linear_sum_assignment(cost.cpu().numpy())
            out.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(ti, dtype=torch.long)))
        return out


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, cls: float = 2.0, l1: float = 5.0, giou: float = 2.0,
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.weights = {"cls": cls, "l1": l1, "giou": giou}
        self.alpha, self.gamma = alpha, gamma
        self.matcher = HungarianMatcher(cls, l1, giou, alpha, gamma)

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor,
                targets: list[dict]) -> dict[str, torch.Tensor]:
        logits, boxes = logits.float(), boxes.float()
        indices = self.matcher(logits, boxes, targets)
        num_boxes = max(sum(t["labels"].numel() for t in targets), 1)

        cls_target = torch.zeros_like(logits)
        src_boxes, tgt_boxes = [], []
        for b, (qi, ti) in enumerate(indices):
            cls_target[b, qi, targets[b]["labels"][ti]] = 1.0
            src_boxes.append(boxes[b, qi])
            tgt_boxes.append(targets[b]["boxes"][ti].float())
        src, tgt = torch.cat(src_boxes), torch.cat(tgt_boxes)

        loss_cls = sigmoid_focal_loss(logits, cls_target, self.alpha, self.gamma).sum() / num_boxes
        loss_l1 = F.l1_loss(src, tgt, reduction="sum") / num_boxes
        giou = torch.diag(pairwise_giou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)))
        loss_giou = (1 - giou).sum() / num_boxes
        parts = {"cls": loss_cls, "l1": loss_l1, "giou": loss_giou}
        parts["loss"] = sum(self.weights[k] * v for k, v in parts.items())
        return parts


@torch.no_grad()
def postprocess(logits: torch.Tensor, boxes: torch.Tensor, top_k: int = 100) -> list[dict]:
    """Top-k (query, class) pairs per image, DETR/RF-DETR style (a query can score for
    several classes). Returns normalised xyxy boxes."""
    b, q, c = logits.shape
    scores = logits.float().sigmoid().view(b, -1)
    k = min(top_k, q * c)
    top, idx = scores.topk(k, dim=1)
    query, label = idx // c, idx % c
    xyxy = box_cxcywh_to_xyxy(boxes.float()).clamp(0, 1)
    return [{"scores": top[i], "labels": label[i],
             "boxes": xyxy[i].gather(0, query[i, :, None].expand(-1, 4))} for i in range(b)]
