"""COCO-style detection metrics (pycocotools), so numbers are comparable with CVA-Net /
RF-DETR on BUV: mAP@[0.5:0.95], AP50, AP75 and per-class AP.

Predictions and targets are passed in normalised xyxy coordinates together with each image's
original (h, w); boxes are scored in original pixels, as the reference implementations do.
"""

from __future__ import annotations

import contextlib
import io

import torch


class CocoDetectionEvaluator:
    def __init__(self, classes: list[str]) -> None:
        self.classes = list(classes)
        self.images: list[dict] = []
        self.gts: list[dict] = []
        self.dts: list[dict] = []

    @staticmethod
    def _to_xywh(boxes: torch.Tensor, size: tuple[int, int]) -> list[list[float]]:
        h, w = size
        b = boxes.float().cpu() * torch.tensor([w, h, w, h])
        return torch.cat([b[:, :2], b[:, 2:] - b[:, :2]], dim=1).tolist()

    @torch.no_grad()
    def update(self, preds: list[dict], targets: list[dict], sizes: list[tuple[int, int]]) -> None:
        """preds[i] = {"scores", "labels", "boxes" (normalised xyxy)};
        targets[i] = {"labels", "boxes" (normalised cxcywh)}; sizes[i] = original (h, w)."""
        from biomllm.training.detection import box_cxcywh_to_xyxy

        for p, t, size in zip(preds, targets, sizes):
            img_id = len(self.images) + 1
            self.images.append({"id": img_id, "height": int(size[0]), "width": int(size[1])})
            for box, label in zip(self._to_xywh(box_cxcywh_to_xyxy(t["boxes"]), size), t["labels"].tolist()):
                self.gts.append({"id": len(self.gts) + 1, "image_id": img_id, "category_id": label + 1,
                                 "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
            for box, label, score in zip(self._to_xywh(p["boxes"], size), p["labels"].tolist(),
                                         p["scores"].tolist()):
                self.dts.append({"image_id": img_id, "category_id": label + 1, "bbox": box,
                                 "score": score})

    def compute(self) -> dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt = COCO()
        gt.dataset = {"images": self.images, "annotations": self.gts,
                      "categories": [{"id": i + 1, "name": c} for i, c in enumerate(self.classes)]}
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            if not self.dts:
                return {"map": 0.0, "map50": 0.0, "map75": 0.0, "n_images": len(self.images)}
            dt = gt.loadRes(self.dts)
            ev = COCOeval(gt, dt, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        out = {"map": float(s[0]), "map50": float(s[1]), "map75": float(s[2]),
               "ar100": float(s[8]), "n_images": len(self.images)}
        # per-class AP@[.5:.95]: precision[T, R, K, A=all, M=maxDets 100]
        prec = ev.eval["precision"][:, :, :, 0, -1]
        for k, c in enumerate(self.classes):
            p = prec[:, :, k]
            p = p[p > -1]
            out[f"ap/{c}"] = float(p.mean()) if p.size else float("nan")
        return out
