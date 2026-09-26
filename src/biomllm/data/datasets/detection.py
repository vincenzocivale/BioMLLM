"""Detection datasets in a common format: {"image": [3, S, S] float in [0, 1],
"boxes": [N, 4] normalised cxcywh, "labels": [N] long, "size": (h, w) original, "id": str}.

`DetectionFrames` reads the per-frame index written by scripts/prepare_data/buv.py
(<root>/{split}.json, images under <root>/images). Images are resized to a square without
letterboxing: boxes are stored normalised, so they are unaffected (BUV frames are square).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from biomllm.data.datasets.segmentation import _load_image


class DetectionFrames(Dataset):
    def __init__(self, root: str, split: str, image_size: int | None = 512,
                 hflip: bool = False, max_frames: int | None = None):
        self.root = Path(root)
        self.image_size = image_size
        self.hflip = hflip
        self.records = json.loads((self.root / f"{split}.json").read_text())
        if max_frames is not None:
            self.records = self.records[:max_frames]
        self.classes = json.loads((self.root / "classes.json").read_text())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        image = _load_image(self.root / "images" / r["file"])
        h, w = image.shape[-2:]
        if self.image_size is not None:
            image = F.interpolate(image[None], size=(self.image_size,) * 2, mode="bilinear",
                                  align_corners=False, antialias=True)[0]
        xyxy = torch.tensor(r["boxes"], dtype=torch.float32).view(-1, 4)
        xyxy = xyxy / torch.tensor([w, h, w, h], dtype=torch.float32)
        if self.hflip and torch.rand(()) < 0.5:
            image = image.flip(-1)
            xyxy = torch.stack([1 - xyxy[:, 2], xyxy[:, 1], 1 - xyxy[:, 0], xyxy[:, 3]], dim=1)
        xyxy = xyxy.clamp(0, 1)
        boxes = torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], dim=1)
        return {"image": image, "boxes": boxes, "labels": torch.tensor(r["labels"], dtype=torch.long),
                "size": (h, w), "id": r["file"]}


def detection_collate(batch: list[dict]) -> dict:
    return {"image": torch.stack([b["image"] for b in batch]),
            "targets": [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch],
            "size": [b["size"] for b in batch], "id": [b["id"] for b in batch]}
