"""Segmentation datasets in a common format: {"image": [3, H, W] float in [0, 1],
"mask": [H, W] long label map (0 = background), "id": str}.

`SegmentationFolder` covers most public medical sets once converted to an images/ + masks/
layout with matching file stems (see scripts/prepare_data/). `SyntheticShapes` is for tests.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_image(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def _load_mask(path: Path, label_map: dict[int, int] | None) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    m = torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.int64))
    if label_map is None:  # binary masks stored as 0 / 255 (or 0 / 1)
        return (m > 0).long()
    out = torch.zeros_like(m)
    for raw, label in label_map.items():
        out[m == raw] = label
    return out


def resize_pair(image: torch.Tensor, mask: torch.Tensor, size: int | None):
    if size is None:
        return image, mask
    image = F.interpolate(image[None], size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)[0]
    mask = F.interpolate(mask[None, None].float(), size=(size, size), mode="nearest")[0, 0].long()
    return image, mask


class SegmentationFolder(Dataset):
    """root/<images_dir>/<stem>.<ext> + root/<masks_dir>/<stem>.<ext>, optional split file
    listing stems (one per line)."""

    def __init__(self, root: str, images_dir: str = "images", masks_dir: str = "masks",
                 split_file: str | None = None, image_size: int | None = 512,
                 label_map: dict[int, int] | None = None, num_classes: int = 2):
        self.root = Path(root)
        self.image_size = image_size
        self.label_map = label_map
        self.num_classes = num_classes
        masks = {p.stem: p for p in (self.root / masks_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS}
        images = {p.stem: p for p in (self.root / images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS}
        stems = sorted(set(images) & set(masks))
        if split_file is not None:
            wanted = {s.strip() for s in (self.root / split_file).read_text().splitlines() if s.strip()}
            stems = [s for s in stems if s in wanted]
        if not stems:
            raise FileNotFoundError(f"no image/mask pairs found under {self.root}")
        self.items = [(s, images[s], masks[s]) for s in stems]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        stem, img_path, mask_path = self.items[i]
        image, mask = resize_pair(_load_image(img_path), _load_mask(mask_path, self.label_map),
                                  self.image_size)
        return {"image": image, "mask": mask, "id": stem}


def load_bitmask(path: Path, num_classes: int) -> torch.Tensor:
    """16-bit PNG where bit c set = pixel belongs to class c -> [C, H, W] bool."""
    import numpy as np
    from PIL import Image

    m = torch.from_numpy(np.asarray(Image.open(path)).astype(np.int64))
    bits = torch.arange(num_classes)
    return ((m[None] >> bits[:, None, None]) & 1).bool()


def save_bitmask(masks, path: Path) -> None:
    """[C, H, W] bool array (C <= 16) -> 16-bit PNG bitmask."""
    import numpy as np
    from PIL import Image

    masks = np.asarray(masks, dtype=bool)
    if masks.shape[0] > 16:
        raise ValueError("at most 16 classes fit in a 16-bit bitmask")
    packed = np.zeros(masks.shape[1:], dtype=np.uint16)
    for c in range(masks.shape[0]):
        packed |= masks[c].astype(np.uint16) << c
    Image.fromarray(packed).save(path)


class MultiLabelSegmentationFolder(SegmentationFolder):
    """Like SegmentationFolder, but masks are 16-bit bitmasks with overlapping classes.
    Returns "mask" as [C, H, W] bool (C = len(classes)). `mask_size` (default: image_size)
    lets masks be stored smaller than images: 13 classes x 512^2 x 3000 images is ~10 GB."""

    def __init__(self, root: str, classes: list[str], images_dir: str = "images",
                 masks_dir: str = "masks", split_file: str | None = None,
                 image_size: int | None = 512, mask_size: int | None = None):
        super().__init__(root, images_dir, masks_dir, split_file, image_size, num_classes=len(classes))
        self.classes = list(classes)
        self.mask_size = mask_size or image_size

    def __getitem__(self, i: int) -> dict:
        stem, img_path, mask_path = self.items[i]
        image = _load_image(img_path)
        mask = load_bitmask(mask_path, len(self.classes))
        if self.image_size is not None:
            s = (self.image_size, self.image_size)
            image = F.interpolate(image[None], size=s, mode="bilinear", align_corners=False,
                                  antialias=True)[0]
        if self.mask_size is not None:
            s = (self.mask_size, self.mask_size)
            mask = F.interpolate(mask[None].float(), size=s, mode="nearest")[0].bool()
        return {"image": image, "mask": mask, "id": stem}


class SyntheticShapes(Dataset):
    """Random ellipses on textured noise; the ellipse interior is class 1."""

    def __init__(self, n: int = 64, image_size: int = 64, seed: int = 0, num_classes: int = 2,
                 multilabel: bool = False):
        """multilabel=True: two overlapping ellipse classes, mask [2, H, W] bool."""
        self.n, self.size, self.seed, self.num_classes = n, image_size, seed, num_classes
        self.multilabel = multilabel

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        s = self.size
        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        cy, cx = (torch.rand(2, generator=g) * 0.6 + 0.2) * s
        ry, rx = (torch.rand(2, generator=g) * 0.25 + 0.1) * s
        mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).long()
        image = torch.rand(3, s, s, generator=g) * 0.3
        image[:, mask.bool()] += 0.5
        if self.multilabel:
            # second class: left half of the ellipse, encoded in the red channel only
            half = mask.bool() & (xx < cx)
            image[0, half] += 0.3
            return {"image": image.clamp(0, 1), "mask": torch.stack([mask.bool(), half]), "id": f"synth_{i}"}
        return {"image": image.clamp(0, 1), "mask": mask, "id": f"synth_{i}"}
