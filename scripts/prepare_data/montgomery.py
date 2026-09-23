"""Convert the Montgomery County CXR set to the SegmentationFolder layout.

Source (open, NLM): MontgomerySet/{CXR_png, ManualMask/leftMask, ManualMask/rightMask}.
Output: <dst>/images/<stem>.png, <dst>/masks/<stem>.png (both lungs = 255), train.txt, val.txt.
Images and masks are downscaled to --max-size (longest side, default 1024) to keep I/O light.

    python scripts/prepare_data/montgomery.py --src /path/MontgomerySet --dst $BIOMLLM_DATA/montgomery
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def _downscale(img, max_size: int | None, resample):
    if max_size is None or max(img.size) <= max_size:
        return img
    scale = max_size / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), resample)


def convert(src: Path, dst: Path, val_fraction: float = 0.2, seed: int = 0,
            max_size: int | None = None) -> tuple[int, int]:
    """max_size: downscale so the longest side is at most this (originals are ~4000x4900 px)."""
    import numpy as np
    from PIL import Image

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    stems = []
    for img in sorted((src / "CXR_png").glob("*.png")):
        left = src / "ManualMask" / "leftMask" / img.name
        right = src / "ManualMask" / "rightMask" / img.name
        if not (left.exists() and right.exists()):
            continue
        mask = (np.asarray(Image.open(left).convert("L")) > 0) | (np.asarray(Image.open(right).convert("L")) > 0)
        mask_img = Image.fromarray(mask.astype("uint8") * 255)
        _downscale(mask_img, max_size, Image.NEAREST).save(dst / "masks" / img.name)
        if max_size is None:
            shutil.copyfile(img, dst / "images" / img.name)
        else:
            _downscale(Image.open(img), max_size, Image.BICUBIC).save(dst / "images" / img.name)
        stems.append(img.stem)
    random.Random(seed).shuffle(stems)
    n_val = max(1, round(len(stems) * val_fraction)) if stems else 0
    (dst / "val.txt").write_text("\n".join(sorted(stems[:n_val])) + "\n")
    (dst / "train.txt").write_text("\n".join(sorted(stems[n_val:])) + "\n")
    return len(stems) - n_val, n_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-size", type=int, default=1024, help="0 keeps the original resolution")
    args = ap.parse_args()
    n_train, n_val = convert(args.src, args.dst, args.val_fraction, max_size=args.max_size or None)
    print(f"wrote {n_train} train / {n_val} val pairs to {args.dst}")


if __name__ == "__main__":
    main()
