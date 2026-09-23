"""Convert ChestX-Det (Deepwise, 3578 NIH ChestX-ray14 images, 13 lesion classes with polygons)
to the MultiLabelSegmentationFolder layout.

Source: ChestX_Det_{train,test}.json + extracted train_data.zip / test_data.zip (PNG images).
Output: <dst>/images/<stem>.png (8-bit, longest side <= --max-size),
        <dst>/masks/<stem>.png (16-bit bitmask: bit c = CLASSES[c]), classes.json,
        train.txt (official train split), val.txt (official test split).

    python scripts/prepare_data/chestx_det.py --src $BIOMLLM_DATA/raw/chestx_det --dst $BIOMLLM_DATA/chestx_det
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = [
    "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule",
    "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pleural Thickening",
    "Pneumothorax",
]


def rasterize(record: dict, size: tuple[int, int]):
    """Polygons in original pixel coordinates -> [C, H, W] bool (overlaps allowed)."""
    import numpy as np
    from PIL import Image, ImageDraw

    w, h = size
    masks = np.zeros((len(CLASSES), h, w), dtype=bool)
    for sym, poly in zip(record["syms"], record["polygons"]):
        if sym not in CLASSES or len(poly) < 3:
            continue
        canvas = Image.new("L", (w, h), 0)
        ImageDraw.Draw(canvas).polygon([tuple(map(float, p)) for p in poly], fill=1, outline=1)
        masks[CLASSES.index(sym)] |= np.asarray(canvas, dtype=bool)
    return masks


def convert(src: Path, dst: Path, max_size: int | None = 512) -> dict[str, int]:
    import numpy as np
    from PIL import Image

    from biomllm.data.datasets.segmentation import save_bitmask

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    index = {p.name: p for p in src.rglob("*.png")}
    counts = {}
    for split_json, split_name in (("ChestX_Det_train.json", "train"), ("ChestX_Det_test.json", "val")):
        records = json.loads((src / split_json).read_text())
        stems, missing = [], 0
        for rec in records:
            path = index.get(rec["file_name"])
            if path is None:
                missing += 1
                continue
            img = Image.open(path).convert("L")
            masks = rasterize(rec, img.size)
            if max_size is not None and max(img.size) > max_size:
                scale = max_size / max(img.size)
                new = (round(img.width * scale), round(img.height * scale))
                img = img.resize(new, Image.BICUBIC)
                masks = np.stack([np.asarray(Image.fromarray(m).resize(new, Image.NEAREST)) for m in masks])
            stem = Path(rec["file_name"]).stem
            img.save(dst / "images" / f"{stem}.png")
            save_bitmask(masks, dst / "masks" / f"{stem}.png")
            stems.append(stem)
        (dst / f"{split_name}.txt").write_text("\n".join(stems) + "\n")
        counts[split_name] = len(stems)
        counts[f"{split_name}_missing"] = missing
    (dst / "classes.json").write_text(json.dumps(CLASSES, indent=1))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--max-size", type=int, default=512, help="0 keeps the original resolution")
    args = ap.parse_args()
    print(convert(args.src, args.dst, args.max_size or None))


if __name__ == "__main__":
    main()
