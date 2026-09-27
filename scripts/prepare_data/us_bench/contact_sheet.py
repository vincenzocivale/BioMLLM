"""Visual check of a converted suite: per dataset, N test samples with the mask overlaid and the
derived boxes drawn (catches misaligned masks, wrong orientation, wrong label ids).

    python scripts/prepare_data/us_bench/contact_sheet.py --dst $BIOMLLM_DATA/us_bench --out sheet.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = [(255, 60, 60), (60, 200, 255), (255, 200, 40)]


def tile(root: Path, rec: dict, size: int = 220) -> Image.Image:
    img = np.array(Image.open(root / rec["file"]).convert("RGB")).astype(np.float32)
    m = np.array(Image.open(root / rec["mask"]))
    for k in range(1, m.max() + 1):
        img[m == k] = 0.55 * img[m == k] + 0.45 * np.array(COLORS[(k - 1) % 3])
    im = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for b, l in zip(rec["boxes"], rec["labels"]):
        d.rectangle(b, outline=COLORS[l % 3], width=max(2, im.width // 150))
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (20, 20, 20))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="/raid/DATASETS/BioMLLMData/datasets/us_bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    dst = Path(args.dst)
    names = sorted(p.name for p in dst.iterdir() if (p / "test.json").exists())
    size = 220
    sheet = Image.new("RGB", (size * args.n + 130, size * len(names)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i, n in enumerate(names):
        recs = [r for r in json.load(open(dst / n / "test.json")) if r["boxes"]]
        for j, r in enumerate(random.Random(0).sample(recs, min(args.n, len(recs)))):
            sheet.paste(tile(dst / n, r, size), (130 + j * size, i * size))
        d.text((5, i * size + size // 2), n, fill=(255, 255, 255))
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
