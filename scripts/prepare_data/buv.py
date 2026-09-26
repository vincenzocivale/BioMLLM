"""Index the BUV breast-ultrasound video set (Lin et al., MICCAI 2022; CVA-Net) as frames for
DetectionFrames.

Source (extracted `Miccai 2022 BUV Dataset.7z`): rawframes/{benign,malignant}/<video>/<frame>.png,
imagenet_vid_train_15frames.json (149 videos) and imagenet_vid_val.json (37 videos), the
official split CVA-Net reports on. Frames are not copied: <dst>/images is a symlink to
rawframes, and <dst>/{train,val}.json list one record per frame:
    {"file": "benign/<video>/000000.png", "video": "benign/<video>", "height", "width",
     "boxes": [[x0, y0, x1, y1], ...] (pixels), "labels": [0 | 1, ...]}
plus classes.json. Every frame has at least one lesion; the lesion class is the video class.
Boxes with a negative width / height in the source JSON (272 of them, drawn right-to-left)
are normalised to x0 < x1, y0 < y1 rather than dropped.

Train video benign/x1282311c38f808f is dropped by default: 106 of its 110 frames are
pixel-identical to val video malignant/x9c1c965d274c457, with the opposite label (found by
the SonoBase leakage audit, see sonobase/docs/DATA_SPLITS.md).

    python scripts/prepare_data/buv.py --src $BIOMLLM_DATA/raw/buv --dst $BIOMLLM_DATA/buv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = ["benign", "malignant"]
SPLITS = {"train": "imagenet_vid_train_15frames.json", "val": "imagenet_vid_val.json"}
TRAIN_TEST_DUPLICATES = ("benign/x1282311c38f808f",)


def convert(src: Path, dst: Path, exclude: tuple[str, ...] = TRAIN_TEST_DUPLICATES) -> dict[str, dict]:
    dst.mkdir(parents=True, exist_ok=True)
    link = dst / "images"
    if not link.exists():
        link.symlink_to((src / "rawframes").resolve(), target_is_directory=True)
    (dst / "classes.json").write_text(json.dumps(CLASSES))

    stats = {}
    for split, name in SPLITS.items():
        coco = json.loads((src / name).read_text())
        cat = {c["id"]: CLASSES.index(c["name"]) for c in coco["categories"]}
        anns: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        records, dropped = [], 0
        for im in coco["images"]:
            video = im["file_name"].rsplit("/", 1)[0]
            if split == "train" and video in exclude:
                dropped += 1
                continue
            boxes, labels = [], []
            for a in anns.get(im["id"], []):
                x, y, w, h = a["bbox"]
                if w == 0 or h == 0:
                    continue
                # ~1% of boxes were drawn right-to-left / bottom-to-top (negative w or h)
                boxes.append([min(x, x + w), min(y, y + h), max(x, x + w), max(y, y + h)])
                labels.append(cat[a["category_id"]])
            records.append({"file": im["file_name"], "video": video, "height": im["height"],
                            "width": im["width"], "boxes": boxes, "labels": labels})
        (dst / f"{split}.json").write_text(json.dumps(records))
        stats[split] = {"frames": len(records), "videos": len({r["video"] for r in records}),
                        "boxes": sum(len(r["boxes"]) for r in records), "dropped_frames": dropped}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--keep-duplicate", action="store_true",
                    help="keep the train video duplicated in val (official, leaky split)")
    args = ap.parse_args()
    stats = convert(args.src, args.dst, () if args.keep_duplicate else TRAIN_TEST_DUPLICATES)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
