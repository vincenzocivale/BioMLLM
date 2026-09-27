"""Shared layout for the ultrasound segmentation / detection benchmark suite.

Every dataset is converted to

    <dst>/images/<id>.png            RGB (grey replicated), original resolution
    <dst>/masks/<id>.png             uint8 label map: 0 background, k = classes[k-1]
    <dst>/{train,val,test}.json      [{"file", "mask", "height", "width", "boxes", "labels",
                                       "group"}, ...]
    <dst>/classes.json               structure names (label k -> classes[k-1])
    <dst>/meta.json                  source, license, citation, split rule, statistics, leakage audit

`boxes` are xyxy pixels derived from the masks, in the same record format as BUV (so the
grounding runner and the COCO evaluator work unchanged). `labels` are 0-based class indices.
Lesion datasets get one box per connected component; anatomical-structure datasets one box per
structure. `group` is the unit that must not cross splits (patient / case / video). Splits are
the official ones when they exist, otherwise a seeded split by group.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def boxes_from_mask(mask: np.ndarray, n_classes: int, per_component: bool,
                    min_area: int = 16) -> tuple[list[list[int]], list[int]]:
    from scipy import ndimage

    boxes, labels = [], []
    for k in range(1, n_classes + 1):
        m = mask == k
        if not m.any():
            continue
        if per_component:
            lab, n = ndimage.label(m)
            for sl, idx in zip(ndimage.find_objects(lab), range(1, n + 1)):
                if (lab[sl] == idx).sum() < min_area:
                    continue
                boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
                labels.append(k - 1)
        else:
            ys, xs = np.nonzero(m)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            labels.append(k - 1)
    return boxes, labels


def fill_ellipse_contour(contour: np.ndarray) -> np.ndarray | None:
    """Filled ellipse from a (possibly broken) ellipse outline: direct least-squares conic fit
    (Fitzgibbon et al., 1999) to the outline pixels, rasterised as {F(x, y) <= 0}."""
    ys, xs = np.nonzero(contour)
    if len(xs) < 6:
        return None
    mx, my, sc = xs.mean(), ys.mean(), max(xs.std(), ys.std())
    x, y = (xs - mx) / sc, (ys - my) / sc
    D = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    evals, evecs = np.linalg.eig(np.linalg.solve(S, C))
    a = None
    for k in range(6):
        v = np.real(evecs[:, k])
        if 4 * v[0] * v[2] - v[1] ** 2 > 0:
            a = v
            break
    if a is None:
        return None
    h, w = contour.shape
    gy, gx = np.mgrid[0:h, 0:w]
    X, Y = (gx - mx) / sc, (gy - my) / sc
    F = a[0] * X * X + a[1] * X * Y + a[2] * Y * Y + a[3] * X + a[4] * Y + a[5]
    inside = F <= 0 if F[int(my), int(mx)] <= 0 else F >= 0
    return inside


def dhash(img: Image.Image, size: int = 16) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class Writer:
    """Collects samples, writes images / masks, then splits and audits them."""

    def __init__(self, dst: Path, classes: list[str], per_component: bool) -> None:
        self.dst, self.classes, self.per_component = Path(dst), list(classes), per_component
        (self.dst / "images").mkdir(parents=True, exist_ok=True)
        (self.dst / "masks").mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.hashes: dict[str, int] = {}

    def add(self, sid: str, image: Image.Image, mask: np.ndarray, group: str, split: str | None = None,
            extra: dict | None = None) -> None:
        sid = sid.replace("/", "__").replace(" ", "_")
        image = image.convert("RGB")
        if mask.shape != (image.height, image.width):
            raise ValueError(f"{sid}: mask {mask.shape} vs image {(image.height, image.width)}")
        mask = mask.astype(np.uint8)
        image.save(self.dst / "images" / f"{sid}.png")
        Image.fromarray(mask).save(self.dst / "masks" / f"{sid}.png")
        boxes, labels = boxes_from_mask(mask, len(self.classes), self.per_component)
        self.hashes[sid] = dhash(image)
        self.records.append({"file": f"images/{sid}.png", "mask": f"masks/{sid}.png", "height": image.height,
                             "width": image.width, "boxes": boxes, "labels": labels, "group": group,
                             "split": split, **(extra or {})})

    def merge_near_duplicates(self, max_dist: int = 6) -> int:
        """Union near-duplicate images (dHash distance <= max_dist) of the not-yet-split records into
        one group, so they cannot end up in different splits (datasets without patient ids)."""
        free = [r for r in self.records if r["split"] is None]
        parent = {r["group"]: r["group"] for r in free}

        def find(g):
            while parent[g] != g:
                parent[g] = parent[parent[g]]
                g = parent[g]
            return g
        hs = [(r["group"], self.hashes[Path(r["file"]).stem]) for r in free]
        merged = 0
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                if bin(hs[i][1] ^ hs[j][1]).count("1") <= max_dist:
                    a, b = find(hs[i][0]), find(hs[j][0])
                    if a != b:
                        parent[b] = a
                        merged += 1
        for r in free:
            r["group"] = find(r["group"])
        return merged

    def finish(self, meta: dict, split_fracs=(0.7, 0.1, 0.2), seed: int = 0, dedup: bool = False) -> dict:
        if dedup:
            meta = {**meta, "near_duplicate_merges_before_split": self.merge_near_duplicates()}
        if any(r["split"] is None for r in self.records):
            groups = sorted({r["group"] for r in self.records if r["split"] is None})
            random.Random(seed).shuffle(groups)
            n = len(groups)
            cut = [round(split_fracs[0] * n), round((split_fracs[0] + split_fracs[1]) * n)]
            assign = {g: ("train" if i < cut[0] else "val" if i < cut[1] else "test") for i, g in enumerate(groups)}
            for r in self.records:
                if r["split"] is None:
                    r["split"] = assign[r["group"]]
        by_split = defaultdict(list)
        for r in self.records:
            by_split[r["split"]].append(r)
        for split, rows in by_split.items():
            (self.dst / f"{split}.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "split"}
                                                                 for r in rows]))
        (self.dst / "classes.json").write_text(json.dumps(self.classes))
        meta = dict(meta)
        meta["classes"] = self.classes
        meta["boxes"] = "one per connected component" if self.per_component else "one per structure"
        meta["stats"] = {s: {"images": len(rows), "groups": len({r["group"] for r in rows}),
                             "empty_masks": sum(not r["boxes"] for r in rows),
                             "boxes": sum(len(r["boxes"]) for r in rows)} for s, rows in sorted(by_split.items())}
        meta["group_leakage"] = sorted({r["group"] for r in by_split.get("test", [])}
                                       & {r["group"] for r in by_split.get("train", [])})[:20]
        meta["near_duplicates_across_splits"] = self.audit(by_split)
        for split, a in meta["near_duplicates_across_splits"].items():
            if a["n"]:  # keep the official split, and also a leakage-free subset of it
                dup = set(a["all"])
                rows = [{k: v for k, v in r.items() if k != "split"} for r in by_split[split]
                        if Path(r["file"]).stem not in dup]
                (self.dst / f"{split}_nodup.json").write_text(json.dumps(rows))
                a["nodup_file"] = f"{split}_nodup.json ({len(rows)} images)"
            a.pop("all")
        (self.dst / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta

    def audit(self, by_split: dict, max_dist: int = 6) -> dict:
        """Near-duplicate images (dHash Hamming distance <= max_dist) between train and val/test."""
        def sid(r):
            return Path(r["file"]).stem
        train = [(sid(r), self.hashes[sid(r)]) for r in by_split.get("train", [])]
        out = {}
        for split in ("val", "test"):
            pairs = []
            for r in by_split.get(split, []):
                h = self.hashes[sid(r)]
                for tid, th in train:
                    if bin(h ^ th).count("1") <= max_dist:
                        pairs.append([sid(r), tid])
                        break
            out[split] = {"n": len(pairs), "examples": pairs[:10], "all": [p[0] for p in pairs]}
        return out
