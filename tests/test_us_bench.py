"""Conversion logic of the ultrasound benchmark suite (scripts/prepare_data/us_bench/common.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/prepare_data/us_bench"))
from common import Writer, boxes_from_mask, fill_ellipse_contour  # noqa: E402


def test_boxes_per_component_and_per_structure():
    m = np.zeros((20, 30), np.uint8)
    m[2:5, 3:8] = 1
    m[10:15, 20:25] = 1
    m[0:3, 25:30] = 2
    boxes, labels = boxes_from_mask(m, 2, per_component=True, min_area=1)
    assert sorted(zip(labels, boxes)) == [(0, [3, 2, 8, 5]), (0, [20, 10, 25, 15]), (1, [25, 0, 30, 3])]
    boxes, labels = boxes_from_mask(m, 2, per_component=False)
    assert boxes == [[3, 2, 25, 15], [25, 0, 30, 3]] and labels == [0, 1]


def test_ellipse_fit_recovers_broken_outline():
    out = Image.new("L", (200, 150))
    ImageDraw.Draw(out).ellipse([30, 20, 170, 120], outline=255)
    a = np.array(out) > 0
    a[:, 95:105] = False
    ref = Image.new("L", (200, 150))
    ImageDraw.Draw(ref).ellipse([30, 20, 170, 120], fill=255)
    r = np.array(ref) > 0
    f = fill_ellipse_contour(a)
    assert (f & r).sum() / (f | r).sum() > 0.97


def test_near_duplicates_never_straddle_splits(tmp_path):
    rng = np.random.default_rng(0)
    w = Writer(tmp_path, ["lesion"], per_component=True)
    for i in range(30):
        base = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        for k in range(2):  # two near-identical frames per "patient", different ids
            img = Image.fromarray(np.clip(base.astype(int) + k, 0, 255).astype(np.uint8))
            mask = np.zeros((32, 32), np.uint8)
            mask[5:10, 5:10] = 1
            w.add(f"img{i}_{k}", img, mask, group=f"img{i}_{k}")
    meta = w.finish({"source": "test", "license": "-", "split": "test"}, dedup=True)
    assert meta["near_duplicate_merges_before_split"] == 30
    assert all(v["n"] == 0 for v in meta["near_duplicates_across_splits"].values())
    split_of = {}
    for s in ("train", "val", "test"):
        for r in json.load(open(tmp_path / f"{s}.json")):
            split_of[Path(r["file"]).stem] = s
    assert all(split_of[f"img{i}_0"] == split_of[f"img{i}_1"] for i in range(30))


def test_mask_shape_mismatch_is_rejected(tmp_path):
    w = Writer(tmp_path, ["lesion"], per_component=True)
    with pytest.raises(ValueError):
        w.add("x", Image.new("RGB", (10, 8)), np.zeros((10, 10), np.uint8), group="x")
