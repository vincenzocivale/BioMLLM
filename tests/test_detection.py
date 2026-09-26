"""Detection task: box ops, Hungarian set loss, COCO evaluator, BUV conversion and the [DET]
queries end to end on the toy MLLM (with and without a conditioner)."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from biomllm.data.datasets.detection import DetectionFrames, detection_collate
from biomllm.evaluation.detection import CocoDetectionEvaluator
from biomllm.models.build import build_model
from biomllm.training.detection import (
    SetCriterion,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    pairwise_giou,
    postprocess,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(overrides):
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        return compose("config", overrides=["+experiment=debug", "task=det", *overrides])


def _target(boxes, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)}


# ----------------------------------------------------------------------------- box ops

def test_box_conversion_roundtrip_and_giou():
    b = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.6, 0.1, 0.1]])
    assert torch.allclose(box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(b)), b)
    g = pairwise_giou(box_cxcywh_to_xyxy(b), box_cxcywh_to_xyxy(b))
    assert torch.allclose(g.diag(), torch.ones(2))
    far = pairwise_giou(torch.tensor([[0.0, 0.0, 0.1, 0.1]]), torch.tensor([[0.9, 0.9, 1.0, 1.0]]))
    assert far.item() < 0


# ------------------------------------------------------------------------ set criterion

def _perfect_logits(q, c, assign):
    logits = torch.full((1, q, c), -10.0)
    for qi, label in assign.items():
        logits[0, qi, label] = 10.0
    return logits


def test_matcher_finds_the_permutation():
    crit = SetCriterion(num_classes=2)
    gt = _target([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]], [0, 1])
    boxes = torch.tensor([[[0.5, 0.5, 0.3, 0.3], [0.7, 0.7, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    (qi, ti), = crit.matcher(_perfect_logits(3, 2, {2: 0, 1: 1}), boxes, [gt])
    assert dict(zip(ti.tolist(), qi.tolist())) == {0: 2, 1: 1}


def test_criterion_perfect_vs_wrong_and_gradients():
    crit = SetCriterion(num_classes=2)
    gt = [_target([[0.2, 0.2, 0.1, 0.1]], [1])]
    good_boxes = torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]])
    good = crit(_perfect_logits(2, 2, {0: 1}), good_boxes, gt)
    assert good["l1"].item() < 1e-6 and good["giou"].item() < 1e-6
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    boxes = torch.full((1, 2, 4), 0.5, requires_grad=True)
    bad = crit(logits, boxes, gt)
    assert bad["loss"] > good["loss"]
    bad["loss"].backward()
    assert logits.grad.abs().sum() > 0 and boxes.grad.abs().sum() > 0


def test_criterion_handles_an_image_without_boxes():
    crit = SetCriterion(num_classes=1)
    empty = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
    out = crit(torch.zeros(2, 3, 1), torch.full((2, 3, 4), 0.5),
               [empty, _target([[0.5, 0.5, 0.2, 0.2]], [0])])
    assert torch.isfinite(out["loss"])


def test_postprocess_topk_shapes_and_range():
    logits, boxes = torch.randn(2, 5, 3), torch.rand(2, 5, 4)
    preds = postprocess(logits, boxes, top_k=7)
    assert len(preds) == 2 and preds[0]["boxes"].shape == (7, 4)
    assert preds[0]["labels"].max() < 3 and 0 <= preds[0]["boxes"].min() <= preds[0]["boxes"].max() <= 1
    assert torch.all(preds[0]["scores"][:-1] >= preds[0]["scores"][1:])


# ------------------------------------------------------------------------- COCO metrics

def test_coco_evaluator_perfect_and_empty():
    gt = [_target([[0.3, 0.3, 0.2, 0.2]], [0]), _target([[0.6, 0.5, 0.3, 0.4]], [1])]
    perfect = [{"boxes": box_cxcywh_to_xyxy(t["boxes"]), "labels": t["labels"],
                "scores": torch.ones(1)} for t in gt]
    ev = CocoDetectionEvaluator(["benign", "malignant"])
    ev.update(perfect, gt, [(400, 400), (600, 600)])
    r = ev.compute()
    assert r["map"] == pytest.approx(1.0) and r["ap/benign"] == pytest.approx(1.0)

    wrong_class = [dict(p, labels=1 - p["labels"]) for p in perfect]
    ev = CocoDetectionEvaluator(["benign", "malignant"])
    ev.update(wrong_class, gt, [(400, 400), (600, 600)])
    assert ev.compute()["map"] == pytest.approx(0.0)

    ev = CocoDetectionEvaluator(["benign", "malignant"])
    none = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
             "scores": torch.zeros(0)}] * 2
    ev.update(none, gt, [(400, 400), (600, 600)])
    assert ev.compute()["map"] == 0.0


# ------------------------------------------------------------------ BUV conversion + data

def _fake_buv(root: Path) -> Path:
    from PIL import Image

    src = root / "raw"
    videos = {"train": ["benign/vA", "benign/x1282311c38f808f"], "val": ["malignant/vB"]}
    for split, name in (("train", "imagenet_vid_train_15frames.json"), ("val", "imagenet_vid_val.json")):
        images, anns = [], []
        for v in videos[split]:
            (src / "rawframes" / v).mkdir(parents=True, exist_ok=True)
            for f in range(2):
                Image.new("RGB", (100, 100), (f * 50, 0, 0)).save(src / "rawframes" / v / f"{f:06d}.png")
                images.append({"id": len(images) + 1, "file_name": f"{v}/{f:06d}.png",
                               "height": 100, "width": 100})
                # second frame of every video: box drawn right-to-left
                bbox = [10, 20, 30, 40] if f == 0 else [40, 20, -30, 40]
                cat = 1 if v.startswith("benign") else 2
                anns.append({"id": len(anns) + 1, "image_id": len(images), "category_id": cat,
                             "bbox": bbox})
        cats = [{"id": 1, "name": "benign"}, {"id": 2, "name": "malignant"}]
        (src / name).write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return src


def test_buv_conversion_fixes_boxes_and_drops_the_duplicate(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "buv", Path(__file__).resolve().parents[1] / "scripts/prepare_data/buv.py")
    buv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(buv)

    src, dst = _fake_buv(tmp_path), tmp_path / "buv"
    stats = buv.convert(src, dst)
    assert stats["train"] == {"frames": 2, "videos": 1, "boxes": 2, "dropped_frames": 2}
    recs = json.loads((dst / "val.json").read_text())
    assert recs[1]["boxes"] == [[10, 20, 40, 60]] and recs[1]["labels"] == [1]

    ds = DetectionFrames(str(dst), "val", image_size=32)
    s = ds[1]
    assert s["image"].shape == (3, 32, 32) and s["size"] == (100, 100)
    assert torch.allclose(s["boxes"], torch.tensor([[0.25, 0.4, 0.3, 0.4]]))
    batch = detection_collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 32, 32) and len(batch["targets"]) == 2


def test_hflip_mirrors_boxes(tmp_path):
    dst = tmp_path / "buv"
    (dst / "images").mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (100, 100)).save(dst / "images" / "a.png")
    (dst / "classes.json").write_text(json.dumps(["lesion"]))
    (dst / "train.json").write_text(json.dumps([{"file": "a.png", "video": "v", "height": 100,
                                                 "width": 100, "boxes": [[10, 20, 30, 60]],
                                                 "labels": [0]}]))
    ds = DetectionFrames(str(dst), "train", image_size=None, hflip=True)
    seen = {tuple(round(v, 3) for v in ds[0]["boxes"][0].tolist()) for _ in range(30)}
    assert seen == {(0.2, 0.4, 0.2, 0.4), (0.8, 0.4, 0.2, 0.4)}


# -------------------------------------------------------- [DET] queries on the toy MLLM

@pytest.mark.parametrize("overrides", [
    ["conditioner=none", "expert=none"],
    ["conditioner=specialist", "projector=cross_attn", "injection=pre_llm"],
    ["conditioner=specialist", "projector=cross_attn", "injection=post_llm"],
    ["conditioner=specialist", "projector=mlp", "injection=pre_llm"],  # global-pool fallback
])
def test_toy_det_forward_loss_and_trainable_params(overrides):
    torch.manual_seed(0)
    model = build_model(_cfg([*overrides, "mllm.num_classes=2"]))
    images = torch.rand(2, 3, 64, 64)
    out = model(images, task="det")
    assert out["logits"].shape == (2, 4, 2) and out["boxes"].shape == (2, 4, 4)
    targets = [_target([[0.3, 0.3, 0.2, 0.2]], [0]), _target([[0.6, 0.6, 0.1, 0.3]], [1])]
    loss = SetCriterion(2)(out["logits"], out["boxes"], targets)["loss"]
    loss.backward()
    assert model.mllm.det_queries.grad is not None and model.mllm.det_queries.grad.abs().sum() > 0
    if model.conditioner is not None:
        with torch.no_grad():
            model.conditioner.gate.logit.fill_(0.5)
        assert not torch.allclose(model(images, task="det")["logits"], out["logits"].detach())
