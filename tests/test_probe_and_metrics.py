import math

import torch

from biomllm.data.datasets.segmentation import SegmentationFolder, SyntheticShapes
from biomllm.evaluation.metrics import BinarySegMetrics, MultiClassSegMetrics, box_iou, mask_to_box
from biomllm.models.experts.registry import build_expert
from biomllm.probing.linear_probe import evaluate_probe, extract_features, train_probe


def test_binary_metrics_known_values():
    pred = torch.zeros(4, 4, dtype=torch.bool)
    target = torch.zeros(4, 4, dtype=torch.bool)
    pred[:2, :2] = True    # 4 px
    target[:2, :4] = True  # 8 px, overlap 4
    m = BinarySegMetrics()
    m.update(pred, target)
    r = m.compute()
    assert math.isclose(r["dice"], 2 * 4 / 12)
    assert math.isclose(r["giou"], 4 / 8)
    assert math.isclose(r["ciou"], 4 / 8)
    assert r["acc_box50"] == 1.0  # box IoU = 0.5


def test_ciou_vs_giou_weighting():
    """cIoU is dominated by large targets, gIoU weighs every sample equally."""
    m = BinarySegMetrics()
    big = torch.ones(10, 10, dtype=torch.bool)
    small_t = torch.zeros(10, 10, dtype=torch.bool)
    small_t[0, 0] = True
    m.update(big, big)                                          # IoU 1, 100 px
    m.update(torch.zeros(10, 10, dtype=torch.bool), small_t)   # IoU 0, 1 px
    r = m.compute()
    assert math.isclose(r["giou"], 0.5)
    assert r["ciou"] > 0.99


def test_empty_prediction_and_target():
    m = BinarySegMetrics()
    empty = torch.zeros(3, 3, dtype=torch.bool)
    m.update(empty, empty)
    r = m.compute()
    assert r["dice"] == 1.0 and r["giou"] == 1.0 and math.isnan(r["acc_box50"])


def test_boxes():
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[1:3, 2:5] = True
    box = mask_to_box(mask)
    assert box.tolist() == [2, 1, 5, 3]
    assert box_iou(box, box) == 1.0
    assert mask_to_box(torch.zeros(2, 2, dtype=torch.bool)) is None


def test_multiclass_metrics():
    target = torch.tensor([[0, 1], [2, 2]])
    m = MultiClassSegMetrics(3)
    m.update(target.clone(), target)
    r = m.compute()
    assert r["dice"] == 1.0 and r["dice_c1"] == 1.0 and r["dice_c2"] == 1.0


def test_segmentation_folder(tmp_path):
    import numpy as np
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(3):
        Image.fromarray((np.random.rand(20, 30, 3) * 255).astype("uint8")).save(tmp_path / "images" / f"a{i}.png")
        mask = np.zeros((20, 30), dtype="uint8")
        mask[5:10, 5:15] = 255
        Image.fromarray(mask).save(tmp_path / "masks" / f"a{i}.png")
    (tmp_path / "train.txt").write_text("a0\na2\n")
    ds = SegmentationFolder(str(tmp_path), split_file="train.txt", image_size=16)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 16, 16) and item["mask"].shape == (16, 16)
    assert set(item["mask"].unique().tolist()) <= {0, 1}
    assert 0 <= item["image"].min() and item["image"].max() <= 1


def test_linear_probe_learns_synthetic_shapes():
    expert = build_expert("toy", dim=32, image_size=64, patch_size=8)
    tr_f, tr_m = extract_features(expert, SyntheticShapes(n=48, seed=0))
    va_f, va_m = extract_features(expert, SyntheticShapes(n=16, seed=1))
    assert tr_f.shape == (48, 32, 8, 8) and tr_m.shape == (48, 64, 64)
    probe = train_probe(tr_f, tr_m, num_classes=2, epochs=30, lr=1e-2)
    r = evaluate_probe(probe, va_f, va_m, num_classes=2)
    assert r["dice"] > 0.6, r  # bright ellipses are linearly separable even for a random conv


def test_prepare_montgomery(tmp_path):
    import importlib.util
    from pathlib import Path

    import numpy as np
    from PIL import Image

    spec = importlib.util.spec_from_file_location(
        "montgomery", Path(__file__).resolve().parents[1] / "scripts/prepare_data/montgomery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = tmp_path / "MontgomerySet"
    for d in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
        (src / d).mkdir(parents=True)
    for i in range(5):
        name = f"MCUCXR_{i:04d}_0.png"
        Image.fromarray(np.zeros((8, 8), "uint8")).save(src / "CXR_png" / name)
        left, right = np.zeros((8, 8), "uint8"), np.zeros((8, 8), "uint8")
        left[:, :2], right[:, 6:] = 255, 255
        Image.fromarray(left).save(src / "ManualMask/leftMask" / name)
        Image.fromarray(right).save(src / "ManualMask/rightMask" / name)
    n_train, n_val = mod.convert(src, tmp_path / "out")
    assert (n_train, n_val) == (4, 1)
    ds = SegmentationFolder(str(tmp_path / "out"), split_file="train.txt", image_size=None)
    assert len(ds) == 4 and ds[0]["mask"].sum() == 8 * 4  # both lungs merged


def test_bitmask_roundtrip(tmp_path):
    from biomllm.data.datasets.segmentation import load_bitmask, save_bitmask

    masks = torch.rand(13, 9, 7) > 0.6
    save_bitmask(masks.numpy(), tmp_path / "m.png")
    assert torch.equal(load_bitmask(tmp_path / "m.png", 13), masks)


def test_multilabel_metrics_positive_only_dice():
    from biomllm.evaluation.metrics import MultiLabelSegMetrics

    target = torch.zeros(2, 2, 4, 4, dtype=torch.bool)
    target[0, 0, :2, :2] = True           # class a present only in image 0
    pred = target.clone()
    pred[1, 1, 0, 0] = True               # false positive of class b on an image without it
    m = MultiLabelSegMetrics(["a", "b"])
    m.update(pred, target)
    r = m.compute()
    assert r["dice_pos/a"] == 1.0 and r["n_pos/a"] == 1
    assert r["n_pos/b"] == 0 and math.isnan(r["dice_pos/b"])
    assert r["ciou/b"] == 0.0             # FP penalised through cIoU
    assert r["dice"] == 1.0               # macro over classes with positives


def test_multilabel_probe_on_synthetic():
    expert = build_expert("toy", dim=32, image_size=64, patch_size=8)
    tr_f, tr_m = extract_features(expert, SyntheticShapes(n=48, seed=0, multilabel=True))
    va_f, va_m = extract_features(expert, SyntheticShapes(n=16, seed=1, multilabel=True))
    assert tr_f.dtype == torch.float16 and tr_m.shape == (48, 2, 64, 64)
    probe = train_probe(tr_f, tr_m, num_classes=2, epochs=30, lr=1e-2, multilabel=True)
    r = evaluate_probe(probe, va_f, va_m, num_classes=2, classes=["ellipse", "left_half"])
    assert r["dice_pos/ellipse"] > 0.6, r


def test_prepare_chestx_det(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from biomllm.data.datasets.segmentation import MultiLabelSegmentationFolder

    spec = importlib.util.spec_from_file_location(
        "chestx_det", Path(__file__).resolve().parents[1] / "scripts/prepare_data/chestx_det.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = tmp_path / "raw"
    (src / "train_data").mkdir(parents=True)
    (src / "test_data").mkdir()
    Image.fromarray(np.zeros((64, 64), "uint8")).save(src / "train_data" / "1.png")
    Image.fromarray(np.zeros((64, 64), "uint8")).save(src / "test_data" / "2.png")
    square = [[8, 8], [24, 8], [24, 24], [8, 24]]
    train = [{"file_name": "1.png", "syms": ["Nodule", "Mass"], "boxes": [], "polygons": [square, square]}]
    test = [{"file_name": "2.png", "syms": [], "boxes": [], "polygons": []},
            {"file_name": "missing.png", "syms": [], "boxes": [], "polygons": []}]
    (src / "ChestX_Det_train.json").write_text(json.dumps(train))
    (src / "ChestX_Det_test.json").write_text(json.dumps(test))

    counts = mod.convert(src, tmp_path / "out", max_size=32)
    assert counts == {"train": 1, "train_missing": 0, "val": 1, "val_missing": 1}
    ds = MultiLabelSegmentationFolder(str(tmp_path / "out"), mod.CLASSES, split_file="train.txt",
                                      image_size=32, mask_size=32)
    m = ds[0]["mask"]
    assert m.shape == (13, 32, 32)
    nodule, mass = mod.CLASSES.index("Nodule"), mod.CLASSES.index("Mass")
    assert torch.equal(m[nodule], m[mass]) and m[nodule].sum() > 0   # overlapping classes kept
    assert m.sum() == 2 * m[nodule].sum()                            # nothing else set
