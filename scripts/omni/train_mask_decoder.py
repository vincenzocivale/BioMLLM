"""Segmentation readout of FROZEN Qwen3-Omni visual features on a us_bench dataset
(docs/omni_segmentation_design.md, docs/us_benchmark_suite.md).

For every structure k of the dataset, a small decoder (1.15 M params) is trained on frozen
`thinker.visual` maps with a box prompt, under two box conditions:

    gt    tight box of the GT mask of k        -> representation-quality oracle
    full  the whole image                      -> lower bound (no localisation)

and evaluated on the test split (and on test_nodup when the dataset has one). The decoder trained
with GT boxes is saved, together with the test features, so the END-TO-END condition (box
predicted by Qwen3-Omni grounding) is evaluated later with the same decoder:

    python scripts/omni/train_mask_decoder.py --dataset tn3k --eval-pred <grounding records.jsonl>

Only the vision encoder is loaded (official weights). `--device cpu` computes it in float32 (the
BF16 weights upcast; used so the GPU stays free for the 30 B model), `--device cuda` in BF16. All
conditions of one study must use the same device. `--steering` reads steered features (MODE 2).

    python scripts/omni/train_mask_decoder.py --dataset tn3k --device cpu --threads 24 \
        --out results/omni/seg/tn3k
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from biomllm.omni.features import VisualTaps
from biomllm.omni.loading import MODEL_ID, load_visual
from biomllm.omni.mask_decoder import OmniMaskDecoder

SUITE = Path("/raid/DATASETS/BioMLLMData/datasets/us_bench")
TAPS = {"blocks.8": 1152, "blocks.16": 1152, "blocks.26": 1152, "merged": 2048}


def mask_box(m: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class Features:
    def __init__(self, device: str, steering: str | None, steer_text: str | None) -> None:
        from transformers import Qwen3OmniMoeProcessor

        self.dev = torch.device(device)
        self.dtype = torch.float32 if self.dev.type == "cpu" else torch.bfloat16
        self.visual = load_visual(device=self.dev).to(self.dtype)
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
        self.ctx = None
        if steering:
            from biomllm.omni.steering import load_steering

            text_enc, self.steer, meta = load_steering(steering, SimpleNamespace(visual=self.visual), device=self.dev)
            text = Path(steer_text).read_text().strip() if steer_text else meta["steer_text"]
            with torch.no_grad():
                self.z, self.mask = text_enc([text])
            self.ctx = lambda: self.steer.condition(self.z, self.mask)  # noqa: E731

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> dict[str, torch.Tensor]:
        x = self.proc.image_processor(images=[img], return_tensors="pt")
        pv, thw = x["pixel_values"].to(self.dev, self.dtype), x["image_grid_thw"].to(self.dev)
        with VisualTaps(self.visual, blocks=(8, 16, 26)) as taps:
            if self.ctx:
                with self.ctx():
                    self.visual(pv, grid_thw=thw)
            else:
                self.visual(pv, grid_thw=thw)
        return {k: v.to("cpu", torch.float16) for k, v in taps.maps()[0].items() if k in TAPS}


def dice_iou(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    inter = (pred & gt).sum().item()
    s = pred.sum().item() + gt.sum().item()
    union = (pred | gt).sum().item()
    return (2 * inter / s if s else 1.0), (inter / union if union else 1.0)


def group_ci(rows: list[dict], key: str, n_boot: int = 2000) -> list[float]:
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r[key])
    groups = list(by)
    rng = random.Random(0)
    vals = sorted(np.mean([x for g in (rng.choice(groups) for _ in groups) for x in by[g]]) for _ in range(n_boot))
    return [float(vals[int(0.025 * n_boot)]), float(vals[int(0.975 * n_boot)])]


def norm_box(b, w, h, dev):
    return torch.tensor([[b[0] / w, b[1] / h, b[2] / w, b[3] / h]], device=dev).clamp(0, 1)


def train_decoder(train: list[dict], k: int, box: str, args, dev) -> OmniMaskDecoder:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dec = OmniMaskDecoder(TAPS).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    order = list(train)
    for _ in range(args.epochs):
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            loss = 0.0
            chunk = order[i:i + args.batch]
            for r in chunk:
                m = r["mask"] == k
                w, h = r["size"]
                b = r["boxes"][k] if box == "gt" else [0, 0, w, h]
                lg = dec({n: v.float().to(dev) for n, v in r["feats"].items()}, norm_box(b, w, h, dev))
                tgt = F.interpolate(m[None, None].float().to(dev), size=lg.shape[-2:], mode="area")
                p = lg.sigmoid()
                loss = loss + F.binary_cross_entropy_with_logits(lg, tgt) + \
                    1 - (2 * (p * tgt).sum() + 1) / (p.sum() + tgt.sum() + 1)
            opt.zero_grad()
            (loss / len(chunk)).backward()
            opt.step()
            sched.step()
    return dec.eval()


@torch.no_grad()
def evaluate(dec, test: list[dict], k: int, boxes: dict[str, list[float] | None] | str, dev) -> list[dict]:
    rows = []
    for r in test:
        w, h = r["size"]
        if boxes == "gt":
            b = r["boxes"][k]
        elif boxes == "full":
            b = [0, 0, w, h]
        else:
            b = boxes.get(r["id"]) or [0, 0, w, h]  # no parsed box: whole image
        lg = dec({n: v.float().to(dev) for n, v in r["feats"].items()}, norm_box(b, w, h, dev))
        pred = F.interpolate(lg, size=(h, w), mode="bilinear", align_corners=False)[0, 0].sigmoid().cpu() > 0.5
        d, j = dice_iou(pred, r["mask"] == k)
        rows.append({"id": r["id"], "group": r["group"], "dice": d, "iou": j})
    return rows


def summarize(rows: list[dict], nodup: set[str] | None) -> dict:
    out = {"n": len(rows), "dice": float(np.mean([r["dice"] for r in rows])),
           "iou": float(np.mean([r["iou"] for r in rows])), "dice_ci95": group_ci(rows, "dice")}
    if nodup is not None:
        sub = [r for r in rows if r["id"] in nodup]
        out["nodup"] = {"n": len(sub), "dice": float(np.mean([r["dice"] for r in sub])),
                        "iou": float(np.mean([r["iou"] for r in sub]))}
    return out


def load_split(root: Path, split: str, feats: Features | None, n_classes: int) -> list[dict]:
    rows = []
    for it in json.load(open(root / f"{split}.json")):
        img = Image.open(root / it["file"]).convert("RGB")
        m = torch.from_numpy(np.array(Image.open(root / it["mask"])))
        boxes = {k: mask_box((m == k).numpy()) for k in range(1, n_classes + 1)}
        if not any(boxes.values()):
            continue  # normal images: nothing to segment
        rows.append({"id": it["file"], "group": it["group"], "size": img.size, "mask": m, "boxes": boxes,
                     "feats": feats(img) if feats else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--suite", default=str(SUITE))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--steering", default=None)
    ap.add_argument("--steer-text", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-pred", default=None, help="grounding records.jsonl: evaluate saved GT-box decoders")
    ap.add_argument("--class-id", type=int, default=1, help="with --eval-pred: structure the records localise")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.suite) / args.dataset
    classes = json.load(open(root / "classes.json"))
    dev = torch.device(args.device)
    nodup_file = root / "test_nodup.json"
    nodup = {r["file"] for r in json.load(open(nodup_file))} if nodup_file.exists() else None

    if args.eval_pred:  # end-to-end: same decoder, Omni-predicted boxes
        k = args.class_id
        test = torch.load(out / "test_cache.pt", weights_only=False)
        dec = OmniMaskDecoder(TAPS).to(dev)
        dec.load_state_dict(torch.load(out / f"decoder_{k}_gt.pt"))
        pred = {}
        for l in open(args.eval_pred):
            r = json.loads(l)
            pred[r["id"]] = r["boxes_px"][0] if r.get("boxes_px") else None
        sub = [r for r in test if r["boxes"][k] and r["id"] in pred]
        res = summarize(evaluate(dec.eval(), sub, k, pred, dev), nodup)
        res.update({"class": classes[k - 1], "box": "pred", "records": args.eval_pred,
                    "n_missing_pred": sum(pred[r["id"]] is None for r in sub)})
        (out / f"result_{k}_pred.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return

    t0 = time.time()
    feats = Features(args.device, args.steering, args.steer_text)
    train = load_split(root, "train", feats, len(classes))
    test = load_split(root, "test", feats, len(classes))
    info = {"dataset": args.dataset, "classes": classes, "n_train": len(train), "n_test": len(test),
            "feature_device": args.device, "feature_dtype": str(feats.dtype), "feature_s": round(time.time() - t0, 1),
            "args": vars(args), "decoder_params": sum(p.numel() for p in OmniMaskDecoder(TAPS).parameters())}
    torch.save(test, out / "test_cache.pt")
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    results = {}
    for k in range(1, len(classes) + 1):
        tr = [r for r in train if r["boxes"][k]]
        te = [r for r in test if r["boxes"][k]]
        for box in ("gt", "full"):
            t1 = time.time()
            dec = train_decoder(tr, k, box, args, dev)
            if box == "gt":
                torch.save(dec.state_dict(), out / f"decoder_{k}_gt.pt")
            res = summarize(evaluate(dec, te, k, box, dev), nodup)
            res.update({"class": classes[k - 1], "box": box, "n_train": len(tr), "train_s": round(time.time() - t1, 1)})
            results[f"{classes[k - 1]}/{box}"] = res
            (out / f"result_{k}_{box}.json").write_text(json.dumps(res, indent=1))
            print(json.dumps(res), flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
