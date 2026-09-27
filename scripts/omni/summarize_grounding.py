"""Summarise grounding records (scripts/omni/run_grounding.py) into the detection metrics.

Frames of one cine loop are strongly correlated, so 95% CIs come from a cluster bootstrap over
videos. Parser failures count as misses (IoU 0), never dropped. COCO mAP uses the parsed boxes
scored by the coordinate-token confidence (the only confidence the text interface provides).

    python scripts/omni/summarize_grounding.py results/omni/buv_vanilla [results/omni/buv_guideline ...]
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from biomllm.omni.grounding import box_iou, parse_grounding, to_pixels

CLASSES = ["benign", "malignant"]
# secondary, lenient reading: any [x1, y1, x2, y2] quadruple in the answer (0..1000), used only to
# separate FORMAT failures (answer not in the official JSON) from LOCALISATION failures
_QUAD = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def lenient_iou(r: dict) -> float:
    if r["failure"] is None:
        return r.get("iou_first", 0.0)
    quads = [[float(v) for v in q] for q in _QUAD.findall(r.get("answer") or "")]
    quads = [q for q in quads if q[2] > q[0] and q[3] > q[1] and max(q) <= 1000]
    if not quads:
        return 0.0
    return box_iou(to_pixels(quads[0], r["width"], r["height"]), r["gt_boxes_px"][0])


def load(run: Path) -> list[dict]:
    """Records re-parsed from `raw_text` with the current parser, so every run is scored the same
    way whatever parser version was installed when it was generated."""
    recs = [json.loads(l) for l in open(run / "records.jsonl")]
    for r in recs:
        p = parse_grounding(r["raw_text"], thinking=r.get("thinking", True))
        r["answer"], r["failure"], r["boxes_norm1000"] = p.answer, p.failure, p.boxes
        r["boxes_px"] = [to_pixels(b, r["width"], r["height"]) for b in p.boxes]
        if "gt_boxes_px" in r:
            pix, gt = r["boxes_px"], r["gt_boxes_px"]
            r["iou_first"] = box_iou(pix[0], gt[0]) if pix else 0.0
            r["iou_best"] = max((box_iou(a, b) for a in pix for b in gt), default=0.0)
    return recs


def point(recs: list[dict]) -> dict:
    n = len(recs)
    iou = [r.get("iou_first", 0.0) for r in recs]
    return {"n": n, "parse_ok": sum(r["failure"] is None for r in recs) / n,
            "miou_first": sum(iou) / n, "miou_best": sum(r.get("iou_best", 0.0) for r in recs) / n,
            "acc@0.5": sum(i >= 0.5 for i in iou) / n, "acc@0.3": sum(i >= 0.3 for i in iou) / n,
            "lenient_miou_first": sum(lenient_iou(r) for r in recs) / n,
            "lenient_acc@0.5": sum(lenient_iou(r) >= 0.5 for r in recs) / n}


def bootstrap(recs: list[dict], key: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    by_vid = defaultdict(list)
    for r in recs:
        by_vid[r.get("video") or r["id"]].append(r)
    vids = list(by_vid)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for v in (rng.choice(vids) for _ in vids) for r in by_vid[v]]
        vals.append(point(sample)[key])
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def coco(recs: list[dict]) -> dict:
    from biomllm.evaluation.detection import CocoDetectionEvaluator

    ev = CocoDetectionEvaluator(["lesion"])  # class-agnostic: the prompt asks for "breast lesion"
    for r in recs:
        w, h = r["width"], r["height"]
        gt = torch.tensor(r["gt_boxes_px"], dtype=torch.float) / torch.tensor([w, h, w, h])
        gt_cxcywh = torch.cat([(gt[:, :2] + gt[:, 2:]) / 2, gt[:, 2:] - gt[:, :2]], 1)
        pb = torch.tensor(r["boxes_px"], dtype=torch.float).reshape(-1, 4) / torch.tensor([w, h, w, h])
        conf = r.get("coord_conf")
        conf = 0.5 if conf is None or conf != conf else conf
        ev.update([{"boxes": pb.clamp(0, 1), "labels": torch.zeros(len(pb), dtype=torch.long),
                    "scores": torch.full((len(pb),), conf)}],
                  [{"boxes": gt_cxcywh, "labels": torch.zeros(len(gt), dtype=torch.long)}], [(h, w)])
    out = ev.compute()
    return {k: out[k] for k in ("map", "map50", "map75")}


def trivial_baselines(recs: list[dict]) -> dict:
    """IoU of boxes that use no image content: the full image, and the mean normalised GT box of
    the evaluated frames (an oracle prior, optimistic). A model is only localising if it beats both."""
    norm = [[g[0] / r["width"], g[1] / r["height"], g[2] / r["width"], g[3] / r["height"]]
            for r in recs for g in r["gt_boxes_px"][:1]]
    mean = [sum(v[i] for v in norm) / len(norm) for i in range(4)]
    out = {}
    for name, box in (("full_image", [0, 0, 1, 1]), ("mean_gt_box", mean)):
        ious = [box_iou([box[0] * r["width"], box[1] * r["height"], box[2] * r["width"], box[3] * r["height"]],
                        r["gt_boxes_px"][0]) for r in recs]
        out[f"{name}_miou"] = sum(ious) / len(ious)
        out[f"{name}_acc@0.5"] = sum(i >= 0.5 for i in ious) / len(ious)
    return out


def summarize(run: Path) -> dict:
    recs = [r for r in load(run) if "gt_boxes_px" in r]
    s = point(recs)
    s["trivial"] = trivial_baselines(recs)
    s["pred_box_area_frac"] = sum(((b[2] - b[0]) * (b[3] - b[1]) / (r["width"] * r["height"])) if r["boxes_px"] else 0
                                  for r in recs for b in r["boxes_px"][:1]) / len(recs)
    for k in ("miou_first", "acc@0.5", "parse_ok"):
        s[f"{k}_ci95"] = bootstrap(recs, k)
    s["failures"] = dict(Counter(r["failure"] for r in recs if r["failure"]))
    s["n_boxes_pred"] = dict(Counter(len(r["boxes_px"]) for r in recs))
    s["mean_new_tokens"] = sum(r["n_new_tokens"] for r in recs) / len(recs)
    s["hit_max_tokens"] = sum(r["hit_max_tokens"] for r in recs)
    s["mean_seconds"] = sum(r["seconds"] for r in recs) / len(recs)
    s["n_videos"] = len({r["video"] for r in recs})
    for c in CLASSES:
        sub = [r for r in recs if r["video"].startswith(c)]
        if sub:
            s[f"miou_first/{c}"] = point(sub)["miou_first"]
    s.update(coco(recs))
    return s


def table(rows: dict) -> str:
    """Markdown matrix: strict (official JSON) and lenient scores, 95% video-bootstrap CI."""
    head = ("| run | n | parse ok | mIoU (strict) [95% CI] | Acc@0.5 (strict) [95% CI] | mIoU (lenient) | "
            "Acc@0.5 (lenient) | AP50 | tokens | s/img |")
    lines = [head, "|" + "|".join(["---"] * 10) + "|"]
    for run, r in rows.items():
        name = Path(run).name
        if r.get("missing"):
            lines.append(f"| {name} | – | missing | | | | | | | |")
            continue
        ci = lambda k: f"[{r[k + '_ci95'][0]:.3f}, {r[k + '_ci95'][1]:.3f}]"  # noqa: E731
        lines.append(f"| {name} | {r['n']} | {r['parse_ok']:.2f} | {r['miou_first']:.3f} {ci('miou_first')} | "
                     f"{r['acc@0.5']:.3f} {ci('acc@0.5')} | {r['lenient_miou_first']:.3f} | "
                     f"{r['lenient_acc@0.5']:.3f} | {r['map50']:.3f} | {r['mean_new_tokens']:.0f} | "
                     f"{r['mean_seconds']:.1f} |")
    first = next((r for r in rows.values() if not r.get("missing")), None)
    if first:
        t = first["trivial"]
        lines.append(f"| *full-image box* | | | {t['full_image_miou']:.3f} | {t['full_image_acc@0.5']:.3f} | | | | | |")
        lines.append(f"| *mean GT box (oracle prior)* | | | {t['mean_gt_box_miou']:.3f} | {t['mean_gt_box_acc@0.5']:.3f} "
                     "| | | | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = {}
    for arg in sys.argv[1:]:
        if not (Path(arg) / "records.jsonl").exists():
            rows[arg] = {"missing": True}
            continue
        rows[arg] = summarize(Path(arg))
        (Path(arg) / "summary.json").write_text(json.dumps(rows[arg], indent=1))
    print(json.dumps(rows, indent=1))
    print()
    print(table(rows))
