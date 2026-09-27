"""Frozen Qwen3-Omni grounding (MODE 0: no LoRA, no steering, no fine-tuning).

  --official   reproduce the two cookbook examples (birds, motorcyclist) and save raw outputs.
  --guideline  MODE 1 (ordinary clinical prompting): guideline text before the official prompt.
  --buv        vanilla ultrasound grounding on BUV frames; one JSONL record per image with the raw
               generated text, parsed boxes, parser failure, confidence, GT box and IoU.

    CUDA_VISIBLE_DEVICES=1 python scripts/omni/run_grounding.py --official --out results/omni/official
    CUDA_VISIBLE_DEVICES=1 python scripts/omni/run_grounding.py --buv --split val --per-video 3 \
        --out results/omni/buv_vanilla
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from PIL import Image

from biomllm.omni.grounding import (FORCED_PREFIX, GROUNDING_TEMPLATE, answer_confidence, box_iou,
                                    build_messages,
                                    generate, parse_grounding, to_pixels)
from biomllm.omni.loading import load_thinker

OFFICIAL = [
    ("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/grounding1.jpeg", "bird"),
    ("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/grounding2.jpg",
     "A person riding a motorcycle while wearing a helmet"),
]
BUV_ROOT = Path("/raid/DATASETS/BioMLLMData/datasets/buv")


def buv_subset(split: str, per_video: int, limit: int | None, seed: int) -> list[dict]:
    """`per_video` evenly spaced frames per cine loop (frames of one loop are near-duplicates)."""
    items = json.load(open(BUV_ROOT / f"{split}.json"))
    by_video = defaultdict(list)
    for it in items:
        by_video[it["video"]].append(it)
    out = []
    for vid in sorted(by_video):
        frames = sorted(by_video[vid], key=lambda x: x["file"])
        k = min(per_video, len(frames))
        idx = [round(i * (len(frames) - 1) / max(k - 1, 1)) for i in range(k)] if k > 1 else [len(frames) // 2]
        out += [frames[i] for i in sorted(set(idx))]
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--buv", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="us_bench dataset folder (docs/us_benchmark_suite.md): every image of --split that "
                         "contains --class-id, GT = the boxes of that class")
    ap.add_argument("--class-id", type=int, default=1, help="with --data-root: 1-based structure id")
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-video", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target", default="breast lesion")
    ap.add_argument("--prompt", default=None, help="full prompt; overrides the official template")
    ap.add_argument("--guideline", default=None,
                    help="MODE 1: text file prepended to the grounding prompt (same checkpoint, no steering)")
    ap.add_argument("--precision", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true",
                    help="official enable_thinking=False template (empty <think></think> pre-filled)")
    ap.add_argument("--forced-json", action="store_true",
                    help="format-controlled protocol: empty think block + official JSON prefix forced up to "
                         "the first coordinate (implies --no-thinking)")
    ap.add_argument("--steering", default=None, help="MODE 2: steering.pt from train_steering.py")
    ap.add_argument("--steer-text", default=None,
                    help="text file conditioning the steering (default: the text it was trained with); "
                         "a different file tests text dependence (wrong-guideline control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.forced_json:
        args.no_thinking = True
        args.max_new_tokens = min(args.max_new_tokens, 96)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    thinker, processor, info = load_thinker(args.precision)
    info["load_s"] = round(time.time() - t0, 1)
    info["args"] = vars(args)
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)
    tok = processor.tokenizer

    steer_ctx = None
    if args.steering:
        from biomllm.omni.steering import load_steering

        text_enc, steer, meta = load_steering(args.steering, thinker, processor.tokenizer)
        steer_text = Path(args.steer_text).read_text().strip() if args.steer_text else meta["steer_text"]
        with torch.no_grad():
            z, mask = text_enc([steer_text])
        steer_ctx = lambda: steer.condition(z, mask)  # noqa: E731
        info["steering"] = {"ckpt": args.steering, "gates": steer.gates(), "steer_text": steer_text,
                            "trained_prompt": meta["prompt"]}
        (out / "run_info.json").write_text(json.dumps(info, indent=1))

    jobs = []
    if args.official:
        for url, target in OFFICIAL:
            img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
            jobs.append({"id": url.rsplit("/", 1)[1], "image": img, "prompt": GROUNDING_TEMPLATE.format(target=target),
                         "gt": None})
    if args.buv:
        prompt = args.prompt or GROUNDING_TEMPLATE.format(target=args.target)
        if args.guideline:
            prompt = Path(args.guideline).read_text().strip() + "\n\n" + prompt
        for it in buv_subset(args.split, args.per_video, args.limit, args.seed):
            jobs.append({"id": it["file"], "path": str(BUV_ROOT / "images" / it["file"]), "prompt": prompt,
                         "gt": it["boxes"], "labels": it["labels"], "video": it["video"]})

    if args.data_root:
        root = Path(args.data_root)
        prompt = args.prompt or GROUNDING_TEMPLATE.format(target=args.target)
        if args.guideline:
            prompt = Path(args.guideline).read_text().strip() + "\n\n" + prompt
        items = json.load(open(root / f"{args.split}.json"))
        k = args.class_id - 1
        for it in items[:args.limit] if args.limit else items:
            gt = [b for b, l in zip(it["boxes"], it["labels"]) if l == k]
            if gt:
                jobs.append({"id": it["file"], "path": str(root / it["file"]), "prompt": prompt, "gt": gt,
                             "labels": [k] * len(gt), "video": it["group"]})

    done = set()
    rec_path = out / "records.jsonl"
    if rec_path.exists():
        done = {json.loads(l)["id"] for l in rec_path.open()}
    with rec_path.open("a") as f:
        for j, job in enumerate(jobs):
            if job["id"] in done:
                continue
            img = job.get("image") or Image.open(job["path"]).convert("RGB")
            w, h = img.size
            t = time.time()
            with (steer_ctx() if steer_ctx else nullcontext()):
                gen = generate(thinker, processor, build_messages(img, job["prompt"]), [img],
                               max_new_tokens=args.max_new_tokens, enable_thinking=not args.no_thinking,
                               answer_prefix=FORCED_PREFIX if args.forced_json else None)
            parsed = parse_grounding(gen["text"], thinking=not args.no_thinking)
            pix = [to_pixels(b, w, h) for b in parsed.boxes]
            rec = {"id": job["id"], "video": job.get("video"), "prompt": job["prompt"], "width": w, "height": h,
                   "thinking": not args.no_thinking, "forced_json": args.forced_json,
                   "raw_text": gen["text"], "answer": parsed.answer, "failure": parsed.failure,
                   "boxes_norm1000": parsed.boxes, "boxes_px": pix, "labels_pred": parsed.labels,
                   "n_new_tokens": gen["n_new_tokens"], "gen_token_ids": gen["token_ids"],
                   "gen_token_logprobs": gen["token_logprobs"], "hit_max_tokens": gen["hit_max_tokens"],
                   "image_grid_thw": gen["image_grid_thw"], "seconds": round(time.time() - t, 1),
                   **answer_confidence(gen, tok)}
            if job["gt"] is not None:
                gt = job["gt"]
                rec["gt_boxes_px"] = gt
                rec["gt_labels"] = job["labels"]
                rec["iou_first"] = box_iou(pix[0], gt[0]) if pix else 0.0
                rec["iou_best"] = max((box_iou(p, g) for p in pix for g in gt), default=0.0)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{j + 1}/{len(jobs)}] {job['id']} fail={parsed.failure} boxes={parsed.boxes[:2]} "
                  f"iou={rec.get('iou_first')} tokens={gen['n_new_tokens']} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
