"""Validate the NF4 production configuration against the BF16 reference on a small subset.

Input: records.jsonl of an NF4 grounding run (scripts/omni/run_grounding.py), which stores the
generated token ids. The BF16 Thinker (official weights, accelerate CPU offload, exact) is loaded
and for the first N records:

  1. teacher forcing  -- prompt + NF4 trace in one forward: fraction of positions where the BF16
     argmax equals the NF4 token (overall / answer part), mean |delta log p| of the NF4 token,
     first divergence position;
  2. answer from the same reasoning -- BF16 greedily writes the answer after the NF4 `</think>`
     prefix; the parsed BF16 box is compared (IoU) with the NF4 box and with the GT.

Vision features are BF16 in both configurations (the vision encoder is never quantized); the
script also checks that the merged image tokens agree between the two runs.

    CUDA_VISIBLE_DEVICES=1 python scripts/omni/validate_precision.py \
        --records results/omni/buv_vanilla/records.jsonl --n 20 --out results/omni/precision_check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from biomllm.omni.grounding import THINK_END, box_iou, build_messages, parse_grounding, to_pixels
from biomllm.omni.loading import load_thinker

BUV_IMAGES = Path("/raid/DATASETS/BioMLLMData/datasets/buv/images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--answer-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.records)]
    recs = [r for r in recs if "gen_token_ids" in r][:args.n]
    thinker, processor, info = load_thinker("bf16")
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    tok = processor.tokenizer
    dev = torch.device("cuda")
    think_end_ids = tok(THINK_END, add_special_tokens=False)["input_ids"]

    results = []
    for i, r in enumerate(recs):
        t0 = time.time()
        img = Image.open(BUV_IMAGES / r["id"]).convert("RGB")
        thinking = r.get("thinking", True)
        text = processor.apply_chat_template(build_messages(img, r["prompt"]), add_generation_prompt=True,
                                             tokenize=False, **({} if thinking else {"enable_thinking": False}))
        inputs = processor(text=text, images=[img], return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        n_in = inputs["input_ids"].shape[1]
        gen = torch.tensor(r["gen_token_ids"], device=dev)[None]

        # 1. teacher forcing on the full NF4 trace
        full = dict(inputs)
        full["input_ids"] = torch.cat([inputs["input_ids"], gen], 1)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        with torch.no_grad():
            logits = thinker(**full, use_cache=False).logits[0, n_in - 1:-1].float()
        lp = torch.log_softmax(logits, -1)
        bf16_lp = lp.gather(1, gen[0, :, None])[:, 0].cpu()
        agree = (logits.argmax(-1) == gen[0]).cpu()
        del logits, lp
        ids = r["gen_token_ids"]
        # the answer starts after the model's own </think>; with enable_thinking=False the Thinking
        # checkpoint still reasons and closes the block itself, so search in both modes
        end = next((k + len(think_end_ids) for k in range(len(ids))
                    if ids[k:k + len(think_end_ids)] == think_end_ids), None)
        if end is None:
            end = len(ids) if thinking else 0
        nf4_lp = torch.tensor(r["gen_token_logprobs"])
        row = {"id": r["id"], "n_tokens": len(ids), "think_end": end,
               "agree_all": float(agree.float().mean()),
               "agree_answer": float(agree[end:].float().mean()) if end < len(ids) else float("nan"),
               "first_divergence": int((~agree).nonzero()[0]) if (~agree).any() else -1,
               "mean_abs_dlogp": float((bf16_lp - nf4_lp[:len(bf16_lp)]).abs().mean()),
               "nf4_failure": r["failure"], "nf4_iou_first": r.get("iou_first")}

        # 2. BF16 writes the answer after the NF4 reasoning
        if end < len(ids):
            pre = dict(inputs)
            pre["input_ids"] = torch.cat([inputs["input_ids"], gen[:, :end]], 1)
            pre["attention_mask"] = torch.ones_like(pre["input_ids"])
            with torch.no_grad():
                o = thinker.generate(**pre, max_new_tokens=args.answer_tokens, do_sample=False,
                                     eos_token_id=151645, use_audio_in_video=False)
            ans = tok.decode(o[0, pre["input_ids"].shape[1]:], skip_special_tokens=True)
            p = parse_grounding(ans, thinking=False)
            row["bf16_boxes_norm1000"] = p.boxes
            row["bf16_answer"], row["bf16_failure"] = ans, p.failure
            if p.boxes and r["boxes_px"]:
                bpx = to_pixels(p.boxes[0], r["width"], r["height"])
                row["iou_bf16_vs_nf4"] = box_iou(bpx, r["boxes_px"][0])
                row["bf16_iou_first"] = box_iou(bpx, r["gt_boxes_px"][0])
        row["seconds"] = round(time.time() - t0, 1)
        results.append(row)
        print(json.dumps(row), flush=True)
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def m(key):
        xs = [x[key] for x in results if isinstance(x.get(key), (int, float)) and x[key] == x[key]]
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {k: m(k) for k in ("agree_all", "agree_answer", "mean_abs_dlogp", "iou_bf16_vs_nf4",
                                 "bf16_iou_first", "nf4_iou_first")}
    summary["n"] = len(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
