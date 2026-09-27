"""MODE 2: train ONLY the clinical visual-steering modules inside the frozen Qwen3-Omni.

Objective: the model's own grounding output. Next-token cross-entropy on the official answer
(```json [{"bbox_2d": [x1,y1,x2,y2] (0..1000), "label": ...}] ```) under the format-controlled
protocol used to evaluate every condition (official `enable_thinking=False` template + the answer
forced up to the first coordinate, `grounding.FORCED_PREFIX`; only what follows is supervised), back-propagated through the frozen Thinker (NF4 decoder, BF16
vision encoder) into the gated cross-attentions of `thinker.visual` and the text MLP. Qwen3-Omni
weights, RoBERTa and the tokenizer are frozen. With alpha=0 at init the model starts exactly at
vanilla Qwen3-Omni.

Conditions (same data, steps, seed; only the texts differ):
  --steer-text  guideline file fed to the steering (clinical, or the neutral capacity control)
  --prompt-guideline  also put the guideline in the LLM prompt (MODE 1 + 2)

Model selection never looks at BUV val (our test split): a fixed step budget, and 10% of the TRAIN
videos are held out only to monitor the loss curve.

    CUDA_VISIBLE_DEVICES=1 python scripts/omni/train_steering.py \
        --steer-text configs/omni/guidelines/breast_us_birads.txt --out results/omni/steer_guideline
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from biomllm.omni.grounding import FORCED_PREFIX, GROUNDING_TEMPLATE, build_messages
from biomllm.omni.loading import load_thinker
from biomllm.omni.steering import DEFAULT_LAYERS, build_steering, save_steering

BUV = Path("/raid/DATASETS/BioMLLMData/datasets/buv")
THINK_BLOCK = "<think>\n\n</think>\n\n"


def answer_text(boxes_px: list[list[float]], w: int, h: int, label: str) -> str:
    """GT boxes in the official output format (0..1000 ints, tab-indented JSON list, fenced).
    Coordinates are clipped to 0..1000 (some BUV boxes extend a few pixels outside the frame; a
    negative first coordinate would also re-tokenize the forced prefix "[" as "[-")."""
    rows = []
    clip = lambda v: min(1000, max(0, round(v)))  # noqa: E731
    for x1, y1, x2, y2 in boxes_px:
        b = [clip(x1 / w * 1000), clip(y1 / h * 1000), clip(x2 / w * 1000), clip(y2 / h * 1000)]
        rows.append("\t" + json.dumps({"bbox_2d": b, "label": label}))
    return "```json\n[\n" + ",\n".join(rows) + "\n]\n```"


def split_train(stride: int, holdout: float, seed: int):
    items = json.load(open(BUV / "train.json"))
    by_vid = defaultdict(list)
    for it in items:
        by_vid[it["video"]].append(it)
    vids = sorted(by_vid)
    random.Random(seed).shuffle(vids)
    n_hold = max(1, round(holdout * len(vids)))
    pick = lambda vs: [f for v in vs for f in sorted(by_vid[v], key=lambda x: x["file"])[::stride]]  # noqa: E731
    return pick(vids[n_hold:]), pick(vids[:n_hold])


class Batcher:
    def __init__(self, processor, prompt: str, label: str) -> None:
        self.p, self.prompt, self.label = processor, prompt, label
        tok = processor.tokenizer
        self.marker = tok(THINK_BLOCK, add_special_tokens=False)["input_ids"]
        self.prefix = tok(FORCED_PREFIX, add_special_tokens=False)["input_ids"]
        self.pad_id = tok.pad_token_id

    def __call__(self, items: list[dict], device) -> dict:
        texts, images = [], []
        for it in items:
            img = Image.open(BUV / "images" / it["file"]).convert("RGB")
            prompt = self.p.apply_chat_template(build_messages(img, self.prompt), add_generation_prompt=True,
                                                tokenize=False, enable_thinking=False)
            if not prompt.endswith(THINK_BLOCK):
                raise RuntimeError("chat template no longer ends with the empty think block")
            answer = answer_text(it["boxes"], img.width, img.height, self.label)
            if not answer.startswith(FORCED_PREFIX):
                raise RuntimeError("target answer does not start with the forced prefix")
            texts.append(prompt + answer + "<|im_end|>")
            images.append(img)
        enc = self.p(text=texts, images=images, return_tensors="pt", padding=True)
        ids = enc["input_ids"]
        labels = torch.full_like(ids, -100)
        m = len(self.marker)
        for b in range(ids.shape[0]):
            row = ids[b].tolist()
            start = max(k + m for k in range(len(row) - m + 1) if row[k:k + m] == self.marker)
            # the forced prefix is part of the prompt at evaluation: not supervised, and it must be
            # tokenized exactly as the evaluator's prompt ends
            if row[start:start + len(self.prefix)] != self.prefix:
                raise RuntimeError("forced prefix tokenized differently inside the target")
            start += len(self.prefix)
            labels[b, start:] = ids[b, start:]
        labels[enc["attention_mask"] == 0] = -100
        enc = {k: v.to(device) for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        enc["labels"] = labels.to(device)
        return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-text", required=True)
    ap.add_argument("--prompt-guideline", default=None)
    ap.add_argument("--target", default="breast lesion")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--text-source", default="roberta", choices=["roberta", "omni"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=2, help="gradient accumulation chunk (memory)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gate-lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    thinker, processor, info = load_thinker("nf4")
    layers = tuple(int(x) for x in args.layers.split(","))
    text_enc, steer = build_steering(thinker, layers, args.text_source, tokenizer=processor.tokenizer)
    steer_text = Path(args.steer_text).read_text().strip()
    prompt = GROUNDING_TEMPLATE.format(target=args.target)
    if args.prompt_guideline:
        prompt = Path(args.prompt_guideline).read_text().strip() + "\n\n" + prompt
    batcher = Batcher(processor, prompt, args.target)
    train, hold = split_train(args.frame_stride, 0.1, args.seed)
    hold = random.Random(1).sample(hold, min(64, len(hold)))

    gates = [p for n, p in steer.named_parameters() if n.endswith("alpha")]
    others = [p for n, p in steer.named_parameters() if not n.endswith("alpha")] + list(text_enc.proj.parameters())
    opt = torch.optim.AdamW([{"params": others, "lr": args.lr, "weight_decay": 0.01},
                             {"params": gates, "lr": args.gate_lr, "weight_decay": 0.0}])
    base = [g["lr"] for g in opt.param_groups]
    n_train = sum(p.numel() for p in others + gates)
    meta = {"layers": list(layers), "text_source": args.text_source, "text_model": "FacebookAI/roberta-large",
            "steer_text": steer_text, "prompt": prompt, "args": vars(args), "trainable_params": n_train,
            "n_train_frames": len(train), "n_holdout_frames": len(hold), "load": info}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("steer_text", "prompt", "load")}), flush=True)

    def loss_on(items):
        enc = batcher(items, dev)
        z, mask = text_enc([steer_text] * len(items))
        with steer.condition(z, mask):
            return thinker(**enc, use_cache=False).loss

    @torch.no_grad()
    def holdout_loss():
        steer.eval()
        ls = [loss_on(hold[i:i + args.micro_batch]).item() for i in range(0, len(hold), args.micro_batch)]
        steer.train()
        return sum(ls) / len(ls)

    log = open(out / "train_log.jsonl", "a")
    rec = {"step": 0, "holdout_loss": holdout_loss(), "gates": steer.gates()}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(rec, flush=True)
    order, pos = [], 0
    t0 = time.time()
    steer.train()
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(order):
            order, pos = random.sample(train, len(train)), 0
        items = order[pos:pos + args.batch]
        pos += args.batch
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        chunks = [items[i:i + args.micro_batch] for i in range(0, len(items), args.micro_batch)]
        for c in chunks:
            l = loss_on(c) * len(c) / len(items)
            l.backward()
            loss = loss + l.detach()
        gn = torch.nn.utils.clip_grad_norm_(others + gates, 1.0)
        opt.step()
        if step % 10 == 0:
            rec = {"step": step, "loss": loss.item(), "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"],
                   "s_per_step": round((time.time() - t0) / step, 2),
                   "gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            if step % args.eval_every == 0 or step == args.steps:
                rec["holdout_loss"] = holdout_loss()
                rec["gates"] = steer.gates()
                save_steering(out / "steering.pt", text_enc, steer, meta)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
    save_steering(out / "steering.pt", text_enc, steer, meta)


if __name__ == "__main__":
    main()
