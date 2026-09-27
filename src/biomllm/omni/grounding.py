"""Object grounding with the official Qwen3-Omni convention (cookbooks/object_grounding.ipynb):

    prompt   "Locate the object: <description>."   (image first, then text, single user turn)
    decoding greedy (`do_sample=False`), eos = <|im_end|> (151645)
    answer   ```json [{"bbox_2d": [x1, y1, x2, y2], "label": ...}, ...] ```
             coordinates normalised to 0..1000 of the ORIGINAL image width / height.

The Thinking model first writes a reasoning block terminated by `</think>`; only the text after
it is parsed. A truncated reasoning block (no `</think>` within the token budget) is recorded
as a parser failure, not silently parsed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers import LogitsProcessorList

GROUNDING_TEMPLATE = "Locate the object: {target}."
# Format-controlled protocol: empty think block (official enable_thinking=False) + the official
# answer format forced up to the first coordinate, so the model only writes the numbers. Used for
# every condition of the controlled matrix and as the steering training target, so that differences
# between conditions cannot come from answer formatting.
FORCED_PREFIX = '```json\n[\n\t{"bbox_2d": ['

THINK_END = "</think>"


@dataclass
class ParsedBoxes:
    boxes: list[list[float]] = field(default_factory=list)   # 0..1000 xyxy, model order
    labels: list[str] = field(default_factory=list)
    failure: str | None = None                                # None when parsing succeeded
    answer: str = ""                                          # text after </think>


def extract_json_from_string(text: str) -> str:
    """Verbatim logic of the official cookbook helper."""
    start_brace, start_bracket = text.find("{"), text.find("[")
    if start_brace == -1:
        start = start_bracket
    elif start_bracket == -1:
        start = start_brace
    else:
        start = min(start_brace, start_bracket)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end == -1:
        return text
    return text[start:end + 1]


_BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
                     r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def parse_grounding(text: str, thinking: bool = True) -> ParsedBoxes:
    if THINK_END in text:
        # also with enable_thinking=False: the Thinking checkpoint keeps reasoning after the
        # pre-filled empty block and closes it with its own </think> before answering
        answer = text.rsplit(THINK_END, 1)[1].strip()
    elif thinking:
        return ParsedBoxes(failure="no_think_end", answer="")
    else:
        answer = text.strip()
    out = ParsedBoxes(answer=answer)
    raw = extract_json_from_string(answer)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # tolerate truncated / slightly malformed JSON, but record it
        found = _BOX_RE.findall(answer)
        if not found:
            out.failure = "no_json" if raw == answer and "[" not in answer else "json_error"
            return out
        out.boxes = [[float(v) for v in b] for b in found]
        out.labels = [""] * len(out.boxes)
        out.failure = None
        return out
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        out.failure = "bad_json_type"
        return out
    for item in obj:
        if isinstance(item, dict) and isinstance(item.get("bbox_2d"), (list, tuple)) and len(item["bbox_2d"]) == 4:
            try:
                out.boxes.append([float(v) for v in item["bbox_2d"]])
            except (TypeError, ValueError):
                continue
            out.labels.append(str(item.get("label", "")))
    if not out.boxes:
        out.failure = "empty" if len(obj) == 0 else "no_bbox"
    return out


def to_pixels(box: list[float], width: int, height: int) -> list[float]:
    """0..1000 normalised xyxy -> original-image pixels (cookbook `draw_normalized_bounding_boxes`)."""
    return [box[0] / 1000 * width, box[1] / 1000 * height, box[2] / 1000 * width, box[3] / 1000 * height]


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_messages(image: Any, prompt: str, system: str | None = None) -> list[dict]:
    msgs = [] if system is None else [{"role": "system", "content": [{"type": "text", "text": system}]}]
    msgs.append({"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]})
    return msgs


class _GreedyLogprob:
    """Records log p(argmax token) at every step. Under greedy decoding the argmax IS the emitted
    token, so this equals the emitted token's log-probability without keeping `output_scores`
    (152k floats per step) for thousands of thinking tokens."""

    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.values.append(torch.log_softmax(scores[0].float(), -1).max().view(1).cpu())
        return scores


@torch.no_grad()
def generate(thinker, processor, messages: list[dict], images: list, max_new_tokens: int = 8192,
             device: str | torch.device = "cuda", enable_thinking: bool = True,
             answer_prefix: str | None = None) -> dict:
    """One greedy generation through the official processor + Thinker. Returns the raw text and,
    for every generated token, its log-probability (the confidence signal we can report)."""
    # enable_thinking=False is the official template switch: it pre-fills an empty
    # "<think>\n\n</think>\n\n" block, so the answer is generated without a reasoning trace
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         **({} if enable_thinking else {"enable_thinking": False}))
    if answer_prefix:
        text += answer_prefix
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    rec = _GreedyLogprob()
    out = thinker.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           eos_token_id=151645, use_audio_in_video=False,
                           logits_processor=LogitsProcessorList([rec]), return_dict_in_generate=True)
    n_in = inputs["input_ids"].shape[1]
    seq = out.sequences[0, n_in:]
    logprobs = torch.cat(rec.values)[:seq.numel()]
    tok = processor.tokenizer
    return {"text": (answer_prefix or "") + tok.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "n_prompt_tokens": n_in, "n_new_tokens": int(seq.numel()),
            "hit_max_tokens": bool(seq.numel() >= max_new_tokens and seq[-1].item() != 151645),
            "token_ids": seq.tolist(), "token_logprobs": logprobs.tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist()}


def answer_confidence(gen: dict, tokenizer) -> dict:
    """Confidence of the final answer only (tokens after </think>): mean log-prob over all answer
    tokens and over the numeric tokens of the coordinates."""
    ids, lps = gen["token_ids"], gen["token_logprobs"]
    pieces = [tokenizer.decode([i]) for i in ids]
    start = 0
    acc = ""
    for k, p in enumerate(pieces):
        acc += p
        if acc.endswith(THINK_END) or THINK_END in p:
            start = k + 1
    ans_lp = lps[start:]
    num_lp = [lp for p, lp in zip(pieces[start:], ans_lp) if p.strip().isdigit()]
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    return {"answer_mean_logprob": mean(ans_lp), "coord_mean_logprob": mean(num_lp),
            "coord_conf": float(torch.tensor(mean(num_lp)).exp()) if num_lp else float("nan"),
            "n_answer_tokens": len(ans_lp)}


def record(**kw) -> dict:
    return {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in kw.items()}
