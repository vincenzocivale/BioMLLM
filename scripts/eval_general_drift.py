"""H2 check: does a modality package change the MLLM's general abilities?

On MMStar (1500 multiple-choice questions on general images, answer = option letter), the
same trained model is scored with the package active and inactive (alpha scale 0, which is
exactly the frozen base model). Only `injection=native` / `native_prepend` can change this
path -- the other injection points only touch task tokens, which general chat never uses.

    python scripts/eval_general_drift.py mllm=qwen_vl condition=c3_rad_dino injection=native \\
        +checkpoint=outputs/.../trainable.pt +run_name=drift_c3_native

Reports accuracy on / off, the agreement of the predicted letters and the KL(off || on) of
the full next-token distribution at the answer position, overall and per category. Images
are resized to the adapter's square input (mllm.image_size), for both settings alike.
"""

import io
import json
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from PIL import Image

from biomllm.models.build import build_model, load_trainable
from biomllm.models.conditioning.conditioner import InjectionPoint

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."


def load_image(data: bytes) -> torch.Tensor:
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)[None]


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    run_name = cfg.get("run_name", "drift")
    model = build_model(cfg).to(device).eval()
    if cfg.get("checkpoint"):
        log.info("loaded %d tensors from %s", load_trainable(model, cfg.checkpoint), cfg.checkpoint)
    cond = model.conditioner
    if cond is None or cond.injection_point is not InjectionPoint.NATIVE:
        log.warning("this package never touches the general chat path: on == off by construction")

    df = pd.read_parquet(hf_hub_download("Lin-Chen/MMStar", "mmstar.parquet", repo_type="dataset"))
    if cfg.get("limit"):
        df = df.iloc[: cfg.limit]
    tok = model.mllm.tokenizer
    letter_ids = [tok.convert_tokens_to_ids(x) for x in LETTERS]

    rows = []
    with torch.no_grad():
        for _, item in df.iterrows():
            images = load_image(item["image"]).to(device)
            visual = model.mllm.visual_features(images)
            question = item["question"] + INSTRUCTION
            off = model.mllm.chat_next_token_logits(visual, question)
            on = off
            if cond is not None and cond.injection_point is InjectionPoint.NATIVE:
                feats = model.conditioning_features(images, visual, {})
                visual_on, extra = cond.condition_visual(visual, feats)
                on = model.mllm.chat_next_token_logits(visual_on, question, extra)
            kl = F.kl_div(on.log_softmax(-1), off.log_softmax(-1), log_target=True, reduction="sum").item()
            pred_off = LETTERS[int(off[letter_ids].argmax())]
            pred_on = LETTERS[int(on[letter_ids].argmax())]
            rows.append({"category": item["category"], "answer": item["answer"],
                         "off": pred_off, "on": pred_on, "kl": kl})

    def summarise(rs):
        n = len(rs)
        return {"n": n,
                "acc_off": sum(r["off"] == r["answer"] for r in rs) / n,
                "acc_on": sum(r["on"] == r["answer"] for r in rs) / n,
                "agreement": sum(r["off"] == r["on"] for r in rs) / n,
                "kl_off_on": sum(r["kl"] for r in rs) / n}

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    result = {"overall": summarise(rows), **{c: summarise(rs) for c, rs in sorted(by_cat.items())}}
    for k, v in result.items():
        log.info("%s: %s", k, {m: round(x, 4) if isinstance(x, float) else x for m, x in v.items()})

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"run_name": run_name, "checkpoint": cfg.get("checkpoint"), "result": result}, indent=2))


if __name__ == "__main__":
    main()
