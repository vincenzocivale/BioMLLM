"""Linear segmentation probes on the frozen LLM's hidden states at the image-token positions.

Does the expert signal injected into the visual tokens (injection=native) survive inside the
LLM, or does the LLM wash it out? For each layer, a linear probe (same as experiment 0,
scripts/probe_experts.py) is trained on the hidden states of the seg prefill at the image
positions, on ChestX-Det "any lesion" (the setup of scripts/train_seg_c0_vs_c3.py). Layer 0 is
the LLM input itself (V, or V' = V + alpha * P(F^S)).

    # C0 / C3 on task tokens: the image context is the native V
    python scripts/probe_llm_hidden.py mllm=qwen_vl condition=c0_none +run_name=probe_c0
    # a trained native run (its trainable.pt: conditioner + task parameters)
    python scripts/probe_llm_hidden.py mllm=qwen_vl condition=c3_rad_dino injection=native \\
        +checkpoint=outputs/.../trainable.pt +run_name=probe_c3_native

With pre_llm / post_llm injection the image positions never see the expert (the task tokens
come after them in the causal sequence), and with native_prepend neither (the extra tokens
come after the image), so only `native` changes these hidden states.
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg_c0_vs_c3 import CLASSES, collapse_binary  # noqa: E402

from biomllm.data.datasets.segmentation import MultiLabelSegmentationFolder  # noqa: E402
from biomllm.models.build import build_model, load_trainable  # noqa: E402
from biomllm.models.conditioning.conditioner import InjectionMode, InjectionPoint  # noqa: E402
from biomllm.probing.linear_probe import evaluate_probe, train_probe  # noqa: E402

log = logging.getLogger(__name__)


@torch.no_grad()
def extract(model, dataset, layers, device, batch_size, limit=None):
    """{layer: [N, D, h, w] fp16 on CPU}, masks [N, H, W] long."""
    cond = model.conditioner
    native = cond is not None and cond.injection_point is InjectionPoint.NATIVE
    feats = {layer: [] for layer in layers}
    masks, seen = [], 0
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collapse_binary, num_workers=4):
        images = batch["image"].to(device)
        visual = model.mllm.visual_features(images)
        if native:
            visual, _ = cond.condition_visual(visual, model.conditioning_features(images, visual, {}))
        for layer, fm in model.mllm.image_hidden_states(visual, layers).items():
            feats[layer].append(fm.as_image().to(torch.float16).cpu())
        masks.append(batch["mask"].long())
        seen += images.shape[0]
        if limit and seen >= limit:
            break
    return {k: torch.cat(v)[:limit] for k, v in feats.items()}, torch.cat(masks)[:limit]


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    layers = list(cfg.get("layers", [0, 4, 8, 12, 18, 24, 30, 36]))
    run_name = cfg.get("run_name", "probe_llm_hidden")

    model = build_model(cfg).to(device).eval()
    cond = model.conditioner
    if cond is not None and cond.injection_point is InjectionPoint.NATIVE and cond.mode is InjectionMode.PREPEND:
        raise ValueError("native_prepend leaves the image-position hidden states unchanged; "
                         "probe it as C0")
    if cfg.get("checkpoint"):
        log.info("loaded %d tensors from %s", load_trainable(model, cfg.checkpoint), cfg.checkpoint)

    root = f"{os.environ.get('BIOMLLM_DATA', 'data')}/chestx_det"
    kw = dict(image_size=cfg.mllm.image_size, mask_size=256)  # as in train_seg_c0_vs_c3.py
    train_ds = MultiLabelSegmentationFolder(root, CLASSES, split_file="train.txt", **kw)
    val_ds = MultiLabelSegmentationFolder(root, CLASSES, split_file="val.txt", **kw)
    bs = cfg.get("extract_batch_size", 8)
    train_f, train_m = extract(model, train_ds, layers, device, bs, cfg.get("n_train"))
    val_f, val_m = extract(model, val_ds, layers, device, bs)
    log.info("features: train %d, val %d images, layers %s", len(train_m), len(val_m), layers)
    del model
    torch.cuda.empty_cache()

    results = {}
    for layer in layers:
        probe = train_probe(train_f[layer], train_m, 2, epochs=cfg.get("probe_epochs", 20),
                            lr=cfg.get("probe_lr", 1e-3), batch_size=16, device=device, seed=cfg.seed)
        results[layer] = evaluate_probe(probe, val_f[layer], val_m, 2, device=device)
        log.info("layer %d: %s", layer, results[layer])

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"run_name": run_name, "checkpoint": cfg.get("checkpoint"), "layers": results}, indent=2))
    log.info("dice per layer: %s", {k: round(v["dice"], 3) for k, v in results.items()})


if __name__ == "__main__":
    main()
