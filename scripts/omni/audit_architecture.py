"""Phase 0 audit of the official HF Qwen3-Omni implementation (no assumptions from Qwen3-VL).

1. Instantiates the full model on the meta device and records the module tree + parameter
   counts per sub-module.
2. Runs the official processor + chat template on a real BUV frame and records every tensor
   the Thinker receives (input_ids, pixel_values, image_grid_thw, 3D/4D position ids).
3. Loads ONLY `thinker.visual` with the official weights in BF16 on the GPU and records the
   shape of every intermediate (patch_embed, each block, deepstack mergers, final merger).

    CUDA_VISIBLE_DEVICES=1 python scripts/omni/audit_architecture.py --out results/omni/audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


def module_tree(model: torch.nn.Module, max_depth: int = 4) -> list[dict]:
    """Named modules up to `max_depth`, collapsing numbered repeats to their first element."""
    rows = []
    for name, mod in model.named_modules():
        parts = name.split(".") if name else []
        if len(parts) > max_depth:
            continue
        if any(p.isdigit() and p != "0" for p in parts):
            continue
        n = sum(p.numel() for p in mod.parameters())
        direct = {k: list(v.shape) for k, v in mod.named_parameters(recurse=False)}
        rows.append({"path": name or "<root>", "type": type(mod).__name__, "params": n,
                     "own_params": direct})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/raid/DATASETS/BioMLLMData/datasets/buv/images/"
                                       "malignant/2c12ff6464cfb1ff/000000.png")
    ap.add_argument("--out", default="results/omni/audit.json")
    args = ap.parse_args()

    from accelerate import init_empty_weights
    from PIL import Image
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    from biomllm.omni.layout import (block_tokens_to_grid, merged_tokens_to_grid,
                                     pixel_patches_to_image)

    report: dict = OrderedDict()
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    tc = cfg.thinker_config
    report["config"] = {
        "architectures": cfg.architectures, "enable_audio_output": cfg.enable_audio_output,
        "vision": tc.vision_config.to_dict(), "text": {k: v for k, v in tc.text_config.to_dict().items()
                                                        if not isinstance(v, dict) or k == "rope_scaling"},
        "image_token_id": tc.image_token_id, "video_token_id": tc.video_token_id,
        "audio_token_id": tc.audio_token_id, "vision_start_token_id": getattr(tc, "vision_start_token_id", None),
        "position_id_per_seconds": tc.position_id_per_seconds,
    }

    # 1. meta-device graph
    with init_empty_weights():
        full = Qwen3OmniMoeForConditionalGeneration(cfg)
    report["top_level"] = {n: {"type": type(m).__name__, "params": sum(p.numel() for p in m.parameters())}
                           for n, m in full.named_children()}
    report["thinker_children"] = {n: {"type": type(m).__name__,
                                      "params": sum(p.numel() for p in m.parameters())}
                                  for n, m in full.thinker.named_children()}
    report["thinker_tree"] = module_tree(full.thinker, max_depth=5)

    # 2. official processor + chat template on a real BUV frame
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": args.image},
                                             {"type": "text", "text": "Locate the object: breast lesion."}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, images=[image], return_tensors="pt", padding=True)
    ids = inputs["input_ids"]
    thw = inputs["image_grid_thw"][0].tolist()
    report["processor"] = {
        "chat_text": text, "image_size_in": list(image.size),
        "keys": {k: list(v.shape) for k, v in inputs.items() if torch.is_tensor(v)},
        "image_grid_thw": thw, "n_image_tokens": int((ids == tc.image_token_id).sum()),
        "pixel_values_dtype": str(inputs["pixel_values"].dtype),
        "pixel_range": [float(inputs["pixel_values"].min()), float(inputs["pixel_values"].max())],
    }
    # token-order check: un-patchify the processor output and compare with a plain resize
    vc = tc.vision_config
    rec = pixel_patches_to_image(inputs["pixel_values"], thw, vc.patch_size, vc.spatial_merge_size,
                                 vc.temporal_patch_size)
    ref = torch.from_numpy(__import__("numpy").asarray(
        image.resize((rec.shape[-1], rec.shape[-2]), Image.BICUBIC))).permute(2, 0, 1).float() / 255
    rec01 = rec[0] * 0.5 + 0.5
    report["processor"]["unpatchify_vs_resize_mae"] = float((rec01 - ref).abs().mean())
    report["processor"]["frames_identical_over_temporal_patch"] = bool(torch.equal(rec[0], rec[1]))

    position_ids, rope_deltas = full.thinker.get_rope_index(
        ids, image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
    img_pos = position_ids[:, 0, (ids[0] == tc.image_token_id)]
    report["positions"] = {"position_ids_shape": list(position_ids.shape),
                           "rope_deltas": rope_deltas.tolist(),
                           "image_pos_t_range": [int(img_pos[0].min()), int(img_pos[0].max())],
                           "image_pos_h_range": [int(img_pos[1].min()), int(img_pos[1].max())],
                           "image_pos_w_range": [int(img_pos[2].min()), int(img_pos[2].max())]}
    del full

    # 3. vision encoder alone, official weights, BF16 on GPU
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json"]))
    wmap = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    prefix = "thinker.visual."
    state = {}
    for shard in sorted({f for k, f in wmap.items() if k.startswith(prefix)}):
        with safe_open(str(Path(snapshot_download(MODEL_ID, allow_patterns=[shard])) / shard), "pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = f.get_tensor(k)
    vc._attn_implementation = "sdpa"
    visual = Qwen3OmniMoeVisionEncoder._from_config(vc, dtype=torch.bfloat16)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    report["visual_load"] = {"n_tensors": len(state), "missing": missing, "unexpected": unexpected,
                             "params": sum(p.numel() for p in visual.parameters())}
    visual = visual.cuda().eval()

    shapes: dict = OrderedDict()

    def rec_hook(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            shapes[name] = {"in": [list(i.shape) for i in inp if torch.is_tensor(i)],
                            "out": list(o.shape), "dtype": str(o.dtype),
                            "rms": float(o.float().pow(2).mean().sqrt())}
        return hook

    visual.patch_embed.register_forward_hook(rec_hook("patch_embed"))
    for i, b in enumerate(visual.blocks):
        b.register_forward_hook(rec_hook(f"blocks.{i}"))
    for i, m in enumerate(visual.merger_list):
        m.register_forward_hook(rec_hook(f"merger_list.{i}"))
    visual.merger.register_forward_hook(rec_hook("merger"))
    with torch.no_grad():
        out = visual(inputs["pixel_values"].cuda().to(torch.bfloat16), grid_thw=inputs["image_grid_thw"].cuda())
    report["visual_shapes"] = shapes
    report["visual_output"] = {"last_hidden_state": list(out.last_hidden_state.shape),
                               "pooler_output": list(out.pooler_output.shape),
                               "deepstack_features": [list(d.shape) for d in out.deepstack_features]}
    g = block_tokens_to_grid(out.last_hidden_state, thw, vc.spatial_merge_size)
    m = merged_tokens_to_grid(out.pooler_output, thw, vc.spatial_merge_size)
    report["visual_output"]["block_grid"] = list(g.shape)
    report["visual_output"]["merged_grid"] = list(m.shape)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("thinker_tree", "visual_shapes", "config")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
