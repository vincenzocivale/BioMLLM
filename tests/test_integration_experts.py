"""Integration tests on real checkpoints (need the HF cache: source scripts/env.sh).

    BIOMLLM_INTEGRATION=1 pytest tests/test_integration_experts.py
"""

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(os.environ.get("BIOMLLM_INTEGRATION") != "1",
                                reason="set BIOMLLM_INTEGRATION=1 to run on real checkpoints")

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "expert"
QWEN = "Qwen/Qwen3-VL-4B-Instruct"


@pytest.mark.parametrize("name,grid,dim", [
    ("dinov2", (16, 16), 768), ("dinov2_518", (37, 37), 768), ("rad_dino", (37, 37), 768),
    ("siglip", (14, 14), 768), ("biomedclip", (14, 14), 768), ("qwen3vl_4b", (16, 16), 2560),
])
def test_expert_shapes(name, grid, dim):
    from biomllm.models.build import build_expert_from_cfg

    e = build_expert_from_cfg(OmegaConf.load(CONFIGS / f"{name}.yaml"))
    f = e(torch.rand(2, 3, 512, 512))
    assert f.grid == grid and f.tokens.shape == (2, grid[0] * grid[1], dim)
    assert torch.isfinite(f.tokens).all()


def test_qwen_vision_matches_official_model_fp32():
    from transformers import Qwen3VLForConditionalGeneration

    from biomllm.models.experts.registry import build_expert

    e = build_expert("qwen_vl_vision", model_id=QWEN, image_size=512, dtype="float32")
    x = torch.rand(1, 3, 512, 512)
    full = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.float32).eval()
    b = e.processor(images=[x[0].permute(1, 2, 0).numpy()], do_rescale=False, do_resize=False,
                    return_tensors="pt")
    with torch.no_grad():
        ref = full.model.get_image_features(b["pixel_values"], image_grid_thw=b["image_grid_thw"]).pooler_output[0]
    ours = e(x).tokens[0]
    assert (ours - ref).norm() / ref.norm() < 1e-5


def test_qwen_vision_token_order_is_row_major():
    from biomllm.models.experts.registry import build_expert

    torch.manual_seed(0)
    e = build_expert("qwen_vl_vision", model_id=QWEN, image_size=512, dtype="float32")
    x = torch.rand(1, 3, 512, 512)
    base = e(x).tokens[0]
    hits = 0
    for r, c in [(2, 5), (10, 3), (13, 13), (7, 0)]:
        x2 = x.clone()
        x2[0, :, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = 1 - x2[0, :, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32]
        d = (e(x2).tokens[0] - base).norm(dim=-1).view(16, 16)
        rr, cc = divmod(int(d.argmax()), 16)
        hits += abs(rr - r) <= 1 and abs(cc - c) <= 1
    assert hits == 4
