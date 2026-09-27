"""Checks on the real Qwen3-Omni checkpoint (opt-in: ~18 GB of GPU, ~1 min to load).

    BIOMLLM_OMNI_CHECKPOINT=1 CUDA_VISIBLE_DEVICES=1 pytest tests/test_omni_checkpoint.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("BIOMLLM_OMNI_CHECKPOINT") != "1" or not torch.cuda.is_available(),
                                reason="set BIOMLLM_OMNI_CHECKPOINT=1 (needs the checkpoint and a GPU)")

BUV_FRAME = "/raid/DATASETS/BioMLLMData/datasets/buv/images/malignant/2c12ff6464cfb1ff/000000.png"


@pytest.fixture(scope="module")
def omni():
    from biomllm.omni.loading import load_thinker

    thinker, processor, _ = load_thinker("nf4")
    return thinker, processor


@pytest.fixture(scope="module")
def inputs(omni):
    from PIL import Image

    from biomllm.omni.grounding import build_messages

    _, processor = omni
    img = Image.open(BUV_FRAME).convert("RGB")
    text = processor.apply_chat_template(build_messages(img, "Locate the object: breast lesion."),
                                         add_generation_prompt=True, tokenize=False)
    x = processor(text=text, images=[img], return_tensors="pt")
    x = {k: v.cuda() for k, v in x.items()}
    x["pixel_values"] = x["pixel_values"].to(torch.bfloat16)
    return x


def test_taps_equal_what_the_official_forward_feeds_the_thinker(omni, inputs):
    from biomllm.omni.features import VisualTaps
    from biomllm.omni.layout import merged_tokens_to_grid

    thinker, _ = omni
    seen = {}

    def grab(mod, args, kwargs):
        seen["embeds"] = kwargs["inputs_embeds"].detach().clone()
        seen["deepstack"] = [d.detach().clone() for d in kwargs["deepstack_visual_embeds"]]
        seen["mask"] = kwargs["visual_pos_masks"].detach().clone()

    h = thinker.model.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad(), VisualTaps(thinker.visual, blocks=(8, 26)) as taps:
            thinker(**inputs)
    finally:
        h.remove()
    maps = taps.maps()[0]
    thw = inputs["image_grid_thw"][0].tolist()
    img_embeds = seen["embeds"][seen["mask"]]  # [196, 2048] tokens the LLM actually reads
    assert img_embeds.shape[0] == thw[1] * thw[2] // 4
    assert torch.equal(maps["merged"], merged_tokens_to_grid(img_embeds, thw, 2))
    for k in range(3):
        assert torch.equal(maps[f"deepstack.{k}"], merged_tokens_to_grid(seen["deepstack"][k], thw, 2))
    assert maps["blocks.26"].shape == (1, 1152, thw[1], thw[2])


def test_steering_with_zero_gates_reproduces_vanilla_logits(omni, inputs):
    from biomllm.omni.steering import VisualSteering

    thinker, _ = omni
    with torch.no_grad():
        ref = thinker(**inputs).logits
    steer = VisualSteering(thinker.visual, layers=tuple(range(1, 27, 2)), text_dim=1024).cuda().to(torch.bfloat16)
    steer.attach()
    try:
        text = torch.randn(1, 12, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), steer.condition(text, torch.ones(1, 12, dtype=torch.bool, device="cuda")):
            got = thinker(**inputs).logits
            with torch.no_grad():
                for m in steer.blocks.values():
                    m.alpha.fill_(0.5)
            moved = thinker(**inputs).logits
    finally:
        steer.detach()
    assert torch.equal(ref, got)
    assert not torch.equal(ref, moved)
