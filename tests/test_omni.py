"""CPU tests of the Qwen3-Omni layout / feature taps / steering code against the official HF
modules (tiny random-weight vision encoder with the real architecture; no checkpoint needed).
The checkpoint-level checks live in tests/test_omni_checkpoint.py (opt-in, GPU)."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeVisionEncoderConfig)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeVisionEncoder)

from biomllm.omni.features import VisualTaps  # noqa: E402
from biomllm.omni.grounding import box_iou, parse_grounding, to_pixels  # noqa: E402
from biomllm.omni.layout import (block_tokens_to_grid, grid_to_block_tokens,  # noqa: E402
                                 merged_tokens_to_grid, pixel_patches_to_image)
from biomllm.omni.steering import VisualSteering  # noqa: E402

PATCH, MERGE, TEMPORAL = 4, 2, 2


@pytest.fixture(scope="module")
def visual():
    torch.manual_seed(0)
    cfg = Qwen3OmniMoeVisionEncoderConfig(depth=4, hidden_size=32, intermediate_size=64, num_heads=2,
                                          out_hidden_size=48, patch_size=PATCH, spatial_merge_size=MERGE,
                                          temporal_patch_size=TEMPORAL, num_position_embeddings=64,
                                          deepstack_visual_indexes=[1, 2], in_channels=3)
    cfg._attn_implementation = "sdpa"
    enc = Qwen3OmniMoeVisionEncoder._from_config(cfg).eval()
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return enc.requires_grad_(False)


def patchify(images: torch.Tensor):
    """The official processor's patchify (Qwen2VLImageProcessor)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    flat, gh, gw = Qwen2VLImageProcessor.patchify(None, images, PATCH, MERGE, TEMPORAL)
    return flat, (1, gh, gw)


def two_images():
    torch.manual_seed(1)
    a, b = torch.randn(1, 3, 16, 24), torch.randn(1, 3, 24, 16)
    (pa, ga), (pb, gb) = patchify(a), patchify(b)
    return (a, b), torch.cat([pa[0], pb[0]]), torch.tensor([ga, gb])


# -- layout --------------------------------------------------------------------------------
def test_unpatchify_inverts_official_processor():
    (a, b), pix, thw = two_images()
    n = int(thw[0].prod())
    rec = pixel_patches_to_image(pix[:n], thw[0].tolist(), PATCH, MERGE, TEMPORAL)
    assert torch.equal(rec[0], a[0]) and torch.equal(rec[1], a[0])


def test_block_grid_roundtrip_and_merge_windows():
    x = torch.randn(1, 5, 4, 6)
    tok = grid_to_block_tokens(x, MERGE)
    assert torch.equal(block_tokens_to_grid(tok, (1, 4, 6), MERGE), x)
    # each run of MERGE*MERGE tokens is one 2x2 window, windows in raster order
    win = tok.view(-1, MERGE * MERGE, 5)
    assert torch.equal(win[1], x[0, :, 0:2, 2:4].permute(1, 2, 0).reshape(4, 5))
    m = torch.randn(6, 7)
    assert merged_tokens_to_grid(m, (1, 4, 6), MERGE)[0, :, 1, 0].equal(m[3])


# -- feature taps -------------------------------------------------------------------------
def test_taps_match_official_outputs(visual):
    _, pix, thw = two_images()
    with VisualTaps(visual, blocks=(3,)) as taps:
        out = visual(pix, grid_thw=thw)
    maps = taps.maps()
    assert len(maps) == 2
    n0 = int(thw[0].prod())
    assert torch.equal(maps[0]["blocks.3"], block_tokens_to_grid(out.last_hidden_state[:n0], thw[0].tolist(), MERGE))
    assert torch.equal(maps[1]["merged"], merged_tokens_to_grid(out.pooler_output[n0 // 4:], thw[1].tolist(), MERGE))
    assert torch.equal(maps[0]["deepstack.1"],
                       merged_tokens_to_grid(out.deepstack_features[1][:n0 // 4], thw[0].tolist(), MERGE))
    assert maps[1]["blocks.3"].shape == (1, 32, 6, 4)


# -- steering -----------------------------------------------------------------------------
def _run(visual, pix, thw):
    o = visual(pix, grid_thw=thw)
    return [o.last_hidden_state, o.pooler_output, *o.deepstack_features]


def test_zero_gate_is_bit_identical_to_vanilla(visual):
    _, pix, thw = two_images()
    ref = _run(visual, pix, thw)
    steer = VisualSteering(visual, layers=(0, 2), text_dim=10, num_heads=2).attach()
    text, mask = torch.randn(2, 5, 10), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).bool()
    with steer.condition(text, mask):
        got = _run(visual, pix, thw)
    steer.detach()
    assert all(torch.equal(r, g) for r, g in zip(ref, got))


def test_open_gate_steers_each_item_with_its_own_text(visual):
    _, pix, thw = two_images()
    n0 = int(thw[0].prod())
    ref = visual(pix, grid_thw=thw).last_hidden_state
    steer = VisualSteering(visual, layers=(1,), text_dim=10, num_heads=2).attach()
    with torch.no_grad():
        steer.blocks["1"].alpha.fill_(1.0)
    t1, mask = torch.randn(2, 4, 10), torch.ones(2, 4, dtype=torch.bool)
    t2 = t1.clone()
    t2[1] = torch.randn(4, 10)
    with steer.condition(t1, mask):
        a = visual(pix, grid_thw=thw).last_hidden_state
    with steer.condition(t2, mask):
        b = visual(pix, grid_thw=thw).last_hidden_state
    # outside the context the hooks are a no-op
    assert torch.equal(visual(pix, grid_thw=thw).last_hidden_state, ref)
    steer.detach()
    assert not torch.allclose(a, ref)
    assert torch.equal(a[:n0], b[:n0])            # image 0: same text -> same tokens
    assert not torch.allclose(a[n0:], b[n0:])     # image 1: its own text changed


def test_only_steering_parameters_receive_gradients(visual):
    _, pix, thw = two_images()
    steer = VisualSteering(visual, layers=(0, 3), text_dim=10, num_heads=2).attach()
    with torch.no_grad():
        for m in steer.blocks.values():
            m.alpha.fill_(0.3)
    with steer.condition(torch.randn(2, 3, 10), torch.ones(2, 3, dtype=torch.bool)):
        visual(pix, grid_thw=thw).pooler_output.pow(2).mean().backward()
    steer.detach()
    assert all(p.grad is None for p in visual.parameters())
    assert all(p.grad is not None for p in steer.parameters())
    assert not any(n.startswith("_visual") for n, _ in steer.named_parameters())


def test_mismatched_condition_count_raises(visual):
    _, pix, thw = two_images()
    steer = VisualSteering(visual, layers=(0,), text_dim=10, num_heads=2).attach()
    try:
        with steer.condition(torch.randn(1, 3, 10), torch.ones(1, 3, dtype=torch.bool)):
            with pytest.raises(ValueError):
                visual(pix, grid_thw=thw)
    finally:
        steer.detach()


# -- grounding parser ---------------------------------------------------------------------
def test_parse_official_cookbook_answer():
    text = ('<think>\nThe lesion is dark.\n</think>\n\n```json\n[\n  {"bbox_2d": [11, 519, 95, 717], '
            '"label": "lesion"}\n]\n```')
    p = parse_grounding(text)
    assert p.failure is None and p.boxes == [[11, 519, 95, 717]] and p.labels == ["lesion"]
    assert to_pixels(p.boxes[0], 1000, 500) == [11, 259.5, 95, 358.5]


@pytest.mark.parametrize("text,failure", [
    ("<think>still reasoning", "no_think_end"),
    ("<think>x</think> There is no lesion.", "no_json"),
    ("<think>x</think> ```json\n[]\n```", "empty"),
    ('<think>x</think> [{"label": "a"}]', "no_bbox"),
])
def test_parse_failures_are_labelled(text, failure):
    assert parse_grounding(text).failure == failure


def test_parse_nothink_output_that_reasons_anyway():
    text = 'Got it, the box is [1, 2, 3, 4]. Yes.\n</think>\n\n```json\n[{"bbox_2d": [5, 6, 7, 8]}]\n```'
    assert parse_grounding(text, thinking=False).boxes == [[5, 6, 7, 8]]
    assert parse_grounding("**Bounding Box**: [5, 6, 7, 8]", thinking=False).failure == "no_bbox"


def test_parse_truncated_json_recovers_boxes():
    p = parse_grounding('<think>x</think>[{"bbox_2d": [1, 2, 3, 4], "label": "a"}, {"bbox_2d": [5, 6')
    assert p.failure is None and p.boxes == [[1, 2, 3, 4]]


def test_box_iou():
    assert box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


# -- mask decoder ------------------------------------------------------------------------
def test_mask_decoder_shapes_and_size():
    from biomllm.omni.mask_decoder import OmniMaskDecoder, box_channel

    dims = {"blocks.8": 1152, "blocks.16": 1152, "blocks.26": 1152, "merged": 2048}
    dec = OmniMaskDecoder(dims)
    maps = {k: torch.randn(2, c, 28, 28) if k.startswith("blocks") else torch.randn(2, c, 14, 14)
            for k, c in dims.items()}
    boxes = torch.tensor([[0.1, 0.2, 0.5, 0.6], [0.0, 0.0, 1.0, 1.0]])
    out = dec(maps, boxes)
    assert out.shape == (2, 1, 112, 112)
    assert sum(p.numel() for p in dec.parameters()) < 3e6
    bc = box_channel(boxes, 10, 10)
    assert bc[0, 0].sum() == 16 and bc[1, 0].sum() == 100


# -- NF4 experts (GPU: bitsandbytes 4-bit kernels) ---------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nf4_experts_decode_path_matches_loop_and_bf16():
    from types import SimpleNamespace

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts

    from biomllm.omni.loading import NF4Experts

    torch.manual_seed(0)
    cfg = SimpleNamespace(num_experts=8, hidden_size=128, moe_intermediate_size=64, hidden_act="silu",
                          _experts_implementation=None)
    ref = Qwen3OmniMoeThinkerTextExperts(cfg)
    torch.nn.init.normal_(ref.gate_up_proj, std=0.05)
    torch.nn.init.normal_(ref.down_proj, std=0.05)
    ref = ref.cuda().to(torch.bfloat16)
    q = NF4Experts(ref, torch.device("cuda"))
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor([[0, 5], [5, 7], [2, 0]], device="cuda")
    w = torch.softmax(torch.randn(3, 2, device="cuda"), -1).to(torch.bfloat16)
    fast = q(x, idx, w)
    q.decode_rows = 0
    loop = q(x, idx, w)
    assert torch.allclose(fast, loop, atol=1e-2, rtol=1e-2)
    exact = ref(x, idx, w)
    rel = (loop.float() - exact.float()).norm() / exact.float().norm()
    assert rel < 0.2  # NF4 error on random Gaussian weights, two matmuls: ~0.15 measured
