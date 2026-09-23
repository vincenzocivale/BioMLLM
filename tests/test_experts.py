import pytest
import torch

from biomllm.models.experts.registry import EXPERTS, build_expert


def test_toy_expert_shapes_and_freeze():
    e = build_expert("toy", dim=32, image_size=64, patch_size=8)
    f = e(torch.rand(2, 3, 128, 96))  # resized to the expert's own resolution
    assert f.tokens.shape == (2, 64, 32) and f.grid == (8, 8)
    assert all(not p.requires_grad for p in e.parameters())
    assert not f.tokens.requires_grad


def test_grayscale_input_is_expanded():
    e = build_expert("toy")
    assert e(torch.rand(1, 1, 64, 64)).tokens.shape[0] == 1


def test_registry_contains_paper_experts():
    assert {"hf_vit", "siglip", "timm", "open_clip", "sam_encoder"} <= set(EXPERTS)


def test_unknown_expert():
    with pytest.raises(KeyError):
        build_expert("nope")
