"""Tests for the phase-1 components: local cross-attention, static / shuffle controls, prepend
injection, native-term drop, freeze policies, modality packages and cost tracking."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from biomllm.models.build import build_model
from biomllm.models.conditioned_mllm import ConditionedMLLM
from biomllm.models.conditioning.projector import build_projector
from biomllm.models.conditioning.regularization import drop_native
from biomllm.models.packages import ModalityPackage, PackageRegistry
from biomllm.models.types import FeatureMap, TaskQueries
from biomllm.training.costs import CostTracker
from biomllm.training.param_groups import apply_freeze_policy

DIM = 64
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(overrides):
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        return compose("config", overrides=["+experiment=debug", *overrides])


@pytest.fixture
def images():
    torch.manual_seed(1)
    return torch.rand(2, 3, 64, 64)


def _open_gate(model, value=0.7):
    with torch.no_grad():
        model.conditioner.gate.logit.fill_(value)


# ----------------------------------------------------------------- local cross-attention

def _changed_positions(kernel_size, pos, grid=(5, 5)):
    torch.manual_seed(0)
    n = grid[0] * grid[1]
    proj = build_projector("local_cross_attn", 16, DIM, num_heads=4, kernel_size=kernel_size)
    q = TaskQueries(torch.randn(1, n, DIM), grid=grid)
    f = torch.randn(1, n, 16)
    d0 = proj(q, FeatureMap(f, grid))
    f2 = f.clone()
    f2[0, pos] += 5.0 * torch.randn(16)  # not a constant shift: LayerNorm would cancel it
    d1 = proj(q, FeatureMap(f2, grid))
    return set(((d1 - d0).abs().sum(-1)[0] > 1e-6).nonzero().flatten().tolist())


def test_local_cross_attn_k1_is_pointwise():
    assert _changed_positions(1, pos=12) == {12}


def test_local_cross_attn_receptive_field_is_k_by_k():
    # centre of a 5x5 grid: its 3x3 neighbourhood
    assert _changed_positions(3, pos=12) == {6, 7, 8, 11, 12, 13, 16, 17, 18}


def test_local_cross_attn_border_padding_is_masked():
    # a corner only reaches its 2x2 neighbourhood; padded slots must not produce NaNs
    assert _changed_positions(3, pos=0) == {0, 1, 5, 6}


def test_local_cross_attn_needs_spatial_queries():
    proj = build_projector("local_cross_attn", 16, DIM, num_heads=4)
    with pytest.raises(ValueError):
        proj(TaskQueries(torch.randn(1, 1, DIM)), FeatureMap(torch.randn(1, 4, 16), (2, 2)))


# ----------------------------------------------------------------------- static / shuffle

def test_static_source_ignores_the_image(images):
    model = build_model(_cfg(["conditioner=static", "conditioner.static_dim=16"]))
    visual = model.mllm.visual_features(images)
    f = model.conditioning_features(images, visual, {})
    f_other = model.conditioning_features(torch.rand_like(images), visual, {})
    torch.testing.assert_close(f.tokens, f_other.tokens)
    assert model.static_features.requires_grad


def test_shuffle_pairs_images_with_other_features(images):
    model = build_model(_cfg(["conditioner=specialist"]))
    _open_gate(model)
    out = model(images, "seg")["mask_logits"]
    model.shuffle_features = True
    shuffled = model(images, "seg")["mask_logits"]
    assert not torch.allclose(out, shuffled)
    with pytest.raises(ValueError):
        model(images[:1], "seg")


# -------------------------------------------------------------------------------- prepend

def test_prepend_strips_prefix_and_alpha_zero_is_baseline(images):
    model = build_model(_cfg(["conditioner=specialist", "injection=prepend", "projector=mlp"]))
    base = ConditionedMLLM(model.mllm, "none")
    _open_gate(model)
    out = model(images, "seg")["mask_logits"]
    assert out.shape == (2, 1, 8, 8)
    assert not torch.allclose(out, base(images, "seg")["mask_logits"])
    model.conditioner.set_alpha_scale(0.0)
    torch.testing.assert_close(model(images, "seg"), base(images, "seg"))


@pytest.mark.parametrize("overrides", [
    ["injection=prepend", "projector=cross_attn"],
    ["injection=prepend", "injection.point=post_llm"],
])
def test_prepend_invalid_configurations(overrides):
    with pytest.raises(ValueError):
        build_model(_cfg(["conditioner=specialist", *overrides]))


# ------------------------------------------------------------------------ native injection

def test_native_injection_biases_the_llm_context(images):
    """V' = V + alpha * P(F^S) reaches the LLM (and the task tokens built from V), and the
    projector is trained through the frozen LLM."""
    model = build_model(_cfg(["conditioner=specialist", "injection=native", "projector=mlp"]))
    base = ConditionedMLLM(model.mllm, "none")
    torch.testing.assert_close(model(images, "seg"), base(images, "seg"))
    _open_gate(model)
    seen = {}
    hook = model.mllm.llm.register_forward_hook(lambda m, args, out: seen.setdefault("x", args[0]))
    out = model(images, "seg")["mask_logits"]
    hook.remove()
    visual = model.mllm.visual_features(images).tokens
    n_vis = visual.shape[1]
    assert not torch.allclose(seen["x"][:, :n_vis], visual)
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in model.conditioner.projector.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.mllm.vision.parameters())


def test_native_prepend_adds_image_tokens_and_alpha_zero_is_baseline(images):
    model = build_model(_cfg(["conditioner=specialist", "injection=native_prepend", "projector=mlp"]))
    base = ConditionedMLLM(model.mllm, "none")
    _open_gate(model)
    seen = {}
    hook = model.mllm.llm.register_forward_hook(lambda m, args, out: seen.setdefault("x", args[0]))
    out = model(images, "seg")["mask_logits"]
    hook.remove()
    assert out.shape == (2, 1, 8, 8)
    n_vis, n_extra, n_q = 64, 8 * 8, 64
    assert seen["x"].shape[1] == n_vis + n_extra + n_q
    # V itself is untouched: only extra tokens are added.
    torch.testing.assert_close(seen["x"][:, :n_vis], model.mllm.visual_features(images).tokens)
    model.conditioner.set_alpha_scale(0.0)
    torch.testing.assert_close(model(images, "seg"), base(images, "seg"))


# ----------------------------------------------------------------------------- native drop

def test_drop_native():
    native = torch.randn(4, 6, DIM)
    base = torch.randn(4, 6, DIM)
    q = TaskQueries(base + native, grid=(2, 3), native=native)
    dropped = drop_native(q, p=1.0, training=True)
    torch.testing.assert_close(dropped.tokens, base)
    assert drop_native(q, p=1.0, training=False) is q
    assert drop_native(q, p=0.0, training=True) is q


def test_native_drop_only_in_training(images):
    model = build_model(_cfg(["conditioner=specialist", "conditioner.native_drop=1.0"]))
    base = ConditionedMLLM(model.mllm, "none")
    model.eval()
    torch.testing.assert_close(model(images, "seg"), base(images, "seg"))
    model.train()
    assert not torch.allclose(model(images, "seg")["mask_logits"], base(images, "seg")["mask_logits"])


# ------------------------------------------------------------------------- freeze policies

def test_frozen_policy_trains_only_task_params_and_conditioner(images):
    model = build_model(_cfg(["conditioner=specialist", "gate=fixed"]))
    with torch.no_grad():
        model.conditioner.projector.output_layer().weight.normal_()
    model(images, "seg")["mask_logits"].pow(2).mean().backward()
    task_ids = {id(p) for p in model.mllm.task_parameters().values()}
    for name, p in model.mllm.named_parameters():
        if id(p) in task_ids:
            continue
        assert not p.requires_grad and p.grad is None, name
    assert any(p.grad is not None for p in model.conditioner.parameters())
    assert all(p.grad is None for p in model.expert.parameters())


def test_full_policy_keeps_native_encoder_frozen():
    model = build_model(_cfg(["conditioner=none", "train.policy=full"]))
    assert all(not p.requires_grad for p in model.mllm.vision.parameters())
    assert all(p.requires_grad for p in model.mllm.llm.parameters())


def test_lora_policy_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        build_model(_cfg(["conditioner=none", "train.policy=lora"]))


def test_unknown_policy():
    model = build_model(_cfg(["conditioner=none"]))
    with pytest.raises(ValueError):
        apply_freeze_policy(model, "everything")


# -------------------------------------------------------------------------------- packages

def _train_package(base_cfg_overrides, mllm, images):
    """Simulate a trained package: new conditioner + perturbed task params on the shared mllm."""
    cfg = _cfg(base_cfg_overrides)
    trained = build_model(cfg)
    trained.mllm = mllm
    _open_gate(trained, 0.5)
    with torch.no_grad():
        for p in mllm.task_parameters().values():
            p.add_(0.1 * torch.randn_like(p))
    return trained, cfg


def test_package_deactivation_restores_base_bitwise(images, tmp_path):
    base = build_model(_cfg(["conditioner=none"]))
    reference = {k: v.clone() for k, v in base(images, "seg").items()}
    registry = PackageRegistry(base)

    trained, cfg = _train_package(["conditioner=specialist"], base.mllm, images)
    expected = trained(images, "seg")["mask_logits"].detach().clone()
    registry.register(ModalityPackage.from_model("cxr", trained, cfg))
    registry.deactivate()  # the simulated training modified the shared task params
    assert torch.equal(base(images, "seg")["mask_logits"], reference["mask_logits"])

    registry.activate("cxr")
    torch.testing.assert_close(base(images, "seg")["mask_logits"], expected)
    registry.deactivate()
    assert torch.equal(base(images, "seg")["mask_logits"], reference["mask_logits"])


def test_two_packages_do_not_interfere(images):
    base = build_model(_cfg(["conditioner=none"]))
    registry = PackageRegistry(base)
    outputs = {}
    for name, overrides in {"cxr": ["conditioner=specialist"], "path": ["conditioner=self"]}.items():
        trained, cfg = _train_package(overrides, base.mllm, images)
        outputs[name] = trained(images, "seg")["mask_logits"].detach().clone()
        registry.register(ModalityPackage.from_model(name, trained, cfg))
    for name in ["cxr", "path", "cxr"]:
        registry.activate(name)
        torch.testing.assert_close(base(images, "seg")["mask_logits"], outputs[name])


def test_package_save_load_roundtrip(images, tmp_path):
    base = build_model(_cfg(["conditioner=none"]))
    registry = PackageRegistry(base)
    trained, cfg = _train_package(["conditioner=static", "conditioner.static_dim=16"], base.mllm, images)
    expected = trained(images, "seg")["mask_logits"].detach().clone()
    ModalityPackage.from_model("s", trained, cfg).save(tmp_path / "s.pt")
    registry.deactivate()
    registry.register(ModalityPackage.load(tmp_path / "s.pt", base.mllm))
    registry.activate("s")
    torch.testing.assert_close(base(images, "seg")["mask_logits"], expected)


def test_registry_requires_base_model():
    with pytest.raises(ValueError):
        PackageRegistry(build_model(_cfg(["conditioner=specialist"])))


# ------------------------------------------------------------------------------------ cost

def test_cost_tracker(images, tmp_path):
    model = build_model(_cfg(["conditioner=specialist"]))
    with CostTracker(model, num_gpus=1) as cost:
        model(images, "seg")["mask_logits"].sum().backward()
    report = cost.dump(tmp_path / "cost.json")
    assert report["params"]["mllm_base"]["trainable"] == 0
    assert report["params"]["expert"]["trainable"] == 0
    assert report["trainable_params"] == (report["params"]["task_params"]["trainable"]
                                          + report["params"]["conditioner"]["trainable"])
    assert report["gpu_hours"] > 0
