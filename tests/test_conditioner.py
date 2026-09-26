import itertools

import pytest
import torch

from biomllm.models.conditioned_mllm import ConditionedMLLM
from biomllm.models.conditioning.conditioner import TaskTokenConditioner
from biomllm.models.conditioning.gate import build_gate
from biomllm.models.conditioning.projector import build_projector
from biomllm.models.experts.registry import build_expert
from biomllm.models.mllm.adapters.toy import ToyMLLM
from biomllm.models.types import FeatureMap, TaskQueries

DIM = 64
PROJECTORS = ("linear", "mlp", "cross_attn", "local_cross_attn")
NON_SPATIAL_PROJECTORS = ("linear", "mlp", "cross_attn")
INJECTIONS = ("pre_llm", "post_llm", "native")
GATES = ("scalar", "token", "fixed")


def make_model(projector="mlp", injection="pre_llm", gate="scalar", spatial=True, seed=0):
    torch.manual_seed(seed)
    mllm = ToyMLLM(dim=DIM, spatial_queries=spatial)
    mllm.freeze_native()
    expert = build_expert("toy", dim=32)
    g = build_gate(gate, DIM)
    p = build_projector(projector, expert.dim, DIM, zero_init_output=not g.zero_at_init)
    cond = TaskTokenConditioner(p, g, injection)
    return ConditionedMLLM(mllm, "expert", cond, expert), ConditionedMLLM(mllm, "none")


@pytest.fixture
def images():
    return torch.rand(2, 3, 64, 64)


@pytest.mark.parametrize("projector,injection,gate", list(itertools.product(PROJECTORS, INJECTIONS, GATES)))
@pytest.mark.parametrize("task", ["seg", "box"])
def test_identical_to_baseline_at_init(projector, injection, gate, task, images):
    """At init the conditioned model must be exactly the baseline C0."""
    cond, base = make_model(projector, injection, gate)
    torch.testing.assert_close(cond(images, task), base(images, task))


@pytest.mark.parametrize("projector,injection", list(itertools.product(PROJECTORS, INJECTIONS)))
def test_alpha_scale_zero_recovers_baseline(projector, injection, images):
    cond, base = make_model(projector, injection, "scalar")
    with torch.no_grad():
        cond.conditioner.gate.logit.fill_(0.7)
    assert not torch.allclose(cond(images, "seg")["mask_logits"], base(images, "seg")["mask_logits"])
    cond.conditioner.set_alpha_scale(0.0)
    torch.testing.assert_close(cond(images, "seg"), base(images, "seg"))


@pytest.mark.parametrize("gate", GATES)
def test_no_gradient_deadlock_at_init(gate, images):
    """Zero-init gate + zero-init projector would give zero gradient to both."""
    cond, _ = make_model(gate=gate)
    cond(images, "seg")["mask_logits"].pow(2).mean().backward()
    grads = [p.grad for p in cond.conditioner.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_native_encoder_and_expert_receive_no_gradient(images):
    cond, _ = make_model(gate="fixed")
    cond(images, "seg")["mask_logits"].pow(2).mean().backward()
    assert all(p.grad is None for p in cond.mllm.vision.parameters())
    assert all(not p.requires_grad and p.grad is None for p in cond.expert.parameters())


def test_expert_stays_in_eval_mode():
    cond, _ = make_model()
    cond.train()
    assert not cond.expert.training
    assert cond.conditioner.training


def test_pointwise_correction_is_spatially_local():
    """With spatial queries, the correction for T_i comes only from F_i^S."""
    torch.manual_seed(0)
    proj = build_projector("linear", 16, DIM)
    q = TaskQueries(torch.randn(1, 16, DIM), grid=(4, 4))
    f = torch.randn(1, 16, 16)
    d0 = proj(q, FeatureMap(f, (4, 4)))
    f2 = f.clone()
    f2[0, 5] += 10
    d1 = proj(q, FeatureMap(f2, (4, 4)))
    changed = (d1 - d0).abs().sum(-1)[0] > 1e-6
    assert changed.nonzero().flatten().tolist() == [5]


def test_expert_grid_resampled_to_query_grid():
    proj = build_projector("mlp", 16, DIM)
    q = TaskQueries(torch.randn(2, 64, DIM), grid=(8, 8))
    delta = proj(q, FeatureMap(torch.randn(2, 37 * 37, 16), (37, 37)))
    assert delta.shape == q.tokens.shape


@pytest.mark.parametrize("projector", NON_SPATIAL_PROJECTORS)
@pytest.mark.parametrize("injection", INJECTIONS)
def test_non_spatial_task_tokens(projector, injection, images):
    """Single [SEG]-like token (LISA-style): pointwise projectors pool, cross-attn attends."""
    cond, base = make_model(projector, injection, "scalar", spatial=False)
    with torch.no_grad():
        cond.conditioner.gate.logit.fill_(0.5)
    out = cond(images, "box")["boxes"]
    assert out.shape == (2, 4)
    assert not torch.allclose(out, base(images, "box")["boxes"])


def test_self_source_is_detached(images):
    torch.manual_seed(0)
    mllm = ToyMLLM(dim=DIM)  # native encoder deliberately trainable here
    g = build_gate("fixed", DIM)
    cond = TaskTokenConditioner(build_projector("mlp", DIM, DIM, zero_init_output=False), g)
    model = ConditionedMLLM(mllm, "self", cond)
    feats = model.conditioning_features(images, mllm.visual_features(images), {})
    assert not feats.tokens.requires_grad


def test_noise_source_shapes(images):
    g = build_gate("scalar", DIM)
    cond = TaskTokenConditioner(build_projector("mlp", 48, DIM), g)
    model = ConditionedMLLM(ToyMLLM(dim=DIM), "noise", cond, noise_dim=48)
    assert model(images, "seg")["mask_logits"].shape == (2, 1, 8, 8)


def test_cached_expert_features_are_used(images):
    cond, _ = make_model(gate="fixed")
    with torch.no_grad():
        cond.conditioner.projector.output_layer().weight.normal_()
    live = cond(images, "seg")["mask_logits"]
    cached = cond.expert(images)
    torch.testing.assert_close(cond(images, "seg", {"expert_feats": cached})["mask_logits"], live)


def test_invalid_configurations():
    mllm = ToyMLLM(dim=DIM)
    with pytest.raises(ValueError):
        ConditionedMLLM(mllm, "expert")
    with pytest.raises(ValueError):
        ConditionedMLLM(mllm, "bogus")
