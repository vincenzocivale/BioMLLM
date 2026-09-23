"""Build a ConditionedMLLM from the composed Hydra config."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from biomllm.models.conditioned_mllm import ConditionedMLLM
from biomllm.models.conditioning.conditioner import InjectionPoint, TaskTokenConditioner
from biomllm.models.conditioning.gate import build_gate
from biomllm.models.conditioning.projector import build_projector
from biomllm.models.experts.registry import build_expert
from biomllm.models.mllm.adapters import build_mllm
from biomllm.training.param_groups import apply_freeze_policy


def _kwargs(node: DictConfig | None, drop: tuple[str, ...] = ()) -> dict:
    if node is None:
        return {}
    d = OmegaConf.to_container(node, resolve=True)
    return {k: v for k, v in d.items() if k not in drop}


def build_expert_from_cfg(expert_cfg: DictConfig):
    if expert_cfg.get("kind") is None:
        return None
    return build_expert(expert_cfg.kind, **_kwargs(expert_cfg, drop=("kind",)))


def build_conditioner(cfg: DictConfig, mllm, in_dim: int) -> TaskTokenConditioner:
    injection = InjectionPoint(cfg.injection.point)
    out_dim = mllm.query_dim if injection is InjectionPoint.PRE_LLM else mllm.hidden_dim
    gate = build_gate(cfg.gate.name, out_dim, **_kwargs(cfg.gate, drop=("name",)))
    # A zero-initialised gate already makes the model start as the baseline; zero-initialising
    # the projector as well would leave both with zero gradient. With a fixed alpha, the
    # projector output is zero-initialised instead.
    projector = build_projector(cfg.projector.name, in_dim, out_dim,
                                zero_init_output=not gate.zero_at_init,
                                **_kwargs(cfg.projector, drop=("name",)))
    return TaskTokenConditioner(projector, gate, injection,
                                mode=cfg.injection.get("mode", "add"),
                                prepend_grid=cfg.injection.get("prepend_grid"),
                                native_drop=cfg.conditioner.get("native_drop", 0.0))


def build_model(cfg: DictConfig) -> ConditionedMLLM:
    mllm = build_mllm(cfg.mllm.name, **_kwargs(cfg.mllm, drop=("name",)))
    source = cfg.conditioner.source

    if source == "none":
        model = ConditionedMLLM(mllm, source="none")
    else:
        expert = None
        extra = {}
        if source == "expert":
            expert = build_expert_from_cfg(cfg.expert)
            if expert is None:
                raise ValueError(f"conditioner '{cfg.conditioner.name}' needs an expert: set expert=<name>")
            in_dim = expert.dim
        elif source == "self":
            in_dim = _visual_dim(mllm)
        elif source == "static":
            in_dim = cfg.conditioner.static_dim
            extra = {"static_dim": in_dim, "static_grid": tuple(cfg.conditioner.static_grid)}
        else:  # noise
            in_dim = cfg.conditioner.noise_dim
            extra = {"noise_dim": in_dim}
        model = ConditionedMLLM(mllm, source=source, conditioner=build_conditioner(cfg, mllm, in_dim),
                                expert=expert, **extra)

    apply_freeze_policy(model, cfg.train.get("policy", "frozen"))
    return model


def _visual_dim(mllm) -> int:
    dim = getattr(mllm, "visual_dim", None)
    if dim is None:
        raise AttributeError(f"{type(mllm).__name__} must expose visual_dim for source='self'")
    return dim
