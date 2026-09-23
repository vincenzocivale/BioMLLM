"""Plug-and-play modality packages.

A package is everything a modality adds on top of the frozen base MLLM:
    the frozen expert, the conditioner (P_theta, alpha, optional static map)
    and the MLLM task parameters (e_task embeddings, heads) trained for that modality.
Activating a package swaps these in; deactivating restores the base model exactly (the base
task parameters are snapshotted and copied back, so outputs are bit-identical).

    registry = PackageRegistry(model)          # model built with conditioner=none
    registry.register(ModalityPackage.from_model("cxr", trained_cxr_model))
    registry.activate("cxr"); ...; registry.deactivate()
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from biomllm.models.conditioned_mllm import ConditionedMLLM


def _snapshot(mllm) -> dict[str, torch.Tensor]:
    return {k: p.detach().clone() for k, p in mllm.task_parameters().items()}


def _load_task_state(mllm, state: dict[str, torch.Tensor]) -> None:
    params = mllm.task_parameters()
    if set(params) != set(state):
        raise KeyError(f"task parameter mismatch: {sorted(set(params) ^ set(state))}")
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(state[k])


PACKAGE_CFG_KEYS = ("mllm", "conditioner", "expert", "projector", "injection", "gate")


def _package_cfg(cfg: DictConfig) -> dict:
    """The sub-configs needed to rebuild a package (skips run-level keys such as hydra paths)."""
    return {k: OmegaConf.to_container(cfg[k], resolve=True) for k in PACKAGE_CFG_KEYS}


class ModalityPackage(nn.Module):
    def __init__(self, name: str, source: str, conditioner: nn.Module | None,
                 expert: nn.Module | None, task_state: dict[str, torch.Tensor],
                 static_features: nn.Parameter | None = None,
                 static_grid: tuple[int, int] | None = None, noise_dim: int | None = None,
                 cfg: dict | None = None):
        super().__init__()
        self.name = name
        self.source = source
        self.conditioner = conditioner
        self.expert = expert
        self.task_state = task_state
        self.static_features = static_features
        self.static_grid = static_grid
        self.noise_dim = noise_dim
        self.cfg = cfg  # composed config used to build it, for save / load

    @classmethod
    def from_model(cls, name: str, model: ConditionedMLLM, cfg: DictConfig | None = None) -> ModalityPackage:
        """Snapshot the modality-specific parts of a trained ConditionedMLLM."""
        return cls(name, model.source, model.conditioner, model.expert, _snapshot(model.mllm),
                   model.static_features, getattr(model, "static_grid", None), model.noise_dim,
                   _package_cfg(cfg) if cfg is not None else None)

    def save(self, path: str | Path) -> None:
        """Saves only what was trained; the expert is rebuilt from its config (frozen hub weights)."""
        if self.cfg is None:
            raise ValueError("save() needs the package cfg (pass cfg to from_model)")
        torch.save({
            "name": self.name, "source": self.source, "cfg": self.cfg,
            "task_state": self.task_state,
            "conditioner": self.conditioner.state_dict() if self.conditioner is not None else None,
            "static_features": self.static_features.detach() if self.static_features is not None else None,
        }, path)

    @classmethod
    def load(cls, path: str | Path, mllm) -> ModalityPackage:
        from biomllm.models.build import build_conditioner, build_expert_from_cfg

        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(blob["cfg"])
        source = blob["source"]
        expert = build_expert_from_cfg(cfg.expert) if source == "expert" else None
        conditioner = static = None
        if source != "none":
            in_dim = {"expert": lambda: expert.dim, "self": lambda: mllm.visual_dim,
                      "static": lambda: cfg.conditioner.static_dim,
                      "noise": lambda: cfg.conditioner.noise_dim}[source]()
            conditioner = build_conditioner(cfg, mllm, in_dim)
            conditioner.load_state_dict(blob["conditioner"])
        if blob["static_features"] is not None:
            static = nn.Parameter(blob["static_features"])
        grid = tuple(cfg.conditioner.static_grid) if source == "static" else None
        noise_dim = cfg.conditioner.noise_dim if source == "noise" else None
        return cls(blob["name"], source, conditioner, expert, blob["task_state"], static, grid,
                   noise_dim, blob["cfg"])


class PackageRegistry:
    """Manages the packages of one base model. The model must start without a conditioner."""

    def __init__(self, model: ConditionedMLLM):
        if model.source != "none":
            raise ValueError("PackageRegistry needs the base model (conditioner=none)")
        self.model = model
        self.base_task_state = _snapshot(model.mllm)
        self.packages: dict[str, ModalityPackage] = {}
        self.active: str | None = None

    def register(self, package: ModalityPackage) -> None:
        self.packages[package.name] = package

    def activate(self, name: str) -> None:
        pkg = self.packages[name]
        m = self.model
        m.source, m.conditioner, m.expert = pkg.source, pkg.conditioner, pkg.expert
        m.static_features, m.noise_dim = pkg.static_features, pkg.noise_dim
        if pkg.static_grid is not None:
            m.static_grid = pkg.static_grid
        _load_task_state(m.mllm, pkg.task_state)
        self.active = name

    def deactivate(self) -> None:
        m = self.model
        m.source, m.conditioner, m.expert = "none", None, None
        m.static_features, m.noise_dim = None, None
        _load_task_state(m.mllm, self.base_task_state)
        self.active = None
