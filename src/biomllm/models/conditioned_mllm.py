"""ConditionedMLLM: an MLLM whose task tokens are corrected by an external encoder.

Conditioning sources (the comparison conditions of the paper):
    none    C0  standard task tokens (same trainable task parameters, no expert)
    static  ctrl alpha * P(learned image-independent map)   capacity without image signal
    self    C1  alpha * P(F^MLLM)        extra capacity, no new information
    expert  C2/C3  alpha * P(F^S)        generalist or specialist encoder (set by `expert`)
    noise   ctrl  alpha * P(N(0, I))     extra capacity fed with noise
A randomly initialised expert (`expert.pretrained=false`) gives the "random encoder" control,
and `shuffle_features=True` (test time) pairs each image with another image's features.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from biomllm.models.conditioning.conditioner import InjectionPoint, TaskTokenConditioner
from biomllm.models.experts.base import FrozenExpert
from biomllm.models.mllm.base import TaskTokenMLLM
from biomllm.models.types import FeatureMap, TaskQueries

SOURCES = ("none", "static", "self", "expert", "noise")


class ConditionedMLLM(nn.Module):
    def __init__(self, mllm: TaskTokenMLLM, source: str = "none",
                 conditioner: TaskTokenConditioner | None = None,
                 expert: FrozenExpert | None = None, noise_dim: int | None = None,
                 static_dim: int | None = None, static_grid: tuple[int, int] = (16, 16)):
        super().__init__()
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got '{source}'")
        if (source == "none") != (conditioner is None):
            raise ValueError("a conditioner is required iff source != 'none'")
        if (source == "expert") != (expert is not None):
            raise ValueError("an expert is required iff source == 'expert'")
        if source == "noise" and not noise_dim:
            raise ValueError("source='noise' requires noise_dim")
        if source == "static" and not static_dim:
            raise ValueError("source='static' requires static_dim")
        self.mllm = mllm
        self.source = source
        self.conditioner = conditioner
        self.expert = expert
        self.noise_dim = noise_dim
        self.static_features = None
        if source == "static":
            self.static_grid = tuple(static_grid)
            n = self.static_grid[0] * self.static_grid[1]
            self.static_features = nn.Parameter(torch.randn(1, n, static_dim) * 0.02)
        # Test-time control: condition each image on another image's features.
        self.shuffle_features = False

    def conditioning_features(self, images: torch.Tensor, visual: FeatureMap,
                              batch: dict[str, Any]) -> FeatureMap:
        feats = self._source_features(images, visual, batch)
        if self.shuffle_features:
            if feats.tokens.shape[0] < 2:
                raise ValueError("shuffle_features needs a batch of at least 2 images")
            feats = FeatureMap(feats.tokens.roll(1, dims=0), feats.grid)
        return feats

    def _source_features(self, images: torch.Tensor, visual: FeatureMap,
                         batch: dict[str, Any]) -> FeatureMap:
        if self.source == "self":
            # Same features the MLLM already sees; detached so the native encoder is untouched.
            return FeatureMap(visual.tokens.detach(), visual.grid)
        if self.source == "expert":
            if "expert_feats" in batch:  # pre-computed by scripts/cache_expert_features.py
                cached = batch["expert_feats"]
                return cached if isinstance(cached, FeatureMap) else FeatureMap(*cached)
            return self.expert(images)
        b = visual.tokens.shape[0]
        if self.source == "static":
            return FeatureMap(self.static_features.expand(b, -1, -1), self.static_grid)
        if self.source == "noise":
            noise = torch.randn(b, visual.tokens.shape[1], self.noise_dim,
                                device=visual.tokens.device, dtype=visual.tokens.dtype)
            return FeatureMap(noise, visual.grid)
        raise RuntimeError(f"no conditioning features for source '{self.source}'")

    def forward(self, images: torch.Tensor, task: str,
                batch: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
        batch = batch or {}
        visual = self.mllm.visual_features(images)
        feats = None
        if self.conditioner is not None:
            feats = self.conditioning_features(images, visual, batch)

        queries = self.mllm.build_task_queries(task, visual, batch)
        if feats is not None and self.conditioner.injection_point is InjectionPoint.PRE_LLM:
            queries = self.conditioner(queries, feats)

        out = self.mllm.llm_forward(task, visual, queries, batch)
        hidden = out.task_hidden
        if hidden.tokens.shape[1] != queries.tokens.shape[1]:
            raise RuntimeError("llm_forward must return one hidden state per task token")
        task_hidden = TaskQueries(hidden.tokens, queries.grid, None, queries.num_prefix).without_prefix()
        if feats is not None and self.conditioner.injection_point is InjectionPoint.POST_LLM:
            task_hidden = self.conditioner(task_hidden, feats)

        preds = self.mllm.decode(task, task_hidden, visual, batch)
        if out.lm_loss is not None:
            preds["lm_loss"] = out.lm_loss
        return preds
