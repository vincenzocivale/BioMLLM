"""TaskTokenConditioner: T_i <- T_i + alpha * P_theta(F_i^S).

The conditioner never touches F^MLLM or the MLLM visual tokens; it only adds a correction
to the task tokens, either before the LLM (the task queries e_task + F^MLLM) or after it
(the task-token hidden states that feed the mask / box / detection heads).

`mode="prepend"` is the VPT-style alternative: instead of being summed into each T_i, the
projected expert features are placed as extra tokens in front of the task tokens (pre-LLM
only). Unlike `add` with a zero-initialised gate, it does not start exactly as the baseline,
because the prepended tokens take part in attention even when they are zero.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

from biomllm.models.conditioning.gate import Gate
from biomllm.models.conditioning.projector import PointwiseProjector, Projector
from biomllm.models.conditioning.regularization import drop_native
from biomllm.models.types import FeatureMap, TaskQueries


class InjectionPoint(str, Enum):
    PRE_LLM = "pre_llm"
    POST_LLM = "post_llm"


class InjectionMode(str, Enum):
    ADD = "add"
    PREPEND = "prepend"


class TaskTokenConditioner(nn.Module):
    def __init__(self, projector: Projector, gate: Gate,
                 injection_point: InjectionPoint | str = InjectionPoint.PRE_LLM,
                 mode: InjectionMode | str = InjectionMode.ADD,
                 prepend_grid: tuple[int, int] | None = None,
                 native_drop: float = 0.0):
        super().__init__()
        self.projector = projector
        self.gate = gate
        self.injection_point = InjectionPoint(injection_point)
        self.mode = InjectionMode(mode)
        self.prepend_grid = tuple(prepend_grid) if prepend_grid else None
        # Probability of dropping the F^MLLM term of the task tokens during training, so the
        # model cannot ignore the expert (ClinFusion's stochastic residual). Pre-LLM only.
        self.native_drop = native_drop
        if self.mode is InjectionMode.PREPEND:
            if self.injection_point is not InjectionPoint.PRE_LLM:
                raise ValueError("prepend injection is only defined before the LLM")
            if not isinstance(projector, PointwiseProjector):
                raise ValueError("prepend injection needs a pointwise projector (linear / mlp)")
        # Inference-time multiplier on alpha, for the alpha sweep (0 recovers the baseline).
        self.alpha_scale = 1.0

    def set_alpha_scale(self, scale: float) -> None:
        self.alpha_scale = float(scale)

    def correction(self, queries: TaskQueries, feats: FeatureMap) -> torch.Tensor:
        delta = self.projector(queries, feats)
        alpha = self.gate(queries.tokens) * self.alpha_scale
        return alpha * delta

    def forward(self, queries: TaskQueries, feats: FeatureMap) -> TaskQueries:
        if self.injection_point is InjectionPoint.PRE_LLM:
            queries = drop_native(queries, self.native_drop, self.training)
        if self.alpha_scale == 0.0:
            return queries
        if self.mode is InjectionMode.PREPEND:
            return self._prepend(queries, feats)
        tokens = queries.tokens + self.correction(queries, feats).to(queries.tokens.dtype)
        return TaskQueries(tokens, queries.grid, queries.native, queries.num_prefix)

    def _prepend(self, queries: TaskQueries, feats: FeatureMap) -> TaskQueries:
        grid = self.prepend_grid or queries.grid or feats.grid
        b, _, d = queries.tokens.shape
        slots = TaskQueries(queries.tokens.new_zeros(b, grid[0] * grid[1], d), grid=grid)
        extra = self.correction(slots, feats).to(queries.tokens.dtype)
        tokens = torch.cat([extra, queries.tokens], dim=1)
        native = None
        if queries.native is not None:
            native = torch.cat([torch.zeros_like(extra), queries.native], dim=1)
        return TaskQueries(tokens, queries.grid, native, queries.num_prefix + extra.shape[1])
