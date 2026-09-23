from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FeatureMap:
    """Patch tokens of a vision encoder laid out on a (h, w) grid.

    tokens: [B, h*w, C] in row-major order. CLS / register tokens are excluded.
    """

    tokens: torch.Tensor
    grid: tuple[int, int]

    def __post_init__(self) -> None:
        h, w = self.grid
        if self.tokens.dim() != 3 or self.tokens.shape[1] != h * w:
            raise ValueError(
                f"tokens {tuple(self.tokens.shape)} incompatible with grid {self.grid}"
            )

    @property
    def dim(self) -> int:
        return self.tokens.shape[-1]

    def as_image(self) -> torch.Tensor:
        """[B, C, h, w] view, for spatial resampling."""
        b, _, c = self.tokens.shape
        h, w = self.grid
        return self.tokens.transpose(1, 2).reshape(b, c, h, w)

    @classmethod
    def from_image(cls, x: torch.Tensor) -> FeatureMap:
        b, c, h, w = x.shape
        return cls(x.flatten(2).transpose(1, 2), (h, w))


@dataclass
class TaskQueries:
    """Task tokens T_i fed to (pre-LLM) or read from (post-LLM) the language model.

    tokens:     [B, Q, D].
    grid:       set when the Q queries are indexed by a spatial grid (Q == h*w), i.e. the
                T_i = e_task + F_i^MLLM case; None for a set of instance / [SEG]-like tokens.
    native:     the F^MLLM component already summed into `tokens` (same shape), when the
                adapter exposes it; used to stochastically drop the native term in training.
    num_prefix: extra tokens prepended in front of the Q task tokens (prepend injection);
                they are stripped before the task heads.
    """

    tokens: torch.Tensor
    grid: tuple[int, int] | None = None
    native: torch.Tensor | None = None
    num_prefix: int = 0

    def without_prefix(self) -> TaskQueries:
        if self.num_prefix == 0:
            return self
        native = None if self.native is None else self.native[:, self.num_prefix:]
        return TaskQueries(self.tokens[:, self.num_prefix:], self.grid, native, 0)


@dataclass
class MLLMOutput:
    task_hidden: TaskQueries
    lm_loss: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)
