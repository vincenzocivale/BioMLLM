"""alpha: how much of the specialist correction enters the task token."""

from __future__ import annotations

import torch
import torch.nn as nn


class Gate(nn.Module):
    #: True when alpha starts at exactly 0, i.e. the model starts as the baseline.
    zero_at_init: bool = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns alpha broadcastable to tokens [B, Q, D]."""
        raise NotImplementedError


class FixedGate(Gate):
    def __init__(self, dim: int, value: float = 1.0):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.value.to(tokens.dtype)


class ScalarGate(Gate):
    """Learnable scalar, tanh-bounded, initialised at 0 (Flamingo-style)."""

    zero_at_init = True

    def __init__(self, dim: int, init: float = 0.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(init)))
        self.zero_at_init = init == 0.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.logit).to(tokens.dtype)


class TokenGate(Gate):
    """Per-token alpha_i = tanh(w^T T_i + b), zero-initialised: the model can decide per
    region / per query how much to trust the expert."""

    zero_at_init = True

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(tokens))


GATES: dict[str, type[Gate]] = {
    "fixed": FixedGate,
    "scalar": ScalarGate,
    "token": TokenGate,
}


def build_gate(name: str, dim: int, **kwargs) -> Gate:
    if name not in GATES:
        raise KeyError(f"unknown gate '{name}', available: {sorted(GATES)}")
    return GATES[name](dim, **kwargs)
