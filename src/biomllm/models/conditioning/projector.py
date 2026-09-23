"""P_theta: maps specialist features into the task-token space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomllm.models.conditioning.alignment import global_pool, resample_to_grid
from biomllm.models.types import FeatureMap, TaskQueries


class Projector(nn.Module):
    """Returns the correction P(F^S) with the same shape as the task tokens."""

    def __init__(self, in_dim: int, out_dim: int, zero_init_output: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.zero_init_output = zero_init_output

    def output_layer(self) -> nn.Linear:
        raise NotImplementedError

    def _maybe_zero_init(self) -> None:
        if self.zero_init_output:
            layer = self.output_layer()
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, queries: TaskQueries, feats: FeatureMap) -> torch.Tensor:
        raise NotImplementedError


class PointwiseProjector(Projector):
    """Token-wise map. With spatial queries, F^S is resampled onto the query grid so the
    correction for T_i comes from F_i^S; otherwise F^S is globally pooled and the same
    correction is broadcast to every task token."""

    def __init__(self, in_dim: int, out_dim: int, zero_init_output: bool = False,
                 resample_mode: str = "area"):
        super().__init__(in_dim, out_dim, zero_init_output)
        self.resample_mode = resample_mode

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, queries: TaskQueries, feats: FeatureMap) -> torch.Tensor:
        if queries.grid is not None:
            x = resample_to_grid(feats, queries.grid, self.resample_mode).tokens
        else:
            x = global_pool(feats)
        delta = self._map(x)
        return delta.expand_as(queries.tokens)


class LinearProjector(PointwiseProjector):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self._maybe_zero_init()

    def output_layer(self) -> nn.Linear:
        return self.proj

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class MLPProjector(PointwiseProjector):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None,
                 **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        hidden_dim = hidden_dim or out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self._maybe_zero_init()

    def output_layer(self) -> nn.Linear:
        return self.fc2

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class CrossAttnProjector(Projector):
    """Task tokens attend to F^S. Needed when task tokens are not indexed by the patch grid
    (instance queries, a single [SEG] hidden state after the LLM)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None,
                 num_heads: int = 8, zero_init_output: bool = False):
        super().__init__(in_dim, out_dim, zero_init_output)
        hidden_dim = hidden_dim or out_dim
        self.q_norm = nn.LayerNorm(out_dim)
        self.kv_norm = nn.LayerNorm(in_dim)
        self.q_proj = nn.Linear(out_dim, hidden_dim)
        self.kv_proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out = nn.Linear(hidden_dim, out_dim)
        self._maybe_zero_init()

    def output_layer(self) -> nn.Linear:
        return self.out

    def forward(self, queries: TaskQueries, feats: FeatureMap) -> torch.Tensor:
        q = self.q_proj(self.q_norm(queries.tokens))
        kv = self.kv_proj(self.kv_norm(feats.tokens))
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out(attended)


class LocalCrossAttnProjector(Projector):
    """Each spatial task token T_i attends to the k x k neighbourhood of F^S around the same
    location (ClinFusion's CaSL local cross-attention, used here on task tokens instead of the
    visual stream). F^S is first resampled onto the query grid; k=1 reduces to a pointwise map.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None,
                 num_heads: int = 8, kernel_size: int = 3, zero_init_output: bool = False,
                 resample_mode: str = "area"):
        super().__init__(in_dim, out_dim, zero_init_output)
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        hidden_dim = hidden_dim or out_dim
        self.kernel_size = kernel_size
        self.resample_mode = resample_mode
        self.q_norm = nn.LayerNorm(out_dim)
        self.kv_norm = nn.LayerNorm(in_dim)
        self.q_proj = nn.Linear(out_dim, hidden_dim)
        self.kv_proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out = nn.Linear(hidden_dim, out_dim)
        self._maybe_zero_init()

    def output_layer(self) -> nn.Linear:
        return self.out

    def forward(self, queries: TaskQueries, feats: FeatureMap) -> torch.Tensor:
        if queries.grid is None:
            raise ValueError("local_cross_attn needs spatially indexed task tokens (grid != None)")
        b, n, _ = queries.tokens.shape
        k = self.kernel_size
        kv = self.kv_proj(self.kv_norm(resample_to_grid(feats, queries.grid, self.resample_mode).tokens))
        kv_map = FeatureMap(kv, queries.grid).as_image()                       # [B, H, h, w]
        hd = kv.shape[-1]
        # [B, H*k*k, n] -> [B*n, k*k, H]
        windows = F.unfold(kv_map, k, padding=k // 2).view(b, hd, k * k, n)
        windows = windows.permute(0, 3, 2, 1).reshape(b * n, k * k, hd)
        valid = F.unfold(torch.ones_like(kv_map[:, :1]), k, padding=k // 2)   # [B, k*k, n]
        pad_mask = valid.permute(0, 2, 1).reshape(b * n, k * k) == 0          # True = ignore
        q = self.q_proj(self.q_norm(queries.tokens)).reshape(b * n, 1, hd)
        attended, _ = self.attn(q, windows, windows, key_padding_mask=pad_mask, need_weights=False)
        return self.out(attended.view(b, n, hd))


PROJECTORS: dict[str, type[Projector]] = {
    "linear": LinearProjector,
    "mlp": MLPProjector,
    "cross_attn": CrossAttnProjector,
    "local_cross_attn": LocalCrossAttnProjector,
}


def build_projector(name: str, in_dim: int, out_dim: int, **kwargs) -> Projector:
    if name not in PROJECTORS:
        raise KeyError(f"unknown projector '{name}', available: {sorted(PROJECTORS)}")
    return PROJECTORS[name](in_dim, out_dim, **kwargs)
