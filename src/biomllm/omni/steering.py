"""Clinical feature steering inside the Qwen3-Omni VISUAL pathway (SteerViT, arXiv:2604.02327).

After selected vision blocks l, the packed visual tokens of each image/video are updated with a
gated cross-attention to that item's clinical-text tokens:

    V'_l = V_l + tanh(alpha_l) * CrossAttention(Q = V_l, K = T_clinical, V = T_clinical)

as in SteerViT: no FFN, one scalar gate per layer initialised to 0, text tokens from a frozen
encoder, L2-normalised and projected by a trainable 2-layer MLP (`ClinicalTextEncoder`).
Implementation detail not fixed by the paper: a LayerNorm on the queries (`query_norm`), because
Qwen3-Omni block outputs have very different scales across depth (RMS 0.3 -> 23, see
docs/qwen3_omni_architecture.md). With alpha = 0 the update is exactly `V + 0`, so the steered
model reproduces vanilla Qwen3-Omni bit for bit (tested).

The modules act on the packed token sequence through forward hooks on `thinker.visual.blocks[l]`
and never modify Qwen3-Omni weights. Items (images, video clips) are separated using the
`grid_thw` the vision encoder receives, so the same code steers video/cine inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomllm.omni.layout import tokens_per_item


class GatedCrossAttention(nn.Module):
    def __init__(self, dim: int, text_dim: int, num_heads: int = 16, query_norm: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim) if query_norm else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(text_dim, dim)
        self.v = nn.Linear(text_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """x [B, N, C] visual tokens (padded), text [B, L, Dt], text_mask [B, L] bool (True = keep)."""
        b, n, c = x.shape
        h = self.num_heads
        q = self.q(self.norm(x)).view(b, n, h, -1).transpose(1, 2)
        k = self.k(text).view(b, text.shape[1], h, -1).transpose(1, 2)
        v = self.v(text).view(b, text.shape[1], h, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=text_mask[:, None, None, :])
        return self.o(a.transpose(1, 2).reshape(b, n, c))

    @property
    def gate(self) -> torch.Tensor:
        return torch.tanh(self.alpha)


class VisualSteering(nn.Module):
    """Trainable steering modules + hooks into a frozen `thinker.visual`.

    Usage:
        steer = VisualSteering(thinker.visual, layers=(1, 3, ...), text_dim=1152)
        steer.attach()
        with steer.condition(text_tokens, text_mask):   # one row per image/video, batch order
            thinker.generate(...)                       # or thinker(...), visual(...)
    Outside `condition(...)` (or when `enabled=False`) the hooks are a no-op.
    """

    def __init__(self, visual: nn.Module, layers: tuple[int, ...], text_dim: int, num_heads: int = 16,
                 query_norm: bool = True) -> None:
        super().__init__()
        dim = visual.config.hidden_size
        depth = len(visual.blocks)
        bad = [l for l in layers if not 0 <= l < depth]
        if bad:
            raise ValueError(f"steering layers {bad} outside 0..{depth - 1}")
        self.layers = tuple(sorted(layers))
        self.blocks = nn.ModuleDict({str(l): GatedCrossAttention(dim, text_dim, num_heads, query_norm)
                                     for l in self.layers})
        self._visual = [visual]  # not a submodule: its weights are not ours to train or save
        self._handles: list = []
        self._text: tuple[torch.Tensor, torch.Tensor] | None = None
        self._lengths: list[int] | None = None
        self.enabled = True

    # -- hooks ------------------------------------------------------------------------------
    def attach(self) -> "VisualSteering":
        if self._handles:
            return self
        visual = self._visual[0]

        def pre(mod, args, kwargs):
            grid_thw = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
            self._lengths = tokens_per_item(grid_thw)

        self._handles.append(visual.register_forward_pre_hook(pre, with_kwargs=True))
        for l in self.layers:
            self._handles.append(visual.blocks[l].register_forward_hook(self._make_hook(str(l))))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, key: str):
        def hook(mod, inp, out):
            if not self.enabled or self._text is None:
                return None
            return self._steer(self.blocks[key], out)
        return hook

    def _steer(self, ca: GatedCrossAttention, x: torch.Tensor) -> torch.Tensor:
        text, mask = self._text
        lengths = self._lengths
        if len(lengths) != text.shape[0]:
            raise ValueError(f"{len(lengths)} visual items but {text.shape[0]} text conditions")
        items = torch.split(x, lengths)
        padded = nn.utils.rnn.pad_sequence(items, batch_first=True)  # [B, Nmax, C]
        delta = ca(padded.to(ca.q.weight.dtype), text.to(ca.q.weight.dtype), mask)
        delta = torch.cat([d[:n] for d, n in zip(delta, lengths)], dim=0)
        return x + ca.gate.to(x.dtype) * delta.to(x.dtype)

    # -- conditioning -----------------------------------------------------------------------
    class _Condition:
        def __init__(self, owner, text, mask):
            self.owner, self.text, self.mask = owner, text, mask

        def __enter__(self):
            self.owner._text = (self.text, self.mask.bool())
            return self.owner

        def __exit__(self, *exc):
            self.owner._text = None

    def condition(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> "_Condition":
        return self._Condition(self, text_tokens, text_mask)

    def gates(self) -> dict[int, float]:
        return {int(k): float(m.gate) for k, m in self.blocks.items()}


# -- build / save / load (shared by training and evaluation) ---------------------------------
DEFAULT_LAYERS = tuple(range(1, 27, 2))  # every other block, as in SteerViT (13 of 27)


def build_steering(thinker: nn.Module, layers: tuple[int, ...] = DEFAULT_LAYERS, text_source: str = "roberta",
                   text_model: str = "FacebookAI/roberta-large", tokenizer=None, device: str = "cuda"):
    """(text_encoder, steering) for a frozen thinker. Trainable weights are kept in float32 (the
    frozen backbone is NF4/BF16); steering is attached to `thinker.visual`."""
    from biomllm.omni.text_encoder import ClinicalTextEncoder

    dim = thinker.visual.config.hidden_size
    text = ClinicalTextEncoder(dim, source=text_source, model_id=text_model, thinker=thinker,
                               tokenizer=tokenizer).to(device)
    steer = VisualSteering(thinker.visual, layers=layers, text_dim=dim).to(device).attach()
    return text, steer


def save_steering(path, text, steer, meta: dict) -> None:
    torch.save({"meta": meta, "text_proj": text.proj.state_dict(), "steer": steer.state_dict()}, path)


def load_steering(path, thinker: nn.Module, tokenizer=None, device: str = "cuda"):
    ck = torch.load(path, map_location="cpu")
    m = ck["meta"]
    text, steer = build_steering(thinker, tuple(m["layers"]), m["text_source"], m["text_model"], tokenizer, device)
    text.proj.load_state_dict(ck["text_proj"])
    steer.load_state_dict(ck["steer"])
    return text.eval(), steer.eval(), m
