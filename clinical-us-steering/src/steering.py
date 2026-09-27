"""Shared gated text->vision cross-attention controller (SteerViT-style) and the full model.

For every steered layer l:  V'_l = V_l + tanh(alpha_l) * CrossAttn(q=V_l, k=v=T)
with alpha_l initialised at 0, so the untrained model equals the frozen backbone.
One controller serves every query: there are no per-attribute parameters except the heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from attribute_heads import AttributeHeads


class GatedCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, v, t, t_mask, record: dict | None = None):
        s, w = self.attn(self.norm(v), t, t, key_padding_mask=~t_mask,
                         need_weights=record is not None, average_attn_weights=True)
        update = torch.tanh(self.alpha) * s
        if record is not None:
            record["rel_update"] = (update.norm(dim=-1) / v.norm(dim=-1).clamp(min=1e-6)).mean().item()
            record["attn"] = w.detach()
        return v + update


class SteeringController(nn.Module):
    def __init__(self, dim: int, text_dim: int, layers: list[int], num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.layers = sorted(layers)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(), nn.Linear(dim, dim),
                                       nn.LayerNorm(dim))
        self.blocks = nn.ModuleDict({str(l): GatedCrossAttention(dim, num_heads, dropout)
                                     for l in self.layers})

    def gates(self) -> dict[int, float]:
        return {l: torch.tanh(self.blocks[str(l)].alpha).item() for l in self.layers}

    def make_hook(self, text_tokens, text_mask, batch: int, records: dict | None = None):
        t = self.text_proj(text_tokens).expand(batch, -1, -1)
        m = text_mask.expand(batch, -1)

        def hook(i, h):
            if str(i) not in self.blocks:
                return h
            rec = None
            if records is not None:
                rec = records.setdefault(i, {})
            return self.blocks[str(i)](h, t.to(h.dtype), m, rec)

        return hook


class ClinicalSteeringModel(nn.Module):
    """Frozen backbone + frozen text encoder + (optional) shared controller + attribute heads.

    steering=False is the no-steering baseline: z = pool(E(I)) for every attribute.
    """

    def __init__(self, backbone, text_encoder, attributes: dict, queries: dict,
                 steering: bool, steer_cfg: dict, head_dropout: float = 0.1):
        super().__init__()
        self.backbone, self.text_encoder = backbone, text_encoder
        self.queries = dict(queries)
        self.controller = (SteeringController(backbone.dim, text_encoder.dim, steer_cfg["layers"],
                                              steer_cfg["num_heads"], steer_cfg.get("dropout", 0.0))
                           if steering else None)
        self.heads = AttributeHeads(backbone.dim, attributes, head_dropout)

    def trainable_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(("controller.", "heads."))}

    def encode(self, x: torch.Tensor, texts: list[str | None], records: dict | None = None
               ) -> dict[str | None, torch.Tensor]:
        """Pooled embedding per query text. None (or no controller) = unsteered backbone.

        Blocks before the first steered layer are shared by all queries.
        """
        bb = self.backbone
        n = len(bb.blocks)
        first = self.controller.layers[0] if self.controller is not None else n - 1
        with torch.no_grad():
            prefix = bb.run_blocks(bb.embed(x), 0, first + 1)
        out = {}
        for text in texts:
            if text is None or self.controller is None:
                if None not in out:
                    with torch.no_grad():
                        out[None] = bb.pool(bb.run_blocks(prefix, first + 1, n))
                out[text] = out[None]
                continue
            tok, mask = self.text_encoder.encode(text)
            rec = records.setdefault(text, {}) if records is not None else None
            hook = self.controller.make_hook(tok, mask, x.shape[0], rec)
            h = hook(first, prefix)
            out[text] = bb.pool(bb.run_blocks(h, first + 1, n, hook))
        return out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits per supervised attribute, each from its own query's representation."""
        attrs = self.heads.attributes
        z = self.encode(x, [self.queries[a] for a in attrs])
        return {a: self.heads(a, z[self.queries[a]]) for a in attrs}
