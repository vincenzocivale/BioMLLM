"""Frozen ViT backbones exposing their transformer blocks, so steering can run between them."""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenViT(nn.Module):
    """Uniform interface: embed -> blocks[i] (hook after each) -> final norm -> pool."""

    def __init__(self, patch_embed_fn, blocks, norm, num_prefix: int, dim: int, pool: str,
                 name: str, modules: nn.Module):
        super().__init__()
        self.model = modules
        self._embed = patch_embed_fn
        self.blocks = blocks
        self.norm = norm
        self.num_prefix, self.dim, self.pool_mode, self.name = num_prefix, dim, pool, name
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # the backbone never leaves eval mode
        return super().train(False)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._embed(x)

    def run_blocks(self, h, start: int, end: int, hook=None):
        for i in range(start, end):
            h = self.blocks[i](h)
            if hook is not None:
                h = hook(i, h)
        return h

    def pool(self, h: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        h = self.norm(h)
        mode = mode or self.pool_mode
        if mode == "cls":
            return h[:, 0]
        if mode == "mean":
            return h[:, self.num_prefix:].mean(1)
        raise ValueError(mode)

    def forward(self, x, hook=None):
        return self.pool(self.run_blocks(self.embed(x), 0, len(self.blocks), hook))


def build_backbone(cfg: dict, pretrained: bool = True) -> FrozenViT:
    if cfg["kind"] == "timm_vit":
        import timm
        from huggingface_hub import hf_hub_download

        m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=0,
                              img_size=cfg["image_size"])
        if pretrained:
            sd = torch.load(hf_hub_download(cfg["hf_repo"], cfg["hf_file"]), map_location="cpu",
                            weights_only=True)
            sd = sd.get("model", sd)
            sd = {k: v for k, v in sd.items() if not k.startswith(("decoder", "mask_token"))}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"backbone weights mismatch: {missing[:5]} {unexpected[:5]}")

        def embed(x):
            return m.norm_pre(m._pos_embed(m.patch_embed(x)))

        return FrozenViT(embed, m.blocks, m.norm, m.num_prefix_tokens, m.embed_dim, cfg["pool"],
                         cfg.get("hf_repo", cfg["timm_name"]), m)
    if cfg["kind"] == "hf_dinov2":
        from transformers import Dinov2Config, Dinov2Model

        m = (Dinov2Model.from_pretrained(cfg["hf_repo"]) if pretrained
             else Dinov2Model(Dinov2Config.from_pretrained(cfg["hf_repo"])))
        return FrozenViT(m.embeddings, m.encoder.layer, m.layernorm, 1, m.config.hidden_size,
                         cfg["pool"], cfg["hf_repo"], m)
    if cfg["kind"] == "tiny_random":  # unit tests only
        import timm

        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0,
                              img_size=cfg["image_size"], depth=cfg.get("depth", 4))

        def embed(x):
            return m.norm_pre(m._pos_embed(m.patch_embed(x)))

        return FrozenViT(embed, m.blocks, m.norm, m.num_prefix_tokens, m.embed_dim, cfg["pool"],
                         "tiny_random", m)
    raise ValueError(cfg["kind"])
