"""Lightweight box-prompted mask decoder on FROZEN Qwen3-Omni visual feature maps.

Purpose: read out spatial information already present in the foundation model, not become a new
visual backbone (see docs/omni_segmentation_design.md). Hence:

  * the only image input is Qwen3-Omni features (`VisualTaps` maps; no raw-pixel skip path unless
    `pixel_skip=True`, an explicitly flagged ablation that adds a visual pathway);
  * ~1-2 M parameters (vs 31.7 B in the Thinker);
  * the box prompt (GT box = representation oracle, Omni-predicted box = end-to-end) enters as a
    rasterised box channel plus a Fourier embedding of its coordinates (FiLM).

Input maps are taken at the patch grid (h, w) = image / 16. Merged-grid maps (h/2, w/2: deepstack,
merged) are upsampled to the patch grid. Output: mask logits at 4x the patch grid (stride 4 px),
to be bilinearly resized to the image.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_channel(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """boxes [B, 4] normalised xyxy -> [B, 1, h, w] soft box mask (pixel-centre inclusion)."""
    ys = (torch.arange(h, device=boxes.device, dtype=boxes.dtype) + 0.5) / h
    xs = (torch.arange(w, device=boxes.device, dtype=boxes.dtype) + 0.5) / w
    x1, y1, x2, y2 = boxes.unbind(-1)
    inx = (xs[None] >= x1[:, None]) & (xs[None] <= x2[:, None])
    iny = (ys[None] >= y1[:, None]) & (ys[None] <= y2[:, None])
    return (iny[:, :, None] & inx[:, None, :]).to(boxes.dtype)[:, None]


class FourierBox(nn.Module):
    def __init__(self, dim: int, n_freq: int = 16) -> None:
        super().__init__()
        self.register_buffer("freq", 2 ** torch.arange(n_freq).float() * math.pi, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(4 * 2 * n_freq, dim), nn.GELU(), nn.Linear(dim, 2 * dim))

    def forward(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = boxes[..., None] * self.freq
        emb = torch.cat([a.sin(), a.cos()], -1).flatten(1)
        scale, shift = self.mlp(emb).chunk(2, -1)
        return scale[:, :, None, None], shift[:, :, None, None]


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class OmniMaskDecoder(nn.Module):
    def __init__(self, in_dims: dict[str, int], dim: int = 128, pixel_skip: bool = False) -> None:
        super().__init__()
        self.keys = list(in_dims)
        self.proj = nn.ModuleDict({k.replace(".", "_"): nn.Sequential(nn.LayerNorm(c), nn.Linear(c, dim))
                                   for k, c in in_dims.items()})
        self.box = FourierBox(dim)
        self.fuse = nn.Sequential(ConvBlock(dim + 1, dim), ConvBlock(dim, dim))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GELU(), ConvBlock(dim // 2, dim // 2))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2), nn.GELU(),
                                 ConvBlock(dim // 4 + (8 if pixel_skip else 0), dim // 4))
        self.pixel_skip = (nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU()) if pixel_skip else None)
        self.head = nn.Conv2d(dim // 4, 1, 1)

    def forward(self, maps: dict[str, torch.Tensor], boxes: torch.Tensor,
                pixels: torch.Tensor | None = None) -> torch.Tensor:
        """maps[k] [B, C_k, h_k, w_k] (float), boxes [B, 4] normalised xyxy -> logits [B, 1, 4h, 4w]."""
        h = max(m.shape[-2] for m in maps.values())
        w = max(m.shape[-1] for m in maps.values())
        x = 0
        for k in self.keys:
            m = maps[k].float()
            if m.shape[-2:] != (h, w):
                m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
            x = x + self.proj[k.replace(".", "_")](m.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        scale, shift = self.box(boxes.float())
        x = x * (1 + scale) + shift
        x = self.fuse(torch.cat([x, box_channel(boxes.float(), h, w)], 1))
        x = self.up2[:2](self.up1(x))
        if self.pixel_skip is not None:
            if pixels is None:
                raise ValueError("pixel_skip=True needs the input pixels")
            p = self.pixel_skip(F.interpolate(pixels.float(), size=x.shape[-2:], mode="bilinear", align_corners=False))
            x = torch.cat([x, p], 1)
        return self.head(self.up2[2:](x))
