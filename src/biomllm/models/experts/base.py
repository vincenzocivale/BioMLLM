from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomllm.models.types import FeatureMap


class FrozenExpert(nn.Module):
    """Wraps a foundation model as a frozen feature extractor E(I) -> F^S.

    Each expert owns its preprocessing (resolution, normalisation), so the model can be fed
    the same raw images in [0, 1] as the MLLM. Subclasses implement `extract`.
    """

    def __init__(self, dim: int, image_size: int | tuple[int, int],
                 mean: tuple[float, ...], std: tuple[float, ...], name: str = ""):
        super().__init__()
        self.dim = dim
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.name = name
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def freeze(self) -> FrozenExpert:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode: bool = True) -> FrozenExpert:
        # Always stays in eval mode (dropout / norm statistics frozen).
        return super().train(False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] in [0, 1]."""
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear",
                                   align_corners=False, antialias=True)
        return (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> FeatureMap:
        return self.extract(self.preprocess(images))
