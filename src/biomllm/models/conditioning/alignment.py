import torch
import torch.nn.functional as F

from biomllm.models.types import FeatureMap


def resample_to_grid(feats: FeatureMap, grid: tuple[int, int], mode: str = "area") -> FeatureMap:
    """Bring expert features onto the MLLM patch grid so that index i refers to the same
    image region in F_i^MLLM and F_i^S.

    `area` averages when downsampling (the common case: experts often run at higher
    resolution than the MLLM grid) and falls back to bilinear when upsampling.
    """
    if tuple(feats.grid) == tuple(grid):
        return feats
    x = feats.as_image()
    upsampling = grid[0] > feats.grid[0] or grid[1] > feats.grid[1]
    if mode == "area" and not upsampling:
        x = F.adaptive_avg_pool2d(x, grid)
    else:
        x = F.interpolate(x, size=grid, mode="bilinear", align_corners=False)
    return FeatureMap.from_image(x)


def global_pool(feats: FeatureMap) -> torch.Tensor:
    """[B, 1, C] mean over patches, used when task tokens have no spatial index."""
    return feats.tokens.mean(dim=1, keepdim=True)
