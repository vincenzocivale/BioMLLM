"""Token layout of the Qwen3-Omni vision encoder (verified in docs/qwen3_omni_architecture.md).

The processor (`Qwen2VLImageProcessor.patchify`) flattens each image/video into patches in
*merge-window-major* order: for a grid (t, h, w) of 16x16 patches and merge size m=2 the
sequence index runs over (t, h/m, w/m, m, m). Every `thinker.visual.blocks[i]` works on this
packed sequence, all images of a batch concatenated along dim 0 (boundaries = `cu_seqlens`).

The patch merger groups each contiguous run of m*m tokens into one LLM token, so the merged
sequence (`pooler_output`, `deepstack_features[k]`) is plain raster order over (t, h/m, w/m).
"""

from __future__ import annotations

import torch


def tokens_per_item(grid_thw: torch.Tensor, merge: int = 1) -> list[int]:
    """Sequence length of each image/video in the packed sequence (merge=2 after the merger)."""
    return (grid_thw.prod(-1) // (merge * merge)).tolist()


def split_items(tokens: torch.Tensor, grid_thw: torch.Tensor, merge: int = 1) -> list[torch.Tensor]:
    """Split a packed [N, C] sequence into one tensor per image/video."""
    return list(torch.split(tokens, tokens_per_item(grid_thw, merge)))


def block_tokens_to_grid(tokens: torch.Tensor, thw: tuple[int, int, int], merge: int) -> torch.Tensor:
    """[t*h*w, C] vision-block tokens of ONE item -> [t, C, h, w] spatial map (patch grid)."""
    t, h, w = (int(x) for x in thw)
    c = tokens.shape[-1]
    x = tokens.reshape(t, h // merge, w // merge, merge, merge, c)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(t, c, h, w)


def grid_to_block_tokens(x: torch.Tensor, merge: int) -> torch.Tensor:
    """Inverse of `block_tokens_to_grid`: [t, C, h, w] -> [t*h*w, C] merge-window-major."""
    t, c, h, w = x.shape
    x = x.reshape(t, c, h // merge, merge, w // merge, merge)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(t * h * w, c)


def merged_tokens_to_grid(tokens: torch.Tensor, thw: tuple[int, int, int], merge: int) -> torch.Tensor:
    """[t*(h/m)*(w/m), C] merged tokens of ONE item -> [t, C, h/m, w/m] (raster order)."""
    t, h, w = (int(x) for x in thw)
    c = tokens.shape[-1]
    return tokens.reshape(t, h // merge, w // merge, c).permute(0, 3, 1, 2)


def pixel_patches_to_image(pixel_values: torch.Tensor, thw: tuple[int, int, int], patch: int,
                           merge: int, temporal: int, channels: int = 3) -> torch.Tensor:
    """Undo the processor's patchify for ONE item: [t*h*w, C*T*p*p] -> [t*T, C, h*p, w*p].

    Used to test that our reading of the token order matches the official processor."""
    t, h, w = (int(x) for x in thw)
    x = pixel_values.reshape(t, h // merge, w // merge, merge, merge, channels, temporal, patch, patch)
    # -> [t, T, C, h/m, m, p, w/m, m, p]
    x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    return x.reshape(t * temporal, channels, h * patch, w * patch)
