"""Intermediate visual representations of the Qwen3-Omni vision encoder, as spatial maps.

Reads the tensors the official forward already computes (forward hooks, no re-implementation):

    blocks[l]        output of `thinker.visual.blocks[l]`      [t, 1152, h,   w  ]  patch grid
    deepstack[k]     `thinker.visual.merger_list[k]` output     [t, 2048, h/2, w/2]  (blocks 8/16/24)
    merged           `thinker.visual.merger` output             [t, 2048, h/2, w/2]  = LLM image tokens

one entry per image/video in the batch (packed order of `image_grid_thw`).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from biomllm.omni.layout import block_tokens_to_grid, merged_tokens_to_grid, split_items


class VisualTaps:
    """Context manager capturing selected vision-encoder intermediates during ANY forward that
    runs `visual` (a bare `visual(...)` call, `thinker.forward` or `thinker.generate`)."""

    def __init__(self, visual: nn.Module, blocks: tuple[int, ...] = (), deepstack: bool = True,
                 merged: bool = True) -> None:
        self.visual, self.blocks, self.want_ds, self.want_merged = visual, tuple(blocks), deepstack, merged
        self.merge = visual.spatial_merge_size
        self._handles = []
        self.grid_thw: torch.Tensor | None = None
        self.raw: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "VisualTaps":
        def pre(mod, args, kwargs):
            self.grid_thw = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)

        self._handles.append(self.visual.register_forward_pre_hook(pre, with_kwargs=True))
        for l in self.blocks:
            self._handles.append(self.visual.blocks[l].register_forward_hook(self._store(f"blocks.{l}")))
        if self.want_ds:
            for k, m in enumerate(self.visual.merger_list):
                self._handles.append(m.register_forward_hook(self._store(f"deepstack.{k}")))
        if self.want_merged:
            self._handles.append(self.visual.merger.register_forward_hook(self._store("merged")))
        return self

    def _store(self, key: str):
        def hook(mod, inp, out):
            # generate() runs the ViT once (prefill); keep the first capture only
            self.raw.setdefault(key, out)
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def maps(self) -> list[dict[str, torch.Tensor]]:
        """Per item: {"blocks.l": [t,C,h,w], "deepstack.k" / "merged": [t,C',h/2,w/2]}."""
        if self.grid_thw is None:
            raise RuntimeError("the vision encoder did not run inside the context")
        grids = [tuple(g) for g in self.grid_thw.tolist()]
        out = [dict() for _ in grids]
        for key, x in self.raw.items():
            merged = not key.startswith("blocks.")
            for i, (tok, thw) in enumerate(zip(split_items(x, self.grid_thw, self.merge if merged else 1), grids)):
                out[i][key] = (merged_tokens_to_grid(tok, thw, self.merge) if merged
                               else block_tokens_to_grid(tok, thw, self.merge))
        return out


@contextmanager
def visual_taps(visual: nn.Module, **kw):
    taps = VisualTaps(visual, **kw)
    with taps:
        yield taps
