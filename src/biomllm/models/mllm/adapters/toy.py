"""Tiny self-contained TaskTokenMLLM for tests and the debug experiment.

Mirrors the structure the real adapters must follow: a vision encoder producing F^MLLM,
spatial task queries e_task + F_i^MLLM, a small transformer standing in for the LLM, and
seg / box heads reading the task-token hidden states.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from biomllm.models.mllm.base import TaskTokenMLLM
from biomllm.models.types import FeatureMap, MLLMOutput, TaskQueries


class ToyMLLM(TaskTokenMLLM):
    tasks = ("seg", "box")

    def __init__(self, dim: int = 64, image_size: int = 64, patch_size: int = 8,
                 depth: int = 2, num_heads: int = 4, spatial_queries: bool = True):
        super().__init__()
        self.query_dim = self.hidden_dim = self.visual_dim = dim
        self.image_size = image_size
        self.spatial_queries = spatial_queries
        self.vision = nn.Conv2d(3, dim, patch_size, patch_size)
        self.task_embed = nn.ParameterDict({t: nn.Parameter(torch.randn(dim) * 0.02)
                                            for t in self.tasks})
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.llm = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.mask_embed = nn.Linear(dim, dim)
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))

    def visual_features(self, images: torch.Tensor) -> FeatureMap:
        if images.shape[-1] != self.image_size:
            images = nn.functional.interpolate(images, size=(self.image_size,) * 2,
                                               mode="bilinear", align_corners=False)
        return FeatureMap.from_image(self.vision(images))

    def build_task_queries(self, task: str, visual: FeatureMap, batch: dict[str, Any]) -> TaskQueries:
        e_task = self.task_embed[task]
        if self.spatial_queries:
            return TaskQueries(visual.tokens + e_task, grid=visual.grid, native=visual.tokens)
        # a single [SEG]/[BOX]-like token
        b = visual.tokens.shape[0]
        return TaskQueries(e_task.expand(b, 1, -1), grid=None)

    def llm_forward(self, task: str, visual: FeatureMap, queries: TaskQueries,
                    batch: dict[str, Any]) -> MLLMOutput:
        n_vis = visual.tokens.shape[1]
        h = self.llm(torch.cat([visual.tokens, queries.tokens], dim=1))
        return MLLMOutput(task_hidden=TaskQueries(h[:, n_vis:], grid=queries.grid))

    def decode(self, task: str, task_hidden: TaskQueries, visual: FeatureMap,
               batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        pooled = task_hidden.tokens.mean(dim=1)
        if task == "seg":
            logits = torch.einsum("bd,bnd->bn", self.mask_embed(pooled), visual.tokens)
            return {"mask_logits": logits.view(-1, 1, *visual.grid)}
        if task == "box":
            return {"boxes": self.box_head(pooled).sigmoid()}  # cxcywh, normalised
        raise KeyError(task)

    def freeze_native(self) -> None:
        for p in self.vision.parameters():
            p.requires_grad_(False)

    def task_parameters(self) -> dict[str, nn.Parameter]:
        params = {f"task_embed.{k}": v for k, v in self.task_embed.items()}
        for prefix, module in (("mask_embed", self.mask_embed), ("box_head", self.box_head)):
            params.update({f"{prefix}.{k}": v for k, v in module.named_parameters()})
        return params
