"""Backbone-agnostic interface for MLLMs with dedicated task tokens.

An adapter wraps one concrete MLLM (LISA, GLaMM, VisionLLM-v2-like, ...) and exposes four
hooks. The conditioner plugs into the output of `visual_features` (native injection), of
`build_task_queries` (pre-LLM injection) or into the task-token hidden states before `decode`
(post-LLM injection), so it never needs to know which backbone is underneath.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from biomllm.models.types import FeatureMap, MLLMOutput, TaskQueries


class TaskTokenMLLM(nn.Module):
    #: Task names this adapter can decode ("seg", "box", "det").
    tasks: tuple[str, ...] = ()
    #: Width of the task tokens (the D of T_i) at each injection point.
    query_dim: int
    hidden_dim: int
    #: Channel width of F^MLLM (the projector input for the `self` condition).
    visual_dim: int

    def visual_features(self, images: torch.Tensor) -> FeatureMap:
        """F^MLLM from the native (generalist) vision encoder, on its patch grid."""
        raise NotImplementedError

    def build_task_queries(self, task: str, visual: FeatureMap, batch: dict[str, Any]) -> TaskQueries:
        """T = e_task + F^MLLM (or the backbone's equivalent) before entering the LLM.
        Return grid=None when the queries are not indexed by the patch grid."""
        raise NotImplementedError

    def llm_forward(self, task: str, visual: FeatureMap, queries: TaskQueries,
                    batch: dict[str, Any], extra_visual: torch.Tensor | None = None) -> MLLMOutput:
        """Run the language model; return the hidden states of the task tokens.

        `visual` is the image context the LLM must read: with native injection it differs from
        what the native encoder produced, so adapters must feed these tokens to the LLM rather
        than re-encoding the image. `extra_visual` [B, n, D] (native prepend) are extra image
        tokens placed right after the native ones."""
        raise NotImplementedError

    def decode(self, task: str, task_hidden: TaskQueries, visual: FeatureMap,
               batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Task heads: masks / boxes / detections from the task-token hidden states."""
        raise NotImplementedError

    def freeze_native(self) -> None:
        """Freeze the native vision encoder (F^MLLM must stay intact)."""
        raise NotImplementedError

    def task_parameters(self) -> dict[str, nn.Parameter]:
        """Parameters added for the task tokens (e_task embeddings, mask / box heads).

        They are trainable even when the MLLM is frozen, and belong to a modality package
        rather than to the base model.
        """
        raise NotImplementedError
