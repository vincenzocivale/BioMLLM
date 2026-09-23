"""Freeze policies and parameter accounting.

Policies (configs/train/*.yaml, key `policy`):
    frozen  the whole MLLM is frozen except its task parameters (e_task, heads). Default:
            the base model stays bit-identical, so general abilities cannot be forgotten.
    lora    low-rank adapters on the LLM (phase 2 extension; not implemented yet).
    full    everything trainable except the native vision encoder (costly reference).
Conditioner (projector, gate, static map) is always trainable; the expert is always frozen.

Parameter counts matter for the paper: every run logs this breakdown next to its cost.
"""

from __future__ import annotations

import torch.nn as nn

POLICIES = ("frozen", "lora", "full")


def apply_freeze_policy(model, policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"unknown freeze policy '{policy}', available: {POLICIES}")
    mllm = model.mllm
    if policy == "lora":
        raise NotImplementedError("train.policy=lora is a phase-2 extension (peft adapters)")
    for p in mllm.parameters():
        p.requires_grad_(policy == "full")
    if policy == "full":
        mllm.freeze_native()
    for p in mllm.task_parameters().values():
        p.requires_grad_(True)


def count_params(module: nn.Module | None, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def param_summary(model) -> dict[str, dict[str, int]]:
    """Total / trainable parameters for the base MLLM, its task parameters, the expert and
    the conditioner (projector + gate + static map)."""
    task = model.mllm.task_parameters()
    task_ids = {id(p) for p in task.values()}
    base = [p for p in model.mllm.parameters() if id(p) not in task_ids]
    static = [model.static_features] if model.static_features is not None else []
    cond = list(model.conditioner.parameters()) + static if model.conditioner else static

    def stats(params):
        return {"total": sum(p.numel() for p in params),
                "trainable": sum(p.numel() for p in params if p.requires_grad)}

    return {
        "mllm_base": stats(base),
        "task_params": stats(list(task.values())),
        "expert": stats(list(model.expert.parameters()) if model.expert is not None else []),
        "conditioner": stats(cond),
    }


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
