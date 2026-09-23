"""Training / inference cost tracking: the x-axis of the paper's Pareto plots."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from biomllm.training.param_groups import param_summary


class CostTracker:
    """Records trainable parameters, wall-clock GPU hours and peak GPU memory of a run.

        with CostTracker(model) as cost:
            train(...)
        cost.dump(out_dir / "cost.json")
    """

    def __init__(self, model, num_gpus: int | None = None):
        self.model = model
        self.num_gpus = num_gpus if num_gpus is not None else max(torch.cuda.device_count(), 1)
        self.seconds = 0.0
        self.peak_mem_gb = 0.0

    def __enter__(self) -> CostTracker:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if torch.cuda.is_available():
            self.peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    def report(self) -> dict:
        params = param_summary(self.model)
        return {
            "params": params,
            "trainable_params": sum(v["trainable"] for v in params.values()),
            "gpu_hours": self.seconds * self.num_gpus / 3600,
            "wall_seconds": self.seconds,
            "peak_mem_gb": self.peak_mem_gb,
        }

    def dump(self, path: str | Path) -> dict:
        report = self.report()
        Path(path).write_text(json.dumps(report, indent=2))
        return report
