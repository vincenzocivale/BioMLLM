import torch

from biomllm.models.types import TaskQueries


def drop_native(queries: TaskQueries, p: float, training: bool) -> TaskQueries:
    """Per-sample Bernoulli drop of the F^MLLM term inside the task tokens:
    T = e_task + lambda * F^MLLM (+ alpha * P(F^S)), lambda ~ Bernoulli(1 - p), training only.

    Needs the adapter to expose `queries.native`; a no-op otherwise or at inference.
    """
    if not training or p <= 0.0 or queries.native is None:
        return queries
    b = queries.tokens.shape[0]
    keep = (torch.rand(b, 1, 1, device=queries.tokens.device) >= p).to(queries.tokens.dtype)
    tokens = queries.tokens - (1.0 - keep) * queries.native
    return TaskQueries(tokens, queries.grid, queries.native * keep, queries.num_prefix)
