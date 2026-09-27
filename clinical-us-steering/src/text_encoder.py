"""Frozen text encoder; query token features are computed once and cached."""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    def __init__(self, name: str, max_length: int = 32):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """-> tokens (1, L, D) float32, mask (1, L) bool. An empty string keeps the special tokens."""
        if text not in self._cache:
            dev = next(self.model.parameters()).device
            t = self.tokenizer([text], return_tensors="pt", truncation=True,
                               max_length=self.max_length).to(dev)
            out = self.model(**t).last_hidden_state.float()
            self._cache[text] = (out, t["attention_mask"].bool())
        return self._cache[text]

    def _apply(self, fn, *args, **kwargs):  # device moves invalidate the cache
        self._cache = {}
        return super()._apply(fn, *args, **kwargs)


class RandomTextEncoder(nn.Module):
    """Deterministic stand-in (unit tests): hashes words to fixed random vectors."""

    def __init__(self, dim: int = 32, max_length: int = 16):
        super().__init__()
        self.name, self.dim, self.max_length = "random", dim, max_length
        self.table = nn.Embedding(1000, dim)
        self.table.requires_grad_(False)

    def encode(self, text: str):
        ids = [0] + [sum(map(ord, w)) % 999 + 1 for w in text.split()][: self.max_length - 1]
        ids = torch.tensor([ids], device=self.table.weight.device)
        return self.table(ids), torch.ones_like(ids, dtype=torch.bool)
