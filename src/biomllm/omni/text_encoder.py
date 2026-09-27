"""Clinical-text tokens that condition the visual steering, independent of the Omni graph.

Qwen3-Omni has no CLIP-style standalone text tower, so the text source is pluggable:

  source="roberta"   frozen contextual encoder (default FacebookAI/roberta-large, as in SteerViT)
  source="omni"      frozen hidden states of the Qwen3-Omni Thinker itself at layer `omni_layer`
                     (the Omni-native language representation; ablation)

Both give token-level features Z_t [B, L, d_t]; as in SteerViT they are L2-normalised and passed
through a trainable 2-layer MLP to the visual width (H_t [B, L, d_v]).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalTextEncoder(nn.Module):
    def __init__(self, out_dim: int, source: str = "roberta", model_id: str = "FacebookAI/roberta-large",
                 thinker: nn.Module | None = None, tokenizer=None, omni_layer: int = 24,
                 max_length: int = 256) -> None:
        super().__init__()
        self.source, self.max_length = source, max_length
        if source == "roberta":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._frozen = [AutoModel.from_pretrained(model_id).eval().requires_grad_(False)]
            in_dim = self._frozen[0].config.hidden_size
        elif source == "omni":
            if thinker is None or tokenizer is None:
                raise ValueError("source='omni' needs the frozen thinker and its tokenizer")
            self.tokenizer, self._frozen, self.omni_layer = tokenizer, [thinker], omni_layer
            in_dim = thinker.config.text_config.hidden_size
        else:
            raise ValueError(f"unknown text source {source!r}")
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def to(self, *args, **kwargs):  # the frozen encoder is kept out of the module tree
        if self.source == "roberta":
            self._frozen[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @torch.no_grad()
    def frozen_features(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Z_t [B, L, d_t] (float32) and mask [B, L]. Cached per text: guidelines repeat."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            enc = self._frozen[0]
            dev = next(enc.parameters()).device
            tok = self.tokenizer(missing, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(dev)
            if self.source == "roberta":
                z = enc(**tok).last_hidden_state
            else:
                out = enc.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                output_hidden_states=True)
                z = out.hidden_states[self.omni_layer]
            for i, t in enumerate(missing):
                m = tok["attention_mask"][i].bool()
                self._cache[t] = (z[i][m].float().cpu(), m[m].cpu())
        feats = [self._cache[t][0] for t in texts]
        z = nn.utils.rnn.pad_sequence(feats, batch_first=True)
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(f), dtype=torch.bool) for f in feats], batch_first=True)
        return z, mask

    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        z, mask = self.frozen_features(texts)
        dev = self.proj[0].weight.device
        z = F.normalize(z.to(dev, self.proj[0].weight.dtype), dim=-1)
        return self.proj(z), mask.to(dev)
