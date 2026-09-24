"""Qwen3-VL adapter: one [MASK] task token per patch, STAMP-style dense segmentation.

The native LLM is frozen and run in two passes that reuse Qwen3-VL's own tested code
paths (`Qwen3VLModel.forward`), rather than hand-rolling M-RoPE:

  1. prefill  -- the real image + a short fixed instruction, through the official
     `input_ids` + `pixel_values` path. This computes and caches `rope_deltas`, exactly
     as the first step of `generate()` does.
  2. task step -- the Q = h*w task tokens (T_i = e_task + F_i^MLLM, continuous embeddings,
     not real token ids) are appended via `inputs_embeds` with the prefill's KV cache.
     `Qwen3VLModel` falls back to its cached `rope_deltas` for `inputs_embeds`-only calls,
     which is exactly the decode-step continuation logic already used for generation.

The prompt text and its token-type layout only depend on `image_size` (fixed per model),
so they are tokenized once in `__init__` and reused for every batch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomllm.models.mllm.base import TaskTokenMLLM
from biomllm.models.types import FeatureMap, MLLMOutput, TaskQueries

IMAGENET_MEAN = (0.485, 0.456, 0.406)


class QwenVLAdapter(TaskTokenMLLM):
    tasks = ("seg", "vqa")

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", image_size: int = 512,
                dtype: str = "bfloat16", prompt: str = "Segment the relevant structures.",
                perc_grid: tuple[int, int] = (4, 4)):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, dtype))

        cfg = self.qwen.config
        self.patch_size = cfg.vision_config.patch_size
        self.merge = cfg.vision_config.spatial_merge_size
        if image_size % (self.patch_size * self.merge):
            raise ValueError(f"image_size must be a multiple of {self.patch_size * self.merge}")
        self.image_size = image_size
        side = image_size // (self.patch_size * self.merge)
        self.grid = (side, side)
        self.query_dim = self.hidden_dim = cfg.text_config.hidden_size
        self.visual_dim = cfg.vision_config.out_hidden_size
        if self.visual_dim != self.hidden_dim:
            raise ValueError("expected the vision projector output to match the LLM embedding "
                             f"dim, got {self.visual_dim} vs {self.hidden_dim}")

        tok = self.processor.tokenizer
        self.tokenizer = tok
        image_token = getattr(tok, "image_token", "<|image_pad|>")
        self.image_token_id = getattr(tok, "image_token_id", None) or tok.convert_tokens_to_ids(image_token)
        n_img_tok = side * side
        image_block = f"<|vision_start|>{image_token * n_img_tok}<|vision_end|>"
        self.image_grid_thw_value = torch.tensor([[1, side * self.merge, side * self.merge]])
        self.register_buffer("image_grid_thw", self.image_grid_thw_value, persistent=False)

        # seg: task tokens follow a fixed instruction baked into the same prefill as the image.
        prompt_ids = tok(image_block + prompt, return_tensors="pt")["input_ids"]
        self.register_buffer("prompt_ids", prompt_ids, persistent=False)
        self.register_buffer("mm_token_type_ids", (prompt_ids == self.image_token_id).long(), persistent=False)

        # vqa: prefill is image-only (no baked-in instruction); the question is per-example
        # text appended after the [PERC] task tokens (see `_vqa_forward`).
        img_ids = tok(image_block, return_tensors="pt")["input_ids"]
        self.register_buffer("image_prefix_ids", img_ids, persistent=False)
        self.register_buffer("image_mm_token_type_ids", (img_ids == self.image_token_id).long(),
                             persistent=False)
        self.perc_grid = tuple(perc_grid)

        self.task_embed = nn.ParameterDict({t: nn.Parameter(torch.randn(self.hidden_dim) * 0.02)
                                            for t in self.tasks})
        self.mask_head = nn.Linear(self.hidden_dim, 1)
        self._cache: dict[str, torch.Tensor] = {}

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if tuple(images.shape[-2:]) != (self.image_size,) * 2:
            images = F.interpolate(images, size=(self.image_size,) * 2, mode="bilinear",
                                   align_corners=False, antialias=True)
        return images

    def visual_features(self, images: torch.Tensor) -> FeatureMap:
        imgs = self._preprocess(images)
        np_imgs = [im.permute(1, 2, 0).float().cpu().numpy() for im in imgs]
        proc = self.processor.image_processor(images=np_imgs, do_rescale=False, do_resize=False,
                                              return_tensors="pt")
        pixel_values = proc["pixel_values"].to(imgs.device, self.qwen.dtype)
        grid_thw = proc["image_grid_thw"].to(imgs.device)
        self._cache = {"pixel_values": pixel_values, "image_grid_thw": grid_thw}

        b = imgs.shape[0]
        out = self.qwen.model.get_image_features(pixel_values, image_grid_thw=grid_thw, return_dict=True)
        tokens = torch.cat(list(out.pooler_output), dim=0).to(imgs.dtype).view(b, -1, self.visual_dim)
        return FeatureMap(tokens, self.grid)

    def build_task_queries(self, task: str, visual: FeatureMap, batch: dict[str, Any]) -> TaskQueries:
        if task not in self.tasks:
            raise KeyError(task)
        e_task = self.task_embed[task]
        if task == "seg":
            return TaskQueries(visual.tokens + e_task, grid=visual.grid, native=visual.tokens)
        # vqa: [PERC] tokens on a coarser grid than the native visual tokens (README), average
        # -pooled from F^MLLM so they summarise context rather than repeat every patch.
        pooled = F.adaptive_avg_pool2d(visual.as_image(), self.perc_grid)
        tokens = FeatureMap.from_image(pooled).tokens
        return TaskQueries(tokens + e_task, grid=self.perc_grid, native=tokens)

    def llm_forward(self, task: str, visual: FeatureMap, queries: TaskQueries,
                    batch: dict[str, Any]) -> MLLMOutput:
        if task == "vqa":
            return self._vqa_forward(queries, batch)

        b = visual.tokens.shape[0]
        device = visual.tokens.device

        input_ids = self.prompt_ids.to(device).expand(b, -1)
        mm_types = self.mm_token_type_ids.to(device).expand(b, -1)
        attn1 = torch.ones_like(input_ids)
        # The prefill only depends on frozen weights (image + fixed prompt), never on the
        # trainable task tokens, so it needs no autograd graph -- only its KV cache.
        with torch.no_grad():
            prefill = self.qwen.model(
                input_ids=input_ids,
                pixel_values=self._cache["pixel_values"],
                image_grid_thw=self._cache["image_grid_thw"],
                mm_token_type_ids=mm_types,
                attention_mask=attn1,
                use_cache=True,
            )

        q = queries.tokens.to(prefill.last_hidden_state.dtype)
        # No attention_mask here: with an explicit mask, `compute_3d_position_ids` builds
        # position ids over its *whole* length (mismatching the new, cache-continuation-only
        # `inputs_embeds`). Omitting it takes the `arange(past_len, past_len + seq_len)` path,
        # which is the one that actually continues from the cached prefill positions.
        step = self.qwen.model(
            inputs_embeds=q,
            past_key_values=prefill.past_key_values,
            use_cache=False,
        )
        return MLLMOutput(task_hidden=TaskQueries(step.last_hidden_state, grid=queries.grid))

    def _vqa_forward(self, queries: TaskQueries, batch: dict[str, Any]) -> MLLMOutput:
        """[PERC] tokens (queries) as extra context after the image, then the question and
        answer as real text, teacher-forced. Batch size 1: variable-length text per example,
        no padding/masking logic yet (a training-loop simplification, not an architecture
        limit -- see scripts/train_vqa_c0_vs_c3.py).
        """
        b = queries.tokens.shape[0]
        if b != 1:
            raise NotImplementedError("vqa forward currently only supports batch_size=1")
        device = queries.tokens.device
        tok = self.tokenizer

        input_ids = self.image_prefix_ids.to(device)
        mm_types = self.image_mm_token_type_ids.to(device)
        with torch.no_grad():
            prefill = self.qwen.model(
                input_ids=input_ids,
                pixel_values=self._cache["pixel_values"],
                image_grid_thw=self._cache["image_grid_thw"],
                mm_token_type_ids=mm_types,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
            )

        q = queries.tokens.to(prefill.last_hidden_state.dtype)
        perc_step = self.qwen.model(inputs_embeds=q, past_key_values=prefill.past_key_values, use_cache=True)

        question, answer = batch["question"][0], batch["answer"][0]
        q_ids = tok(f"Question: {question}\nAnswer:", return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
        a_ids = tok(f" {answer}", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        eos = torch.tensor([[tok.eos_token_id]], device=device)
        text_ids = torch.cat([q_ids, a_ids, eos], dim=1)
        text_embeds = self.qwen.get_input_embeddings()(text_ids)
        text_step = self.qwen.model(inputs_embeds=text_embeds, past_key_values=perc_step.past_key_values,
                                    use_cache=False)
        logits = self.qwen.lm_head(text_step.last_hidden_state)

        n_q = q_ids.shape[1]
        targets = text_ids[:, n_q:]                    # answer tokens + eos
        pred_logits = logits[:, n_q - 1:-1, :].float()  # position i predicts token i+1
        loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]), targets.reshape(-1))
        return MLLMOutput(task_hidden=TaskQueries(perc_step.last_hidden_state, grid=queries.grid), lm_loss=loss)

    def decode(self, task: str, task_hidden: TaskQueries, visual: FeatureMap,
               batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if task == "seg":
            b = task_hidden.tokens.shape[0]
            h, w = task_hidden.grid
            logits = self.mask_head(task_hidden.tokens.float()).squeeze(-1).view(b, 1, h, w)
            return {"mask_logits": logits}
        if task == "vqa":
            return {}  # the loss is already in MLLMOutput.lm_loss (ConditionedMLLM copies it into preds)
        raise KeyError(task)

    def freeze_native(self) -> None:
        for p in self.qwen.parameters():
            p.requires_grad_(False)

    def task_parameters(self) -> dict[str, nn.Parameter]:
        params = {f"task_embed.{k}": v for k, v in self.task_embed.items()}
        params.update({f"mask_head.{k}": v for k, v in self.mask_head.named_parameters()})
        return params
