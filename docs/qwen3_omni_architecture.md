# Qwen3-Omni-30B-A3B-Thinking — architecture audit

Checkpoint `Qwen/Qwen3-Omni-30B-A3B-Thinking` (snapshot `2f443cfc`), official Hugging Face
implementation `transformers==5.17.0` (`transformers/models/qwen3_omni_moe/`). Everything below
was measured with `scripts/omni/audit_architecture.py` (full report: `results/omni/audit.json`)
on a real BUV frame (442×442), not carried over from Qwen3-VL. Tests: `tests/test_omni.py`
(CPU, official modules with random weights), `tests/test_omni_checkpoint.py` (real checkpoint, GPU).

## 1. Top level

| module | class | params |
|---|---|---|
| `thinker` | `Qwen3OmniMoeThinkerForConditionalGeneration` | 31.72 B |
| `thinker.audio_tower` | `Qwen3OmniMoeAudioEncoder` (32 layers, d=1280 → proj 2048) | 0.648 B |
| `thinker.visual` | `Qwen3OmniMoeVisionEncoder` | 0.539 B |
| `thinker.model` | `Qwen3OmniMoeThinkerTextModel` (48 MoE decoder layers) | 30.22 B |
| `thinker.lm_head` | `Linear(2048 → 152064)` | 0.311 B |

The Thinking checkpoint has `enable_audio_output=false`: no Talker / Code2Wav weights. With
`return_audio=False` the official `Qwen3OmniMoeForConditionalGeneration.generate` only calls
`self.thinker.generate(max_new_tokens=thinker_max_new_tokens, eos_token_id=151645, ...)`, so the
project calls `thinker.generate` directly with those arguments (identical code path).

## 2. Vision encoder `thinker.visual` (the steering site)

Config: depth 27, hidden 1152, MLP 4304 (`gelu_pytorch_tanh`), 16 heads, patch 16, temporal patch 2,
spatial merge 2, learned abs. position table 2304 = 48×48 (bilinearly resampled to each grid),
2D axial RoPE, `out_hidden_size` 2048 (= LLM width), DeepStack indexes `[8, 16, 24]`.

| path | op | output shape (BUV 442² → 448², grid t,h,w = 1,28,28) |
|---|---|---|
| processor | `Qwen2VLImageProcessor` smart-resize to multiple of 32, norm mean=std=0.5 → [-1,1], patchify | `pixel_values [784, 1536]` (= 3·2·16·16), `image_grid_thw [[1,28,28]]` |
| `visual.patch_embed.proj` | `Conv3d(3, 1152, k=(2,16,16), s=(2,16,16))` | `[784, 1152]` |
| `visual.pos_embed` | `Embedding(2304, 1152)`, bilinear interpolation weights | added: `[784, 1152]` |
| `visual.rotary_pos_emb` | axial 2D RoPE (h, w), no params | cos/sin per token |
| `visual.blocks[i]`, i=0..26 | pre-LN ViT block: `norm1 → attn(qkv, proj) → +`, `norm2 → mlp(fc1, fc2) → +` | `[784, 1152]` each |
| `visual.merger_list[k]`, k=0,1,2 | DeepStack merger on the output of block 8 / 16 / 24: view 4 tokens → 4608, `LayerNorm(4608)` → `Linear 4608→4608 → GELU → Linear 4608→2048` | `[196, 2048]` each |
| `visual.merger` | final merger on the output of block 26: `LayerNorm(1152)` per token, view 4 → 4608, same MLP | `[196, 2048]` = LLM image tokens |

Output (`BaseModelOutputWithDeepstackFeatures`): `last_hidden_state [784,1152]` (block 26 output),
`pooler_output [196,2048]` (image tokens), `deepstack_features` 3 × `[196,2048]`.

**Packed sequence.** All images/videos of a batch are concatenated along dim 0; attention is
block-diagonal per item via `cu_seqlens` (from `grid_thw`). There is no batch dimension inside
the ViT; steering therefore splits the packed tokens per item with `grid_thw`
(`biomllm.omni.layout.tokens_per_item`).

**Token order (verified).** `patchify` permutes `(t, h/2, 2, w/2, 2)` → `(t, h/2, w/2, 2, 2)`:
block tokens are *merge-window-major* (each consecutive run of 4 tokens is one 2×2 window, windows in
raster order). Un-patchifying the processor output with `layout.pixel_patches_to_image` recovers
the resized image (MAE 1.0e-5 vs a bicubic resize of the input; exact in the unit test). Merged
tokens (`pooler_output`, `deepstack_features`) are plain raster order over `(t, h/2, w/2)`.
Conversions: `layout.block_tokens_to_grid` → `[t, 1152, h, w]` (28×28 here),
`layout.merged_tokens_to_grid` → `[t, 2048, h/2, w/2]` (14×14 here). The temporal patch duplicates a
still image (both frames identical), so still images and cine clips share the same path.

**Activation scale across depth** (RMS of block outputs on the BUV frame): 0.34 after patch_embed,
0.25–0.3 in blocks 1–8, 0.6–0.9 in blocks 9–16, 0.8–2.1 in blocks 17–24, 1.54 in block 25 and
**23.4 in block 26** (the final merger normalises it with its per-token LayerNorm). The steering
cross-attention therefore layer-normalises its queries (`query_norm=True`) and uses a per-layer gate.

## 3. From visual tokens to the Thinker

`Qwen3OmniMoeThinkerForConditionalGeneration.forward`:

1. `inputs_embeds = model.embed_tokens(input_ids)`.
2. `get_image_features(pixel_values, image_grid_thw)` → `visual(...)`; `pooler_output` is split per image.
3. `image_mask` = positions where `input_ids == image_token_id (151655)`; the chat template
   expands `<|image_pad|>` to `h·w/4` tokens (196 here) between `<|vision_start|>` and
   `<|vision_end|>`; `inputs_embeds.masked_scatter(image_mask, image_embeds)`.
4. DeepStack: `deepstack_features[k]` is **added** to the hidden states at the image positions
   after decoder layer k (k = 0, 1, 2) — `ThinkerTextModel._deepstack_process`.
5. Position ids: `get_rope_index` → `[3, B, S]` (t, h, w) M-RoPE; `mrope_section [24, 20, 20]`,
   interleaved. For the BUV prompt, image tokens occupy positions t=4, h∈[4,17], w∈[4,17] (the 4
   preceding tokens are `<|im_start|>user\n<|vision_start|>`), `rope_deltas = -182`.
6. Audio (`audio_token_id 151675`) and video (`151656`) go through the same `masked_scatter`
   mechanism; video tokens also carry DeepStack features (image and video masks are merged).

Chat template for grounding (official cookbook, no system prompt):
```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Locate the object: breast lesion.<|im_end|>
<|im_start|>assistant
```
The Thinking model then emits `<think> … </think>` followed by the answer.

## 4. Text decoder `thinker.model`

48 × `Qwen3OmniMoeThinkerTextDecoderLayer`: hidden 2048, GQA 32 q-heads / 4 kv-heads, head_dim 128,
q/k RMSNorm, RoPE θ=1e6; MoE with 128 experts, top-8, expert FFN 768, no shared expert, router
`mlp.gate.weight [128, 2048]`. **Experts are stored fused**: `mlp.experts.gate_up_proj
[128, 1536, 2048]`, `mlp.experts.down_proj [128, 2048, 768]` (the checkpoint stores them per
expert; transformers fuses them on load). Vocabulary 152064, untied `lm_head`.

Text-token representations: `thinker.model` hidden states (per layer via `output_hidden_states`),
width 2048 — the source of the Omni-native ablation of `ClinicalTextEncoder(source="omni")`.

## 5. Precision (deviation from BF16, documented)

BF16 weights are 63.4 GB; the project may use one 40 GB A100 (GPU 1). The HF bitsandbytes
integration only replaces `nn.Linear`, so it leaves the fused experts (29.9 B params) in BF16.
`biomllm.omni.loading` therefore quantizes to NF4 every expert matrix separately (`gate_up`,
`down`; blocksize 64, fp32 absmax, no double quantization) plus the attention projections
(`bnb.nn.Linear4bit`). It keeps **BF16** for the entire vision encoder, the audio tower, the embeddings,
`lm_head`, the norms and the routers (1.82 B params). The perception pathway where steering acts is
therefore bit-identical to the official weights; only the language decoder is approximated.

Expert execution (`NF4Experts`, implementation tag `nf4-per-expert-fp32absmax-fused-decode-v2`,
stored in every `run_info.json`):

* prefill / training: the official loop over the hit experts, with `bnb.matmul_4bit`
  (differentiable w.r.t. hidden states; weights are re-dequantized in backward, never cached);
* decode (≤ 4 rows): the 8 routed experts' NF4 blocks and scales are gathered with one index,
  dequantized in one call and applied with two batched matmuls, with no host syncs.
  Per MoE layer and token this takes 0.49 ms, vs 2.62 ms for a per-expert loop and 11.5 ms for the
  official BF16 loop (A100, measured). The fused and loop paths agree to bf16 rounding (0.8% rel.).

Validation against the BF16 reference (`precision="bf16"`, accelerate CPU offload, GPU budget
30 GiB) on a BUV subset uses `scripts/omni/validate_precision.py`; results are in §7. The first
implementation (per-expert loop at decode, double-quantized scales) was used only for the initial
official-example reproduction (`results/omni/*_v1impl`). Every BUV result uses v2.

FlashAttention-2 is not installed in the environment; all runs use `sdpa` (numerically
equivalent attention, less memory-efficient for long thinking traces).

## 6. Where the project hooks in

| component | attach point | file |
|---|---|---|
| feature taps (mask decoder input) | forward hooks on `visual.blocks[l]`, `visual.merger_list[k]`, `visual.merger`; `grid_thw` from a pre-hook on `visual` | `omni/features.py` |
| clinical steering | forward hooks on `visual.blocks[l]` returning `V + tanh(α_l)·CA(V, T)` per item | `omni/steering.py` |
| clinical text | frozen RoBERTa-large (primary) or Thinker hidden states (ablation) → L2 norm → MLP | `omni/text_encoder.py` |
| grounding | official prompt / JSON `bbox_2d` 0–1000 convention, `</think>` split | `omni/grounding.py` |

## 7. Measurements

(filled from `results/omni/`)
