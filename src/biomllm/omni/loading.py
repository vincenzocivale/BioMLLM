"""Load the Qwen3-Omni Thinker in the precisions used by the project.

`bf16`     reference: official BF16 weights, layers that do not fit the GPU offloaded to CPU
           by accelerate (exact, slow). Used to validate the reduced-precision configuration.
`nf4`      production on one 40 GB A100: the text decoder (attention projections + all MoE
           experts, ~30B params) is NF4-quantized; everything else stays BF16, in particular the
           whole vision encoder `thinker.visual`, where the steering modules are inserted, so the
           perception pathway under study is bit-identical to the official weights.

Why a custom quantizer: in transformers 5.x the MoE experts are fused 3D `nn.Parameter`s
(`Qwen3OmniMoeThinkerTextExperts.gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`), which the HF
bitsandbytes integration (it only swaps `nn.Linear`) leaves in BF16 (~58 GB). `NF4Experts`
quantizes every expert matrix separately and runs `bnb.matmul_4bit` only on the experts the
router selected, so decoding touches 8/128 experts per layer as in the official forward.
Implementation tag recorded in every run_info.json: `NF4_IMPL`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
NF4_IMPL = "nf4-per-expert-fp32absmax-fused-decode-v2"
log = logging.getLogger(__name__)


class NF4Experts(nn.Module):
    """Drop-in replacement for `Qwen3OmniMoeThinkerTextExperts` with per-expert NF4 weights.

    Every expert matrix is quantized on its own (blocksize 64, fp32 absmax: no double
    quantization, so the scales of any subset of experts can be gathered by indexing).

    * decode steps (<= `decode_rows` tokens): the 8 routed experts' NF4 blocks are gathered with
      one index_select, dequantized in one call and applied with two batched matmuls -- no host
      synchronisation and ~10 kernels per layer instead of a Python loop over experts.
    * prefill / training: the official loop over hit experts with `bnb.matmul_4bit`
      (differentiable w.r.t. the hidden states; weights are re-dequantized in backward, never stored).
    """

    #: inputs with at most this many rows (decode steps) take the fused gather path
    decode_rows = 4

    def __init__(self, experts: nn.Module, device: torch.device, blocksize: int = 64) -> None:
        super().__init__()
        import bitsandbytes.functional as bnbF

        self.num_experts = experts.num_experts
        self.act_fn = experts.act_fn
        self.blocksize = blocksize
        self.shapes: dict[str, torch.Size] = {}
        self.states: dict[str, list] = {}
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(experts, name)
            packed, absmax = [], []
            for e in range(self.num_experts):
                q, st = bnbF.quantize_4bit(w[e].to(device, torch.bfloat16).contiguous(), blocksize=blocksize,
                                           quant_type="nf4", compress_statistics=False)
                packed.append(q)
                absmax.append(st.absmax)
            self.shapes[name] = w.shape[1:]
            self.register_buffer(f"{name}_q", torch.stack(packed), persistent=False)       # [E, R*C/2, 1]
            self.register_buffer(f"{name}_absmax", torch.stack(absmax), persistent=False)  # [E, R*C/bs]
            self.register_buffer("code", st.code, persistent=False)
            am = getattr(self, f"{name}_absmax")
            self.states[name] = [self._state(am[e], self.shapes[name]) for e in range(self.num_experts)]

    def _state(self, absmax: torch.Tensor, shape) -> "object":
        import bitsandbytes.functional as bnbF

        return bnbF.QuantState(absmax=absmax, shape=torch.Size(shape), code=self.code, blocksize=self.blocksize,
                               quant_type="nf4", dtype=torch.bfloat16)

    def _gather(self, name: str, sel: torch.Tensor) -> torch.Tensor:
        """Dequantized weights of the experts `sel` [k] -> [k, R, C] (BF16)."""
        import bitsandbytes.functional as bnbF

        q = getattr(self, f"{name}_q")[sel].reshape(-1, 1)
        am = getattr(self, f"{name}_absmax")[sel].reshape(-1)
        return bnbF.dequantize_4bit(q, self._state(am, (sel.numel(), *self.shapes[name])))

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        import bitsandbytes as bnb

        if hidden_states.shape[0] <= self.decode_rows:
            out = []
            for t in range(hidden_states.shape[0]):
                sel = top_k_index[t]
                x = hidden_states[t].expand(sel.numel(), -1)[:, :, None]           # [k, H, 1]
                gate, up = torch.bmm(self._gather("gate_up_proj", sel), x)[..., 0].chunk(2, dim=-1)
                h = (self.act_fn(gate) * up)[:, :, None]                               # [k, I, 1]
                y = torch.bmm(self._gather("down_proj", sel), h)[..., 0]               # [k, H]
                out.append((y * top_k_weights[t, :, None]).sum(0))
            return torch.stack(out).to(hidden_states.dtype)

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in expert_hit[:, 0].tolist():
            top_k_pos, token_idx = torch.where(expert_mask[e])
            x = hidden_states[token_idx]
            gate, up = bnb.matmul_4bit(x, self.gate_up_proj_q[e].t(),
                                       quant_state=self.states["gate_up_proj"][e]).chunk(2, dim=-1)
            h = bnb.matmul_4bit(self.act_fn(gate) * up, self.down_proj_q[e].t(),
                                quant_state=self.states["down_proj"][e])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final


def _linear4bit(lin: nn.Linear, device: torch.device) -> nn.Module:
    import bitsandbytes as bnb

    q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                          compute_dtype=torch.bfloat16, quant_type="nf4", compress_statistics=True)
    q.weight = bnb.nn.Params4bit(lin.weight.data.to(torch.bfloat16).contiguous(), requires_grad=False,
                                 quant_type="nf4", compress_statistics=True)
    if lin.bias is not None:
        q.bias = nn.Parameter(lin.bias.data.to(torch.bfloat16), requires_grad=False)
    return q.to(device)


def quantize_text_decoder_nf4(thinker: nn.Module, device: torch.device) -> dict:
    """In place: move the thinker to `device`, NF4-quantizing the decoder layers one at a time
    (peak GPU memory = quantized model + one BF16 layer)."""
    stats = {"nf4_params": 0, "bf16_params": 0}
    for layer in thinker.model.layers:
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, name)
            stats["nf4_params"] += lin.weight.numel()
            setattr(attn, name, _linear4bit(lin, device))
        experts = layer.mlp.experts
        stats["nf4_params"] += experts.gate_up_proj.numel() + experts.down_proj.numel()
        layer.mlp.experts = NF4Experts(experts, device)
        del experts
        layer.to(device)  # norms, router: BF16
    for mod in (thinker.visual, thinker.audio_tower, thinker.model.embed_tokens, thinker.model.norm,
                thinker.model.rotary_emb, thinker.lm_head):
        mod.to(device)
    for n, p in thinker.named_parameters():
        if p.dtype == torch.bfloat16:
            stats["bf16_params"] += p.numel()
    return stats


def load_thinker(precision: str = "nf4", model_id: str = MODEL_ID, device: str = "cuda:0",
                 gpu_budget_gib: int = 30, attn_implementation: str = "sdpa"):
    """Return (thinker, processor, info). `thinker` is `Qwen3OmniMoeThinkerForConditionalGeneration`
    with the official weights; the Thinking checkpoint has no Talker, and without audio output the
    official `Qwen3OmniMoeForConditionalGeneration.generate` reduces to `thinker.generate`."""
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16, attn_implementation=attn_implementation)
    info = {"model_id": model_id, "precision": precision, "attn_implementation": attn_implementation}
    if precision == "bf16":
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, device_map="auto",
            max_memory={0: f"{gpu_budget_gib}GiB", "cpu": "400GiB"}, **kw)
        info["device_map"] = {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    elif precision == "nf4":
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, device_map="cpu", **kw)
        info.update(quantize_text_decoder_nf4(model.thinker, torch.device(device)))
        info["nf4_impl"] = NF4_IMPL
    else:
        raise ValueError(f"unknown precision {precision!r}")
    thinker = model.thinker.eval()
    thinker.requires_grad_(False)
    if torch.cuda.is_available():
        info["gpu_mem_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    return thinker, processor, info


def load_visual(model_id: str = MODEL_ID, device: str = "cuda", attn_implementation: str = "sdpa"):
    """Only `thinker.visual` with the official BF16 weights (0.54 B params, ~1 GB): enough for the
    mask decoder and for steering the perception pathway without the 30 B language model."""
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    vc = AutoConfig.from_pretrained(model_id).thinker_config.vision_config
    vc._attn_implementation = attn_implementation
    root = Path(snapshot_download(model_id, allow_patterns=["*.json"]))
    wmap = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    prefix, state = "thinker.visual.", {}
    for shard in sorted({f for k, f in wmap.items() if k.startswith(prefix)}):
        with safe_open(str(Path(snapshot_download(model_id, allow_patterns=[shard])) / shard), "pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = f.get_tensor(k)
    visual = Qwen3OmniMoeVisionEncoder._from_config(vc, dtype=torch.bfloat16)
    visual.load_state_dict(state, strict=True)
    return visual.to(device).eval().requires_grad_(False)
