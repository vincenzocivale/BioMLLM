"""Builders for the external encoders used as F^S.

Heavy dependencies (transformers, timm, open_clip) are imported lazily so the core package
and the tests run without them. Preprocessing statistics are read from each model's own
processor / data config rather than hard-coded.

Checkpoint ids used in the paper are set in configs/expert/*.yaml.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from biomllm.models.experts.base import FrozenExpert
from biomllm.models.types import FeatureMap

EXPERTS: dict[str, Callable[..., FrozenExpert]] = {}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def register(kind: str):
    def deco(fn):
        EXPERTS[kind] = fn
        return fn
    return deco


def build_expert(kind: str, **kwargs) -> FrozenExpert:
    if kind not in EXPERTS:
        raise KeyError(f"unknown expert kind '{kind}', available: {sorted(EXPERTS)}")
    return EXPERTS[kind](**kwargs).freeze()


def _drop_prefix(tokens: torch.Tensor, num_prefix: int, grid: tuple[int, int]) -> FeatureMap:
    return FeatureMap(tokens[:, num_prefix:], grid)


# --------------------------------------------------------------------------- toy (tests)

class ToyExpert(FrozenExpert):
    """Random conv patchifier. Used in tests and in the debug experiment."""

    def __init__(self, dim: int = 32, image_size: int = 64, patch_size: int = 8, seed: int = 0):
        super().__init__(dim, image_size, IMAGENET_MEAN, IMAGENET_STD, name="toy")
        g = torch.Generator().manual_seed(seed)
        self.patch = nn.Conv2d(3, dim, patch_size, patch_size)
        with torch.no_grad():
            self.patch.weight.copy_(torch.randn(self.patch.weight.shape, generator=g) * 0.1)
            self.patch.bias.zero_()

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        return FeatureMap.from_image(self.patch(pixel_values))


register("toy")(ToyExpert)


# --------------------------------------------------------------- HF ViTs (DINOv2, RAD-DINO)

class HFViTExpert(FrozenExpert):
    """transformers ViT-style backbones returning [CLS, (registers), patches]:
    DINOv2 (generalist), RAD-DINO (chest X-ray specialist)."""

    def __init__(self, model_id: str, pretrained: bool = True, layer: int = -1,
                 image_size: int | None = None, name: str = ""):
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(model_id)
        size = image_size or _processor_size(proc)
        super().__init__(0, size, tuple(proc.image_mean), tuple(proc.image_std),
                         name=name or model_id)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_id)
        else:
            # "random expert" control: same architecture, no pretraining.
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_id))
        cfg = self.model.config
        self.dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.num_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self.layer = layer

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        out = self.model(pixel_values=pixel_values, output_hidden_states=self.layer != -1)
        tokens = out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]
        h, w = (s // self.patch_size for s in pixel_values.shape[-2:])
        return _drop_prefix(tokens, self.num_prefix, (h, w))


register("hf_vit")(HFViTExpert)


class SiglipExpert(FrozenExpert):
    """SigLIP vision tower (generalist VL encoder; no CLS token)."""

    def __init__(self, model_id: str, name: str = ""):
        from transformers import AutoImageProcessor, SiglipVisionModel

        proc = AutoImageProcessor.from_pretrained(model_id)
        super().__init__(0, _processor_size(proc), tuple(proc.image_mean),
                         tuple(proc.image_std), name=name or model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id)
        self.dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        tokens = self.model(pixel_values=pixel_values).last_hidden_state
        h, w = (s // self.patch_size for s in pixel_values.shape[-2:])
        return FeatureMap(tokens, (h, w))


register("siglip")(SiglipExpert)


# ------------------------------------------------------------------- timm (UNI, generic)

class TimmExpert(FrozenExpert):
    """Any timm ViT, e.g. UNI (hf-hub:MahmoodLab/uni, gated, pathology)."""

    def __init__(self, model_id: str, pretrained: bool = True, model_kwargs: dict | None = None,
                 name: str = ""):
        import timm

        model = timm.create_model(model_id, pretrained=pretrained, num_classes=0,
                                  **(model_kwargs or {}))
        data_cfg = timm.data.resolve_data_config({}, model=model)
        super().__init__(model.num_features, tuple(data_cfg["input_size"][1:]),
                         tuple(data_cfg["mean"]), tuple(data_cfg["std"]), name=name or model_id)
        self.model = model

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        tokens = self.model.forward_features(pixel_values)
        grid = self.model.patch_embed.dynamic_feat_size(pixel_values.shape[-2:]) \
            if getattr(self.model.patch_embed, "dynamic_img_size", False) \
            else self.model.patch_embed.grid_size
        return _drop_prefix(tokens, self.model.num_prefix_tokens, tuple(grid))


register("timm")(TimmExpert)


# ----------------------------------------------------------------- open_clip (BiomedCLIP)

class OpenClipExpert(FrozenExpert):
    """Patch tokens of an open_clip image tower with a timm trunk, e.g. BiomedCLIP
    (hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)."""

    def __init__(self, model_id: str, name: str = ""):
        import open_clip

        model, _ = open_clip.create_model_from_pretrained(model_id)
        visual = model.visual
        cfg = getattr(visual, "preprocess_cfg", {}) or {}
        trunk = visual.trunk
        super().__init__(trunk.num_features, tuple(trunk.patch_embed.img_size),
                         tuple(cfg.get("mean", CLIP_MEAN)), tuple(cfg.get("std", CLIP_STD)),
                         name=name or model_id)
        self.trunk = trunk

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        tokens = self.trunk.forward_features(pixel_values)
        return _drop_prefix(tokens, self.trunk.num_prefix_tokens,
                            tuple(self.trunk.patch_embed.grid_size))


register("open_clip")(OpenClipExpert)


# ------------------------------------------------------------------- SAM-family (MedSAM)

class SamEncoderExpert(FrozenExpert):
    """Image encoder of a transformers SamModel, e.g. MedSAM.

    Caution: if the MLLM's mask decoder is SAM-derived, gains from this expert may reflect
    encoder/decoder affinity rather than specialist knowledge (see README, confounds).
    Also check that the checkpoint's preprocessing matches the one it was trained with.
    """

    def __init__(self, model_id: str, name: str = ""):
        from transformers import SamModel, SamProcessor

        proc = SamProcessor.from_pretrained(model_id).image_processor
        size = proc.size.get("longest_edge", 1024)
        super().__init__(0, size, tuple(proc.image_mean), tuple(proc.image_std),
                         name=name or model_id)
        self.model = SamModel.from_pretrained(model_id)
        self.dim = self.model.config.vision_config.output_channels

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        return FeatureMap.from_image(self.model.get_image_embeddings(pixel_values))


register("sam_encoder")(SamEncoderExpert)


class Sam2EncoderExpert(FrozenExpert):
    """Image encoder of a transformers Sam2Model, e.g. facebook/sam2.1-hiera-base-plus: the
    generalist (natural-image) counterpart of SonoBase. Its last level has the same layout as
    SonoBase's `vision_features` (stride 16, 256 channels: 64 x 64 at 1024 px), so C2 vs C3
    isolates the ultrasound pretraining rather than the feature format."""

    def __init__(self, model_id: str, out_grid: int | None = None, name: str = ""):
        from transformers import Sam2Model, Sam2Processor

        proc = Sam2Processor.from_pretrained(model_id).image_processor
        super().__init__(0, (proc.size.height, proc.size.width), tuple(proc.image_mean),
                         tuple(proc.image_std), name=name or model_id)
        model = Sam2Model.from_pretrained(model_id)
        # Only the image encoder (Hiera + FPN neck); prompt encoder / mask decoder are dropped.
        self.vision_encoder = model.vision_encoder
        self.dim = model.config.vision_config.fpn_hidden_size
        self.out_grid = out_grid

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        import torch.nn.functional as F

        # Raw last FPN level, like SonoBase's `vision_features` (Sam2Model.get_image_embeddings
        # would also add the image-mode `no_memory_embedding`, which SonoBase's output lacks).
        x = self.vision_encoder(pixel_values).fpn_hidden_states[-1]
        if self.out_grid is not None and x.shape[-1] != self.out_grid:
            x = F.adaptive_avg_pool2d(x, self.out_grid)
        return FeatureMap.from_image(x)


register("sam2_encoder")(Sam2EncoderExpert)


# ------------------------------------------------------- Qwen3-VL native encoder (F^MLLM)

class QwenVLVisionExpert(FrozenExpert):
    """The MLLM's own vision encoder (Qwen3-VL), used for the "self" condition in the probes.

    Only the `model.visual.*` weights are loaded. Patchification goes through the official
    image processor, so token order matches what the LLM receives.
        layer="merged": tokens after the 2x2 merger, i.e. exactly what the LLM sees
                        (out_hidden_size, grid = image_size / 32).
        layer="last":   last ViT block before the merger (hidden_size, grid = image_size / 16).
    """

    def __init__(self, model_id: str, image_size: int = 512, layer: str = "merged",
                 dtype: str = "bfloat16", name: str = ""):
        import json

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        proc = AutoImageProcessor.from_pretrained(model_id)
        super().__init__(0, image_size, tuple(proc.image_mean), tuple(proc.image_std),
                         name=name or model_id)
        cfg = AutoConfig.from_pretrained(model_id).vision_config
        self.patch_size, self.merge = cfg.patch_size, cfg.spatial_merge_size
        if image_size % (self.patch_size * self.merge):
            raise ValueError(f"image_size must be a multiple of {self.patch_size * self.merge}")
        if layer not in ("merged", "last"):
            raise ValueError("layer must be 'merged' or 'last'")
        self.layer = layer
        self.dim = cfg.out_hidden_size if layer == "merged" else cfg.hidden_size
        self.processor = proc
        self.visual = Qwen3VLVisionModel(cfg)

        index = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))
        prefix = "model.visual."
        state = {}
        for shard in sorted({f for k, f in index["weight_map"].items() if k.startswith(prefix)}):
            with safe_open(hf_hub_download(model_id, shard), framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        state[k[len(prefix):]] = f.get_tensor(k)
        missing, unexpected = self.visual.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.visual.to(getattr(torch, dtype))

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # resizing only; normalisation and patchification happen in the official processor
        import torch.nn.functional as F

        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear",
                                   align_corners=False, antialias=True)
        return images

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        imgs = [im.permute(1, 2, 0).float().cpu().numpy() for im in pixel_values]
        h, w = self.image_size
        batch = self.processor(images=imgs, do_rescale=False, do_resize=False, return_tensors="pt")
        grid_thw = batch["image_grid_thw"].to(pixel_values.device)
        out = self.visual(batch["pixel_values"].to(pixel_values.device, self.visual.dtype),
                          grid_thw=grid_thw, return_dict=True)
        b = pixel_values.shape[0]
        gh, gw = h // self.patch_size, w // self.patch_size
        if self.layer == "merged":
            tokens = out.pooler_output.view(b, (gh // self.merge) * (gw // self.merge), -1)
            return FeatureMap(tokens, (gh // self.merge, gw // self.merge))
        # pre-merger tokens come in 2x2 merge blocks: (h/2, w/2, 2, 2) -> row-major (h, w)
        m = self.merge
        x = out.last_hidden_state.view(b, gh // m, gw // m, m, m, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, -1)
        return FeatureMap(x, (gh, gw))


register("qwen_vl_vision")(QwenVLVisionExpert)


# ------------------------------------------------------------ SonoBase (ultrasound, SAM2-based)

class SonoBaseExpert(FrozenExpert):
    """Image encoder of SonoBase (arXiv:2609.19230): SAM2 with a Hiera-B (256 px) + ConvNeXt-S
    (512 px) + ConvNeXt-T (1024 px) pyramid trunk and SAM2's FPN neck, pretrained on the 53
    public ultrasound datasets of SonoCorpus (BUV is not among them). Weights: CC BY-NC 4.0.

    The encoder is built from the checkpoint's own resolved config (`config.yaml` in the HF
    repo) with the modelling code of the SonoBase release (`repo_src`, its `src/` folder,
    imported as `nemo_cv`). Feature levels (1024 px input):
        level=-1  `vision_features`, stride 16: 64 x 64 x 256 (what SAM2's decoder reads)
        level=0/1 FPN levels at stride 4 / 8
    `out_grid` average-pools the map (e.g. 32 -> 1024 tokens) to keep cross-attention cheap.
    Caution: SAM2-derived, see the note on SamEncoderExpert about decoder affinity.
    """

    def __init__(self, repo_src: str, model_id: str = "AlfredQin/sonobase",
                 filename: str = "sonobase_hiera_b_conv_s_conv_t.pt", pretrained: bool = True,
                 level: int = -1, out_grid: int | None = None, dtype: str = "bfloat16",
                 name: str = ""):
        import sys

        import hydra
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(hf_hub_download(model_id, "config.yaml"))
        size = int(cfg.model.image_encoder.trunk.img_size2)
        super().__init__(int(cfg.model.image_encoder.neck.d_model), size, IMAGENET_MEAN,
                         IMAGENET_STD, name=name or model_id)
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        enc_cfg = cfg.model.image_encoder
        # The trunk would otherwise load SAM2 / DINOv3 init weights that the checkpoint
        # overwrites anyway (and whose paths only exist on the authors' cluster).
        enc_cfg.trunk.ckpt_path0 = None
        enc_cfg.trunk.branch1.pretrained = False
        enc_cfg.trunk.branch2.pretrained = False
        self.encoder = hydra.utils.instantiate(enc_cfg, _recursive_=True, _convert_="all")
        if pretrained:
            state = torch.load(hf_hub_download(model_id, filename), map_location="cpu",
                               weights_only=True)["model"]
            prefix = "image_encoder."
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"SonoBase weights mismatch: missing={missing[:5]} "
                                   f"unexpected={unexpected[:5]}")
        self.level = level
        self.out_grid = out_grid
        self.autocast_dtype = getattr(torch, dtype)

    def extract(self, pixel_values: torch.Tensor) -> FeatureMap:
        import torch.nn.functional as F

        with torch.autocast(pixel_values.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype != torch.float32):
            out = self.encoder(pixel_values)
        x = out["vision_features"] if self.level == -1 else out["backbone_fpn"][self.level]
        x = x.float()
        if self.out_grid is not None and x.shape[-1] != self.out_grid:
            x = F.adaptive_avg_pool2d(x, self.out_grid)
        return FeatureMap.from_image(x)


register("sonobase")(SonoBaseExpert)


def _processor_size(proc) -> tuple[int, int]:
    size = getattr(proc, "crop_size", None) or proc.size
    if "height" in size:
        return size["height"], size["width"]
    s = size.get("shortest_edge", 224)
    return s, s
