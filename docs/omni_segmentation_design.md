# Masks from Qwen3-Omni: literature audit and chosen design

Goal: obtain a mask while Qwen3-Omni remains the source of semantic and spatial reasoning, so that
segmentation quality measures what the omni-modal model perceives. The main risk is a confound: a
strong second visual backbone doing the segmentation.

## Strategies in the literature

| family | examples | where the mask comes from | confound for our question |
|---|---|---|---|
| [SEG] token → SAM | LISA (Lai et al., CVPR 2024), GLaMM (Rasheed et al., CVPR 2024) | SAM image encoder + SAM mask decoder, prompted by an LLM token embedding | **high**: SAM's own ViT-H encodes the pixels; the MLLM only supplies a prompt |
| MLLM → prompts → SAM | SAM4MLLM (Chen et al., ECCV 2024), Seg-Zero-style RL + SAM2 | the MLLM emits box/points; SAM segments | **high**: same as above; tests the MLLM as a prompter only |
| separate segmentation backbone + query decoder | PSALM (Zhang et al., ECCV 2024), OMG-LLaVA (Zhang et al., NeurIPS 2024) | Mask2Former-style decoder on a dedicated (Swin / ConvNeXt-CLIP) encoder | medium-high: the dense encoder is not the MLLM's |
| light decoder on the MLLM's own vision features | PixelLM (Ren et al., CVPR 2024) | small pixel decoder over the MLLM's CLIP-ViT features, driven by codebook tokens | **low**: mask decoded from the representation the MLLM uses |
| mask as text | Text4Seg (Lan et al., ICLR 2025) | per-patch semantic descriptors emitted as text (optionally refined by SAM) | low without SAM refinement, but patch-level resolution and very long outputs |
| per-patch task tokens through the LLM | STAMP-style (previous Qwen3-VL setup in this repo) | one [MASK] token per patch, read by a linear head | low, but costs h·w extra LLM tokens (784 at 448²) through the 30B MoE |
| linear probe on patch features | SteerViT evaluation (Ruthardt et al., ECCV 2026), DINOv2 probing | per-patch linear classifier | lowest, and the weakest readout |

## Choice

**Box-prompted lightweight decoder on frozen Qwen3-Omni vision maps** (`omni/mask_decoder.py`).
It is in the PixelLM family, restricted further:

* The inputs are feature maps of the Omni vision encoder: patch-grid outputs of `visual.blocks[8,16,26]`
  (1152-d, stride 16) and the merged LLM image tokens (2048-d, stride 32). All are read through hooks
  on the official forward (`VisualTaps`, checked against the tensors the Thinker receives).
* There is **no raw-pixel path by default**. `pixel_skip=True` exists only as a flagged ablation,
  because a pixel convolution adds an edge detector outside the foundation model.
* 1.15 M trainable parameters (Thinker: 31.7 B). It uses convolution + two ×2 transposed-conv upsamplings, and
  outputs logits at stride 4.
* The box prompt is a rasterised box channel plus a Fourier/FiLM embedding of the box.
* Conditions:
  * **oracle**: GT box + frozen Omni features → mask (representation quality).
  * **end-to-end**: Omni-predicted box (parsed grounding output) + frozen Omni features → mask.
  * **lower bound**: the same decoder on a full-image box (no localisation).
* Steering (MODE 2) changes the features the decoder reads. It is the same decoder, retrained per
  condition with an identical budget, so differences are attributable to the representation.

Known limitation: the patch stride (16 px at the default ~448² resize) bounds boundary accuracy.
Qwen3-Omni supports dynamic resolution, so a resolution ablation (for example 896², a 56×56 grid) is part of
the plan. It is a property of the backbone's input and not a new visual module.

## Data requirement

BUV (the detection benchmark) has **boxes only**, so segmentation needs an ultrasound set with masks
(for example BUSI, breast, 780 images with masks). It is not on disk yet (see open questions in the
run log).
