# Qwen3-Omni experiments: matrix, controls, decisions

Backbone: `Qwen/Qwen3-Omni-30B-A3B-Thinking`, one checkpoint for every condition, all Omni weights
frozen. Hardware: GPU 1 only (A100 40 GB). The text decoder runs in NF4 and the vision encoder in BF16
(see `qwen3_omni_architecture.md` §5). Queue: `scripts/omni/run_queue.sh`, then (via
`scripts/omni/handoff.sh`) `scripts/omni/run_queue2.sh`. Logs are in `results/omni/logs/`.

Scope of the claims: these runs exercise only **image + text → text/boxes**. They say nothing about
the audio or video capabilities of the omni-modal model.

## Detection matrix (BUV val, 37 videos × 3 evenly spaced frames = 111 frames)

| cell | prompt to the LLM | visual steering | trained params | run dir (`results/omni/`) |
|---|---|---|---|---|
| MODE 0 vanilla | official `Locate the object: breast lesion.` | none | 0 | `buv_vanilla_{forced,thinking,nothink}` |
| MODE 1 guideline prompt | BI-RADS guideline + official prompt | none | 0 | `buv_guideline_{forced,thinking,nothink}` |
| MODE 2 steering | official | guideline → RoBERTa → CA in 13 vision blocks | 71.6 M (13 × 5.3 M CA + 2.5 M text MLP) | `buv_steer_guideline_*` |
| MODE 1+2 | guideline + official | guideline | 71.6 M | `buv_steer_prompt_guideline_forced` |
| control: capacity | official | **neutral** text of matched length | 71.6 M | `buv_steer_neutral_forced` |
| control: text dependence | official | steering trained on the guideline, evaluated with a **wrong-organ** (thyroid TI-RADS) text, and with the neutral text | 0 extra | `buv_steer_guideline_{wrongtext,neutraltext}_forced` |

How to read it (falsification):

* A steering gain is attributed to **clinical knowledge** only if MODE 2 > capacity control. Both
  have identical architecture, data, steps and seed; only the conditioning text differs.
* A gain reflects **use of the text** only if MODE 2 with the wrong text < MODE 2 with the right text.
  If they are equal, the modules learned a text-independent domain shift, i.e. a small adapter.
* Frames of one cine loop are correlated, so CIs use a cluster bootstrap over videos
  (`summarize_grounding.py`). Parser failures count as IoU 0.

Metrics: IoU of the first box, Acc@0.5 / @0.3, best-box IoU, COCO mAP / AP50 / AP75. The COCO
metrics are class-agnostic and use the coordinate-token confidence as the score. We also report the
parse-failure breakdown, token counts and seconds per image.

## Decisions taken autonomously (2026-09-26, user away) — review

1. **Three decoding protocols.**
   * `thinking`: the official default (`<think>…</think>` then the answer, up to 8192 tokens). This is
     the "true baseline" of the spec. It costs ~4–6 min/image at 11.5 tok/s (NF4 on one A100).
   * `nothink` (free): the official template switch `enable_thinking=False`. **Finding:** the Thinking
     checkpoint ignores the empty think block. It keeps reasoning in plain text (400–800 tokens),
     closes with its own `</think>`, and then usually answers `**Bounding Box**: [x1, y1, x2, y2]`
     instead of the official JSON. Strict parsing counts this as a failure (first 8 frames: strict
     mIoU 0.13, lenient 0.37).
   * `forced` (**primary for the controlled matrix**): `enable_thinking=False` plus the official answer
     forced up to the first coordinate (```` ```json\n[\n\t{"bbox_2d": [ ````, `grounding.FORCED_PREFIX`).
     The model writes only the numbers (~20 tokens, a few minutes per condition). This is also exactly the
     steering training format. Without it, (a) formatting differences would dominate MODE 0 vs 1 vs 2,
     and (b) the steering would spend capacity changing the frozen LLM's answer format through the image
     tokens, a confound that is not perception.
   * Every summary also reports a **lenient** score (first `[x1,y1,x2,y2]` quadruple in the answer),
     to separate format failures from localisation failures. Records are re-parsed from `raw_text`
     with the current parser at summary time.
2. **Steering objective.** Cross-entropy on the coordinates of the official answer under the `forced`
   protocol (the forced prefix is not supervised), back-propagated through the frozen Thinker to the
   vision-side steering. It uses the model's own output channel, with no extra head. SteerViT's original
   objective (patch-level referring segmentation) needs masks (BUV has none), so it can be added as an
   auxiliary objective with a mask dataset. Evaluation texts for the trained steering: the training text,
   the wrong-organ text and the neutral text (text-dependence test).
3. **Steering hyper-parameters.** Blocks 1,3,…,25 (every other, as SteerViT); a LayerNorm on the queries
   (Omni block scale grows from 0.3 to 23 with depth); text = frozen RoBERTa-large, L2-normalised,
   2-layer MLP (as SteerViT); AdamW lr 2e-4 (gates 2e-3), 100 warm-up steps, cosine decay, 1500 steps ×
   batch 4 (2 × 2 accumulation), every 5th train frame (3779 frames). Trainable weights are fp32; the backbone stays frozen.
4. **No model selection on BUV val.** BUV has only train/val, and val is our test set. We use a fixed
   step budget; 10% of the train videos (414 frames) are held out only to monitor the loss.
5. **Guideline text.** `configs/omni/guidelines/breast_us_birads.txt` is my own paraphrase of the
   ACR BI-RADS US lexicon, not a verbatim quote. The controls are `neutral_control.txt` and
   `thyroid_tirads_wrong.txt`, matched in length.
6. **Precision.** NF4 is validated against BF16 (CPU offload) by teacher forcing the NF4 traces and
   letting BF16 write the answer from the same context (`validate_precision.py`).

## Not done yet / open

* **Segmentation**: the decoder, the taps and the design are ready (`omni_segmentation_design.md`).
  Mask data is now available: the 9-dataset ultrasound suite (`us_benchmark_suite.md`, built
  2026-09-26), which also gives detection boxes beyond BUV. Not yet run.
* Omni-native text ablation (`ClinicalTextEncoder(source="omni")`): implemented, not yet run.
* Domain adaptation DA2–DA5: only after the steering results.
* FlashAttention-2 is not installed; runs use SDPA.

Note on size: the steering has 71.6 M trainable parameters (0.23% of the 31.7 B Thinker, 13% of the
0.54 B vision encoder). That is larger than SteerViT's ~21 M, because the Omni ViT is wider (1152) and
deeper (27 blocks). This is exactly why the capacity control is required.

## Suite experiments (added 2026-09-26)

* **Segmentation readout** (`scripts/omni/run_seg_suite.sh`, CPU, float32 vision features): per
  us_bench dataset and structure, the GT-box oracle and a full-image lower bound; decoders are saved.
* **Grounding on the suite** (queue 3, GPU, forced protocol, MODE 0): test split, only images
  containing the structure, one prompt per structure (`Locate the object: <structure>.`). The parsed
  box then feeds the saved GT-box decoder (end-to-end condition, `--eval-pred`).
* **Literature reference on BUV** (supervised, trained on BUV, 2-class AP; see the STNet paper,
  arXiv:2309.04702): Faster R-CNN 49.2 AP50, RetinaNet 50.4, CVA-Net 65.1, STNet 70.3. Frozen zero-shot
  Omni (class-agnostic, 1 box per image, 111 frames): AP 9.2 / AP50 24.5 / AP75 5.7. This is not a
  like-for-like comparison.
