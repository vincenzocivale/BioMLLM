# Clinical Steering for Ultrasound

A minimal, falsification-oriented test of one question: can a **SteerViT-style text steering**
of a **frozen ultrasound foundation model** make its representation *selectively* informative
about the BI-RADS descriptor named in the query (shape, orientation, margin, echo pattern,
posterior acoustic features), and do the steered embeddings help benign/malignant diagnosis?

Everything outside this question (LLM guideline parsing, multi-organ, MoE, segmentation,
MLLMs) is out of scope on purpose.

## Setup actually used (read before interpreting results)

| Item | Specification | Used here | Why |
|---|---|---|---|
| Dataset | BUS-CoT | **BrEaST-Lesions_USG** (256 images/patients) | only already-downloaded datasets may be used; BrEaST is the only local breast US set with expert BI-RADS descriptors |
| Orientation | labelled | **not annotated** in BrEaST: query kept, no head | labels are never inferred |
| Backbone | USFM | **USF-MAE ViT-B/16** (`mfurkan03/USF-MAE`) | USFM weights are not on the HF hub; USF-MAE is a plain ultrasound-pretrained ViT (the specified fallback) |
| Run names | `*_usfm` | `*_usfmae` | the name records the backbone actually used |
| Split | patient/lesion level | 5-fold patient-grouped CV (fold k test, k+1 val) | 256 images: a single split would leave ~50 test images |

Details, caveats (e.g. BrEaST is in SonoCorpus, and possibly in USF-MAE's pretraining set) and results:
`EXPERIMENT_REPORT.md`.

## Model

`V'_l = V_l + tanh(alpha_l) * CrossAttn(q = V_l, k = v = P(T))` after blocks 1,3,5,7,9,11 of the
frozen ViT, with `alpha_l = 0` at init (exactly the frozen backbone, verified). One controller
(`src/steering.py`) is shared by every query, and each attribute has a linear head. Frozen: the vision backbone
and RoBERTa-large. Trainable: text projector, cross-attention, gates and heads.

## Pipeline

```bash
PY=/tmp/vcivale_envs/biomllm/bin/python
$PY scripts/prepare_breast.py                  # manifest, stats, pHash-grouped folds
$PY scripts/check_equivalence.py               # gate=0 == frozen backbone
$PY scripts/train_steering.py --run baseline_attributes_usfmae
$PY scripts/train_steering.py --run clinical_steering_usfmae
$PY scripts/evaluate_steering.py --run clinical_steering_usfmae   # selectivity, controls, gates
$PY scripts/train_diagnosis.py --run diagnosis_base_usfmae        # and the other diagnosis runs
$PY scripts/evaluate_diagnosis.py
$PY scripts/make_report.py                     # -> EXPERIMENT_REPORT_DATA.md
# or everything on one GPU:
bash scripts/run_queue.sh 17 && bash scripts/run_post.sh 17       # seeds 29/43: pass the seed
$PY -m pytest -q
```

Outputs per run: `outputs/runs/<run>/{config.yaml, metrics.json, history.csv, fold*/best.pt,
predictions.csv, steering_selectivity_matrix.csv, steering_selectivity_heatmap.png,
gate_values.csv, representation_similarity.csv}`.
