# Ultrasound segmentation / detection benchmark suite

Nine public 2D ultrasound datasets with pixel masks, converted to one layout. Detection boxes are
derived from the masks, so every dataset serves both tasks. Built 2026-09-26 at
`/raid/DATASETS/BioMLLMData/datasets/us_bench/` (2.2 GB). Raw archives and `download.sh` are in
`…/datasets/raw/us_bench/`.

    bash $BIOMLLM_DATA/raw/us_bench/download.sh                       # raw archives + extraction
    python scripts/prepare_data/us_bench/convert.py --datasets all    # -> us_bench/<name>/
    python scripts/prepare_data/us_bench/contact_sheet.py --out sheet.png   # visual check

## Datasets

| name | organ · target (classes) | train / val / test | boxes | split | license |
|---|---|---|---|---|---|
| `busbra` | breast · lesion | 1114 / 385 / 376 | 1875 | **official** 5-fold patient CV (fold 1 test, fold 2 val) | CC BY 4.0 |
| `busi` | breast · lesion (+133 normal) | 547 / 78 / 155 | 662 | seeded, by near-duplicate cluster | CC BY 4.0 (mirror) |
| `bus_uclm` | breast · lesion (+410 normal) | 474 / 74 / 122 | 277 | seeded, by patient | CC BY 4.0 |
| `breast_lesions` | breast · lesion (BrEaST, TCIA) | 179 / 26 / 51 | 252 | seeded, by case | CC BY 4.0 |
| `tn3k` | thyroid · nodule | 2591 / 288 / 614 | 3824 | **official** test; val = 10% of trainval | MIT (HF tag) |
| `mmotu` | ovary · tumour (8 types kept) | 900 / 100 / 469 | 1489 | **official** val → test; val = 10% of train | CC BY 4.0 (Zenodo copy) |
| `hc18` | fetal · head | 687 / 108 / 204 | 999 | seeded, by fetus | CC BY 4.0 |
| `psfhs` | intrapartum · pubic symphysis, fetal head | 934 / 146 / 278 | 2716 | seeded, by near-duplicate cluster | CC BY 4.0 |
| `camus` | cardiac · LV endocardium, myocardium, LA | 1400 / 200 / 400 | 6000 | seeded, by patient (500 patients, 2CH/4CH × ED/ES) | **CC BY-NC-SA 4.0** |

Why these: they cover five organs and three kinds of target:

* focal lesions: breast ×4, thyroid, ovary;
* anatomical structures: fetal head, pubic symphysis, cardiac chambers;
* negatives: BUSI and BUS-UCLM include normal images, so detection can measure false positives.

Four breast sets from different centres and devices, together with BUV (video, detection), allow
cross-dataset generalisation tests. CAMUS is non-commercial; that is fine for academic research,
but check before any commercial use.

## Layout (per dataset)

    images/<id>.png         RGB (grey replicated), original resolution
    masks/<id>.png          uint8 label map: 0 = background, k = classes[k-1]
    {train,val,test}.json   [{"file", "mask", "height", "width", "boxes" (xyxy px), "labels" (0-based),
                              "group", ...extra (pathology, BI-RADS, tumour class, device)}]
    classes.json, meta.json (source, license, citation, split rule, stats, leakage audit)
    test_nodup.json / val_nodup.json   only where the audit found near-duplicates (see below)

Records use the same format as BUV (`boxes` in pixels, `labels`), so the grounding runner, the COCO
evaluator and `train_mask_decoder.py` (binary: mask > 0) apply directly. Boxes: one per connected
component for lesions, one per structure for anatomy.

## Data-quality work (all recorded in each `meta.json`)

* **Leakage audit.** A dHash near-duplicate search between train and val/test. For sets without
  patient ids (BUSI, PSFHS), near-duplicates are merged into one group *before* the seeded split.
  This removed the 6 (BUSI) and 44 (PSFHS) test images that were near-copies of training images.
  Official splits are kept as published. Where they leak (MMOTU: 15 of 469 official test images are
  near-duplicates of train; CAMUS: 2 cross-patient look-alikes), `test_nodup.json` gives the clean subset.
* **HC18.** The annotations are ellipse outlines. They are filled by hole filling, or by a direct
  least-squares ellipse fit when the outline is broken (IoU 0.984 against the true ellipse in a test).
  Images `NNN_kHC` of the same fetus are grouped. The official test set has no public labels, so it is
  not used.
* **BUS-UCLM.** Patient HESN (13 images) is dropped: its images are cropped relative to the
  masks (856×606), so the alignment is unknowable. RGB masks (green benign / red malignant) become one
  lesion class, and pathology is kept per record.
* **BUSI.** Multiple `_mask_k` files are merged. The known issues (burnt-in markers, duplicates) are
  mitigated by the grouping above.
* **MMOTU.** Images and binary labels come from the Zenodo curated copy. They are pixel-identical to
  the original annotations (checked on 7 images, IoU 1.0). The official split and tumour classes
  come from the original lists.
* **PSFHS.** Label ids were verified from areas (1 = pubic symphysis, 2 = fetal head).
  The images are natively 256×256.
* **Visual check.** A contact sheet of masks and boxes for every dataset was inspected: alignment,
  orientation (CAMUS apex up) and label ids.

## Not included (and why)

DDTI (no reliable open copy with masks), Kaggle nerve segmentation (competition data,
redistribution doubtful), EchoNet-Dynamic (needs a Stanford AIMI agreement), HC18 test set
(labels not public).
