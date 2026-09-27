# Deviations from the specification and methodological notes

1. **Dataset: BrEaST-Lesions_USG instead of BUS-CoT.** Only already-downloaded datasets may be
   used, and BUS-CoT is not on disk. BrEaST (256 images, one per patient, expert BI-RADS
   descriptors, biopsy/follow-up verified) is the only local breast-US dataset with lexicon
   descriptors. This takes away the attribute-level external validation planned on BrEaST:
   no other local dataset has descriptor labels.
2. **Orientation is not annotated in BrEaST.** The column is `null` for all rows (never
   inferred). The orientation query is still encoded (a row of the selectivity matrix and a
   control), but it has no head and no loss.
3. **Label spaces.** The margin annotation is multi-label within "not circumscribed" (indistinct /
   angular / microlobulated / spiculated). It is mapped to 3 classes: `circumscribed`,
   `not_circumscribed_indistinct` (indistinct only), and `not_circumscribed_other` (any angular,
   microlobulated or spiculated component). The raw string is kept in `margin_raw`. Echo pattern
   keeps all 6 BI-RADS classes, posterior features all 4. The 4 `normal` cases have every
   descriptor `not applicable` → null, and malignancy null.
4. **Backbone: USF-MAE ViT-B/16 instead of USFM.** USFM (openmedlab) weights are not on the HF hub
   or on disk. USF-MAE (`mfurkan03/USF-MAE`, file `USF-MAE_full_pretrain_43dataset_100epochs.pt`)
   is an MAE ViT-B/16 pretrained on 43 public US datasets. It loads into timm
   `vit_base_patch16_224` with no missing or unexpected keys. The gated cross-attention sits
   between its standard transformer blocks. USF-MAE ships no preprocessing config, so ImageNet
   mean/std (the MAE default) is assumed. Run names carry `usfmae` instead of `usfm`.
   SonoBase (on disk) was not used as the primary backbone: its SAM2-style trunk is a
   Hiera + two ConvNeXt branches with cross-branch deformable attention. Injection would be
   possible after each of its 11 fusion stages, but it is not a ViT with a uniform token
   sequence, and it would need a different pooling and a different notion of "block".
5. **Pretraining overlap (not label leakage).** BrEaST, BUS-BRA and BUS-UCLM are in USF-MAE's
   self-supervised pretraining set (per the USF-MAE README). BrEaST is also in SonoBase's
   pretraining tier. The backbone has seen these images without labels. The overlap affects
   the baseline and steering conditions equally, but absolute numbers may be optimistic. BUS-BRA
   cannot serve as an external set for a USF-MAE model without this caveat.
6. **Split: 5-fold patient-grouped CV.** Fold k is test, fold k+1 is validation (early
   stopping and checkpointing), and the rest is train. Folds are stratified by malignancy.
   Groups are patients merged with perceptual-hash near-duplicates (Hamming ≤ 6). BrEaST has no
   near-duplicates, so there are 256 groups. All metrics are out-of-fold over the 252 labelled
   images, and each image is predicted by the one model that never saw it.
7. **Lesion-centred input.** The image is cropped to the GT lesion box, padded by 25% per side
   plus 50% extra below (posterior features), then padded to a square (aspect ratio preserved)
   and resized to 224. The lesion is therefore assumed localised (reader-study setting), and
   both conditions receive identical inputs. Augmentation: box jitter and horizontal flip only
   (never a vertical flip).
8. **Gate learning rate.** With ~10 steps per epoch, a zero-initialised gate under AdamW at
   lr 1e-4 could reach at most |alpha| ≈ 0.06 over the whole run. That would make steering
   inactive by construction. The gates get lr 5e-3 without weight decay; the rest of the
   controller gets lr 2e-4.
9. **Selectivity matrix.** The primary matrix trains a fresh linear probe (standardised,
   class-balanced logistic regression, C chosen by inner 3-fold CV) on the query-i embedding of
   the train+val images to predict attribute j, and evaluates it out-of-fold. A secondary matrix
   applies the trained head j to the query-i embedding ("head transfer"). The second mostly
   measures head compatibility, not information content.
10. **Diagnosis option C** (concatenated embeddings of the no-steering multi-task baseline) is
    identical to A by construction here: the baseline has a single frozen embedding shared by
    all heads. Instead, B is reported with the 4 supervised queries and B' with all 5.
