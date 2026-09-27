"""H1 on detection: C0 (no expert) vs C3 (SonoBase) on BUV breast-lesion detection.

Frame-level (no temporal aggregation): each ultrasound frame is detected independently, with
Q learned [DET] query tokens read by a frozen Qwen3-VL and decoded by class / box heads,
trained with the Hungarian set loss (biomllm/training/detection.py). Metrics are COCO
mAP@[0.5:0.95] / AP50 / AP75 on the official CVA-Net val split (37 videos, 4504 frames),
the same protocol as CVA-Net (36.8 mAP) -- see scripts/prepare_data/buv.py for the split and
the leakage fix. The detection queries have no patch grid, so the conditioned runs use
projector=cross_attn (the queries attend to F^S).

    python scripts/train_det_c0_vs_c3.py mllm=qwen_vl_8b task=det train=det_buv \
        condition=c0_none +run_name=det_c0
    python scripts/train_det_c0_vs_c3.py mllm=qwen_vl_8b task=det train=det_buv \
        condition=c3_sonobase projector=cross_attn +run_name=det_c3
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import wandb
from biomllm.data.datasets.detection import DetectionFrames, detection_collate
from biomllm.evaluation.detection import CocoDetectionEvaluator
from biomllm.models.build import build_model
from biomllm.training.detection import SetCriterion, postprocess
from biomllm.training.param_groups import param_summary, trainable_parameters

log = logging.getLogger(__name__)


def _to(targets: list[dict], device) -> list[dict]:
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_name = cfg.get("run_name", "run")
    log.info("run_name=%s", run_name)
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    root = cfg.get("data_root", f"{os.environ.get('BIOMLLM_DATA', 'data')}/buv")

    size = cfg.mllm.image_size
    train_ds = DetectionFrames(root, "train", image_size=size, hflip=cfg.train.get("hflip", False),
                               max_frames=cfg.get("max_train_frames"))
    val_ds = DetectionFrames(root, "val", image_size=size, max_frames=cfg.get("max_val_frames"))
    classes = train_ds.classes
    if cfg.mllm.get("num_classes", len(classes)) != len(classes):
        raise ValueError(f"mllm.num_classes={cfg.mllm.num_classes} but the dataset has {classes}")
    workers = cfg.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=detection_collate, num_workers=workers, drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, collate_fn=detection_collate,
                            num_workers=workers)

    model = build_model(cfg).to(device)
    log.info("parameters: %s", json.dumps(param_summary(model), indent=2))
    criterion = SetCriterion(len(classes), **cfg.task.losses)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=cfg.train.lr,
                            weight_decay=cfg.train.get("weight_decay", 0.0))
    max_steps, warmup = cfg.train.max_steps, cfg.train.get("warmup_steps", 0)

    def lr_factor(step: int) -> float:  # linear warmup, then cosine to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(max_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    wandb.init(project=cfg.get("wandb_project", "biomllm-det-c0-vs-c3"), name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True))

    @torch.no_grad()
    def evaluate(loader=val_loader) -> dict:
        model.eval()
        ev = CocoDetectionEvaluator(classes)
        for batch in loader:
            out = model(batch["image"].to(device), task="det")
            ev.update(postprocess(out["logits"], out["boxes"]), batch["targets"], batch["size"])
        model.train()
        return ev.compute()

    eval_every = cfg.get("eval_every", 0)
    grad_clip = cfg.train.get("grad_clip", 0.0)
    curve = []
    step = 0
    model.train()
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            out = model(batch["image"].to(device), task="det")
            losses = criterion(out["logits"], out["boxes"], _to(batch["targets"], device))
            opt.zero_grad()
            losses["loss"].backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0:
                log.info("step %d %s", step, {k: round(v.item(), 4) for k, v in losses.items()})
                wandb.log({f"train/{k}": v.item() for k, v in losses.items()}
                          | {"train/lr": sched.get_last_lr()[0]}, step=step)
            if eval_every and step % eval_every == 0:
                val = evaluate()
                diag = diagnose_conditioner(model, val_loader, device)
                curve.append({"step": step, "val": val, "conditioner": diag})
                log.info("step %d val %s conditioner %s", step, val, diag)
                payload = {f"val/{k}": v for k, v in val.items()}
                if diag:
                    payload.update({f"conditioner/{k}": v for k, v in diag.items()})
                wandb.log(payload, step=step)

    result = evaluate()
    log.info("val: %s", result)
    diag = diagnose_conditioner(model, val_loader, device)
    if diag:
        log.info("conditioner diagnostics: %s", diag)

    # Test-time ablations of the trained conditioned model (same weights):
    #   shuffled  each frame is conditioned on another frame's features. The in-order val
    #             loader batches consecutive frames of one video (near-identical features),
    #             so this uses a seeded shuffled loader: partners come from other videos.
    #   alpha0    correction switched off -> what the trained queries / heads do alone
    ablations = {}
    if model.conditioner is not None:
        mixed_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(0),
                                  collate_fn=detection_collate, num_workers=workers)
        model.shuffle_features = True
        ablations["shuffled"] = evaluate(mixed_loader)
        model.shuffle_features = False
        model.conditioner.set_alpha_scale(0.0)
        ablations["alpha0"] = evaluate()
        model.conditioner.set_alpha_scale(1.0)
        log.info("ablations: %s", ablations)

    wandb.log({f"final/{k}": v for k, v in result.items()})
    for name, r in ablations.items():
        wandb.log({f"final_{name}/{k}": v for k, v in r.items()})
    if diag:
        wandb.log({f"final_conditioner/{k}": v for k, v in diag.items()})
    wandb.finish()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"run_name": run_name, "val": result, "ablations": ablations, "conditioner": diag,
         "curve": curve}, indent=2))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if k.startswith("conditioner.") or k == "static_features"
                or k in _task_param_keys(model)},
               out_dir / "trainable.pt")


def _task_param_keys(model) -> set[str]:
    ids = {id(p) for p in model.mllm.task_parameters().values()}
    return {k for k, p in model.named_parameters() if id(p) in ids}


@torch.no_grad()
def diagnose_conditioner(model, val_loader, device) -> dict | None:
    """||alpha * P(F^S)|| vs ||T_j|| for the [DET] queries (they have no F^MLLM term, so the
    reference is the learned query itself): a ratio >> 1 means the queries are mostly the
    projected expert features."""
    if model.conditioner is None:
        return None
    model.eval()
    images = next(iter(val_loader))["image"].to(device)
    visual = model.mllm.visual_features(images)
    queries = model.mllm.build_task_queries("det", visual, {})
    feats = model.conditioning_features(images, visual, {})
    correction = model.conditioner.correction(queries, feats)
    alpha = model.conditioner.gate(queries.tokens)
    query_norm = queries.tokens.float().norm(dim=-1).mean().item()
    corr_norm = correction.float().norm(dim=-1).mean().item()
    model.train()
    return {
        "alpha": alpha.reshape(-1).float().mean().item() if torch.is_tensor(alpha) else float(alpha),
        "query_norm": query_norm,
        "correction_norm": corr_norm,
        "correction_over_query": corr_norm / max(query_norm, 1e-8),
    }


if __name__ == "__main__":
    main()
