"""First H1 data point: C0 (no expert) vs C3 (rad_dino) on chestx_det segmentation.

Not the general trainer (src/biomllm/training/trainer.py, still a todo) -- a minimal,
single-task loop just for this comparison, reusing build_model / param_groups / metrics.

Simplification: the 13 ChestX-Det lesion classes are collapsed to one binary "any lesion"
target, matching the adapter's current single-channel seg head (multi-class task tokens are
future work). Masks are loaded at 256px, the same resolution scripts/probe_experts.py
evaluates at, so Dice here is comparable to the Experiment 0 probe numbers -- NOT at the
model's native 16x16 grid, whose Dice is inflated (coarser pixels hide boundary error).
For the loss, the 256px target is average-pooled down to 16x16 (soft coverage per patch);
for the val metric, the 16x16 logits are bilinearly upsampled to 256px before thresholding.


    python scripts/train_seg_c0_vs_c3.py mllm=qwen_vl condition=c0_none     +run_name=c0
    python scripts/train_seg_c0_vs_c3.py mllm=qwen_vl condition=c3_rad_dino +run_name=c3
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from biomllm.data.datasets.segmentation import MultiLabelSegmentationFolder
from biomllm.evaluation.metrics import BinarySegMetrics
from biomllm.models.build import build_model
from biomllm.models.conditioning.conditioner import InjectionMode, InjectionPoint
from biomllm.models.types import TaskQueries
from biomllm.training.param_groups import param_summary, trainable_parameters

log = logging.getLogger(__name__)

CLASSES = ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule",
          "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule",
          "Pleural Thickening", "Pneumothorax"]


def collapse_binary(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch]).any(1).float()  # [B, h, w]
    return {"image": images, "mask": masks}


def seg_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    inter = (probs * target).sum((1, 2))
    denom = probs.sum((1, 2)) + target.sum((1, 2))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("run_name=%s", cfg.get("run_name", "run"))
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    root = f"{os.environ.get('BIOMLLM_DATA', 'data')}/chestx_det"

    grid = cfg.mllm.image_size // 32  # qwen_vl: patch16 * spatial_merge2
    mask_size = 256  # matches scripts/probe_experts.py, so Dice is comparable across the two
    train_ds = MultiLabelSegmentationFolder(root, CLASSES, split_file="train.txt",
                                            image_size=cfg.mllm.image_size, mask_size=mask_size)
    val_ds = MultiLabelSegmentationFolder(root, CLASSES, split_file="val.txt",
                                          image_size=cfg.mllm.image_size, mask_size=mask_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=collapse_binary, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, collate_fn=collapse_binary,
                            num_workers=4)

    model = build_model(cfg).to(device)
    log.info("parameters: %s", json.dumps(param_summary(model), indent=2))
    opt = torch.optim.AdamW(trainable_parameters(model), lr=cfg.train.lr)

    run_name = cfg.get("run_name", "run")
    wandb.init(project=cfg.get("wandb_project", "biomllm-seg-c0-vs-c3"), name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True))

    eval_every = cfg.get("eval_every", 0)
    curve = []

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        metrics = BinarySegMetrics()
        for batch in val_loader:
            images = batch["image"].to(device)
            target = batch["mask"].to(device)
            logits = model(images, task="seg")["mask_logits"][:, 0].float()
            logits_256 = F.interpolate(logits[:, None], size=target.shape[-2:], mode="bilinear",
                                       align_corners=False)[:, 0]
            metrics.update(logits_256 > 0, target.bool())
        model.train()
        return metrics.compute()

    step = 0
    model.train()
    while step < cfg.train.max_steps:
        for batch in train_loader:
            if step >= cfg.train.max_steps:
                break
            images = batch["image"].to(device)
            target_256 = batch["mask"].to(device)
            logits = model(images, task="seg")["mask_logits"][:, 0].float()
            target = F.adaptive_avg_pool2d(target_256[:, None], logits.shape[-2:])[:, 0]
            loss = seg_loss(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                log.info("step %d loss %.4f", step, loss.item())
                wandb.log({"train/loss": loss.item()}, step=step)
            if eval_every and step % eval_every == 0:
                val = evaluate()
                diag = diagnose_conditioner(model, val_loader, device)
                point = {"step": step, "val": val, "conditioner": diag}
                curve.append(point)
                log.info("step %d val %s conditioner %s", step, val, diag)
                log_payload = {f"val/{k}": v for k, v in val.items()}
                if diag:
                    log_payload.update({f"conditioner/{k}": v for k, v in diag.items()})
                wandb.log(log_payload, step=step)

    result = evaluate()
    log.info("val: %s", result)

    diag = diagnose_conditioner(model, val_loader, device)
    if diag:
        log.info("conditioner diagnostics: %s", diag)

    wandb.log({f"final/{k}": v for k, v in result.items()})
    if diag:
        wandb.log({f"final_conditioner/{k}": v for k, v in diag.items()})
    wandb.finish()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"run_name": run_name, "val": result, "conditioner": diag, "curve": curve},
        indent=2))


@torch.no_grad()
def diagnose_conditioner(model, val_loader, device) -> dict | None:
    """||alpha * P(F^S)|| vs ||F^MLLM|| where the correction is applied: in the task tokens
    (pre_llm / post_llm), or in the native visual tokens (native). If the correction dwarfs
    the native term, the model has learned to ignore the frozen MLLM's own visual signal and
    is effectively just routing the specialist features through the LLM as a pass-through,
    rather than genuinely correcting/conditioning them. For prepend variants the correction
    is a set of extra tokens, compared with the native tokens they sit next to."""
    if model.conditioner is None:
        return None
    was_training = model.training
    model.eval()
    cond = model.conditioner
    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    visual = model.mllm.visual_features(images)
    queries = model.mllm.build_task_queries("seg", visual, {})
    feats = model.conditioning_features(images, visual, {})
    target = TaskQueries(visual.tokens, grid=visual.grid) if cond.injection_point is InjectionPoint.NATIVE else queries
    if cond.mode is InjectionMode.PREPEND:
        correction = cond._extra_tokens(target.tokens, target.grid, feats)
    else:
        correction = cond.correction(target, feats)
    alpha = cond.gate(target.tokens)
    native_norm = visual.tokens.float().norm(dim=-1).mean().item()
    task_embed_norm = (queries.tokens - queries.native).float().norm(dim=-1).mean().item()
    corr_norm = correction.float().norm(dim=-1).mean().item()
    alpha_val = alpha.reshape(-1).float().mean().item() if torch.is_tensor(alpha) else float(alpha)
    model.train(was_training)
    return {
        "alpha": alpha_val,
        "native_norm": native_norm,       # ||F^MLLM|| per visual token
        "e_task_norm": task_embed_norm,   # ||e_task|| (should match native_norm's scale)
        "correction_norm": corr_norm,     # ||alpha * P(F^S)|| per corrected / extra token
        "correction_over_native": corr_norm / max(native_norm, 1e-8),
    }

if __name__ == "__main__":
    main()
