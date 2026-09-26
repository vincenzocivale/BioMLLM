"""First H2 data point: C0 (no expert) vs C3 (rad_dino) on VQA-RAD yes/no questions.

Unlike segmentation, a standalone specialist encoder (no language head) cannot do this task
at all -- so if C0 alone gets meaningfully above the 52% majority-class baseline, that's
already evidence the MLLM route does something the encoder-plus-probe route structurally
cannot. C3 vs C0 then asks whether the expert adds anything on top of that.

Simplification: batch_size=1 (see QwenVLAdapter._vqa_forward), yes/no subset only. Eval is
closed-set: compare the teacher-forced loss of the answer "yes" vs "no" and pick the lower
one, rather than free-form generation.

    python scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa condition=c0_none     +run_name=vqa_c0
    python scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa condition=c3_rad_dino +run_name=vqa_c3
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from biomllm.data.datasets.vqa import VQARadYesNo
from biomllm.models.build import build_model
from biomllm.training.param_groups import param_summary, trainable_parameters

log = logging.getLogger(__name__)


def param_groups(model, lr: float, conditioner_lr_scale: float) -> list[dict]:
    """The conditioner (projector + gate) is a separate, usually noisier, param group from
    the task tokens/head, so it can use a lower LR without slowing down the rest."""
    task_ids = {id(p) for p in model.mllm.task_parameters().values()}
    task_params = [p for p in trainable_parameters(model) if id(p) in task_ids]
    other_params = [p for p in trainable_parameters(model) if id(p) not in task_ids]
    groups = [{"params": task_params, "lr": lr}]
    if other_params:
        groups.append({"params": other_params, "lr": lr * conditioner_lr_scale})
    return groups


@torch.no_grad()
def predict(model, images: torch.Tensor, question: str) -> str:
    losses = {}
    for cand in ("yes", "no"):
        out = model(images, task="vqa", batch={"question": [question], "answer": [cand]})
        losses[cand] = out["lm_loss"].item()
    return min(losses, key=losses.get)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("run_name=%s", cfg.get("run_name", "run"))
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device

    train_ds = VQARadYesNo(split="train", image_size=cfg.mllm.image_size)
    val_ds = VQARadYesNo(split="test", image_size=cfg.mllm.image_size)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    model = build_model(cfg).to(device)
    log.info("parameters: %s", json.dumps(param_summary(model), indent=2))
    opt = torch.optim.AdamW(param_groups(model, cfg.train.lr, cfg.get("conditioner_lr_scale", 1.0)))

    run_name = cfg.get("run_name", "run")
    wandb.init(project=cfg.get("wandb_project", "biomllm-vqa-c0-vs-c3"), name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True))

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        correct, majority_correct, n = 0, 0, 0
        for i in range(len(val_ds)):
            item = val_ds[i]
            images = item["image"][None].to(device)
            pred = predict(model, images, item["question"])
            correct += int(pred == item["answer"])
            majority_correct += int(item["answer"] == "no")  # majority class in this split
            n += 1
        model.train()
        return {"accuracy": correct / n, "majority_baseline": majority_correct / n, "n": n}

    eval_every = cfg.get("eval_every", 0)
    curve = []

    step = 0
    model.train()
    while step < cfg.train.max_steps:
        for batch in train_loader:
            if step >= cfg.train.max_steps:
                break
            images = batch["image"].to(device)
            out = model(images, task="vqa", batch={"question": batch["question"], "answer": batch["answer"]})
            loss = out["lm_loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                log.info("step %d loss %.4f", step, loss.item())
                wandb.log({"train/loss": loss.item()}, step=step)
            if eval_every and step % eval_every == 0:
                val = evaluate()
                curve.append({"step": step, "val": val})
                log.info("step %d val %s", step, val)
                wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)

    result = evaluate()
    log.info("val: %s", result)
    wandb.log({f"final/{k}": v for k, v in result.items()})
    wandb.finish()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"run_name": run_name, "val": result, "curve": curve}, indent=2))
    # Conditioner + task parameters only (the MLLM and the expert are frozen): enough to
    # rebuild the trained model, e.g. for scripts/probe_llm_hidden.py.
    task_keys = {k for k, v in model.state_dict(keep_vars=True).items()
                 if id(v) in {id(p) for p in model.mllm.task_parameters().values()}}
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if k.startswith("conditioner.") or k in task_keys}, out_dir / "trainable.pt")


if __name__ == "__main__":
    main()
