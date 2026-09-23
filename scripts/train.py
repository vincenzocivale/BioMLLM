"""Entry point.

Currently builds the model from the composed config, logs the parameter breakdown and runs a
forward/backward pass on synthetic images. The training loop (src/biomllm/training/trainer.py)
lands with the first dataset.

    python scripts/train.py +experiment=debug
    python scripts/train.py +experiment=debug conditioner=self injection=post_llm projector=cross_attn
"""

from __future__ import annotations

import json
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from biomllm.models.build import build_model
from biomllm.training.param_groups import param_summary, trainable_parameters

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device

    model = build_model(cfg).to(device)
    log.info("parameters: %s", json.dumps(param_summary(model), indent=2))

    images = torch.rand(cfg.train.batch_size, 3, 224, 224, device=device)
    preds = model(images, task=cfg.task.name)
    loss = sum(v.float().pow(2).mean() for v in preds.values())
    loss.backward()
    n_grad = sum(p.grad is not None for p in trainable_parameters(model))
    log.info("forward ok: %s | params with grad: %d",
             {k: tuple(v.shape) for k, v in preds.items()}, n_grad)


if __name__ == "__main__":
    main()
