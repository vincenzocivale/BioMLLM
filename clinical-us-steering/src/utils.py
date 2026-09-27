"""Config loading, seeding, run directories."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        return yaml.safe_load(f)


def save_yaml(obj, path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def setup_env(model_cfg: dict) -> None:
    """Point HF at the local cache; every checkpoint used here is already downloaded."""
    os.environ.setdefault("HF_HOME", model_cfg["hf_home"])
    os.environ.setdefault("HF_HUB_CACHE", str(Path(model_cfg["hf_home"]) / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_dir(root, name: str) -> Path:
    d = ROOT / root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def count_params(module: torch.nn.Module) -> dict:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}
