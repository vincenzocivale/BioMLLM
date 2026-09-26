from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from biomllm.models.build import build_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONDITIONS = sorted(p.stem for p in (CONFIGS / "condition").glob("*.yaml"))


def _compose(overrides):
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        return compose("config", overrides=overrides)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_composes(condition):
    cfg = _compose([f"condition={condition}"])
    assert (cfg.conditioner.source == "expert") == (cfg.expert.kind is not None)


@pytest.mark.parametrize("conditioner", ["none", "self", "noise", "specialist"])
@pytest.mark.parametrize("injection", ["pre_llm", "post_llm", "native"])
@pytest.mark.parametrize("projector", ["linear", "mlp", "cross_attn"])
def test_debug_model_builds_and_runs(conditioner, injection, projector):
    cfg = _compose(["+experiment=debug", f"conditioner={conditioner}",
                    f"injection={injection}", f"projector={projector}",
                    "conditioner.noise_dim=16" if conditioner == "noise" else "seed=0"])
    model = build_model(cfg)
    out = model(torch.rand(2, 3, 64, 64), task="seg")
    assert out["mask_logits"].shape == (2, 1, 8, 8)
