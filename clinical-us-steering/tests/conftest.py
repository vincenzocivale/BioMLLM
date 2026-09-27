import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from steering import ClinicalSteeringModel  # noqa: E402
from text_encoder import RandomTextEncoder  # noqa: E402
from vision_backbone import build_backbone  # noqa: E402

ATTRS = {"shape": ["oval", "round", "irregular"], "margin": ["a", "b"],
         "echo_pattern": ["x", "y", "z"]}
QUERIES = {"shape": "shape of the breast lesion", "margin": "margin of the breast lesion",
           "echo_pattern": "echo pattern of the breast lesion"}


def tiny_model(steering=True, seed=0, layers=(1, 3)):
    torch.manual_seed(seed)
    bb = build_backbone({"kind": "tiny_random", "image_size": 64, "pool": "mean", "depth": 4},
                        pretrained=False)
    return ClinicalSteeringModel(bb, RandomTextEncoder(dim=32), ATTRS, QUERIES, steering,
                                 {"layers": list(layers), "num_heads": 4}, 0.0)


@pytest.fixture
def model():
    return tiny_model()
