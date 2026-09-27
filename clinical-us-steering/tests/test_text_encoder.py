import os

import pytest

from utils import load_yaml, setup_env


def test_text_encoder_frozen():
    cfg = load_yaml("configs/model.yaml")
    setup_env(cfg)
    from text_encoder import FrozenTextEncoder
    try:
        enc = FrozenTextEncoder(cfg["text_encoder"]["name"])
    except OSError as e:  # weights not in the local cache
        pytest.skip(str(e))
    assert all(not p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.model.training
    tok, mask = enc.encode("margin of the breast lesion")
    assert tok.shape[-1] == enc.dim and not tok.requires_grad and mask.all()
    e_tok, e_mask = enc.encode("")
    assert e_tok.shape[1] == 2  # <s></s>
