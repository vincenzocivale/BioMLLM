import torch

from attribute_heads import class_weights, masked_ce
from conftest import ATTRS, QUERIES, tiny_model


def x(n=3):
    return torch.randn(n, 3, 64, 64, generator=torch.Generator().manual_seed(1))


def test_vision_backbone_frozen(model):
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.train()
    assert not model.backbone.training


def test_gates_initialised_to_zero(model):
    assert all(v == 0.0 for v in model.controller.gates().values())
    assert all(b.alpha.item() == 0.0 for b in model.controller.blocks.values())


def test_gate_zero_reproduces_backbone(model):
    model.eval()
    with torch.no_grad():
        ref = model.backbone(x())
        z = model.encode(x(), list(QUERIES.values()) + [""])
    for v in z.values():
        torch.testing.assert_close(v, ref, rtol=0, atol=1e-6)


def test_single_controller_shared_by_queries(model):
    names = [n for n, _ in model.controller.named_parameters()]
    assert not any(a in n for n in names for a in ATTRS)  # no per-attribute controller
    with torch.no_grad():
        for b in model.controller.blocks.values():
            b.alpha.fill_(1.0)
    model.eval()
    grads = {}
    for q in QUERIES.values():
        model.zero_grad()
        model.encode(x(), [q])[q].sum().backward()
        grads[q] = {n for n, p in model.controller.named_parameters() if p.grad is not None
                    and p.grad.abs().sum() > 0}
    first = next(iter(grads.values()))
    assert first and all(g == first for g in grads.values())


def test_different_queries_shapes_and_values(model):
    with torch.no_grad():
        for b in model.controller.blocks.values():
            b.alpha.fill_(1.0)
    model.eval()
    with torch.no_grad():
        z = model.encode(x(4), list(QUERIES.values()))
        logits = model(x(4))
    assert all(v.shape == (4, model.backbone.dim) for v in z.values())
    zs = list(z.values())
    assert not torch.allclose(zs[0], zs[1])
    assert {a: tuple(v.shape) for a, v in logits.items()} == {a: (4, len(c)) for a, c in ATTRS.items()}


def test_missing_labels_do_not_contribute():
    logits = torch.randn(4, 3, requires_grad=True)
    y = torch.tensor([0, -1, 2, -1])
    loss = masked_ce(logits, y)
    loss.backward()
    assert torch.all(logits.grad[1] == 0) and torch.all(logits.grad[3] == 0)
    ref = torch.nn.functional.cross_entropy(logits[[0, 2]], y[[0, 2]])
    torch.testing.assert_close(loss, ref)
    assert masked_ce(logits, torch.full((4,), -1)) is None
    w = class_weights(torch.tensor([0, 0, 2, -1]), 3)
    assert w[1] == 0 and w[0] < w[2]


def test_trainable_params_restricted():
    for steering in (True, False):
        m = tiny_model(steering)
        names = [n for n, p in m.named_parameters() if p.requires_grad]
        assert names and all(n.startswith(("controller.", "heads.")) for n in names)
        if not steering:
            assert all(n.startswith("heads.") for n in names)


def test_checkpoint_roundtrip(tmp_path):
    m = tiny_model(seed=0)
    with torch.no_grad():
        for p in m.controller.parameters():
            p.add_(0.1 * torch.randn_like(p))
    m.eval()
    torch.save({"state": m.trainable_state_dict()}, tmp_path / "best.pt")
    m2 = tiny_model(seed=0)  # same frozen backbone init, different trainable state
    with torch.no_grad():
        for p in m2.controller.parameters():
            p.zero_()
    m2.load_state_dict(torch.load(tmp_path / "best.pt")["state"], strict=False)
    m2.eval()
    with torch.no_grad():
        a, b = m(x()), m2(x())
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_tiny_overfit():
    torch.manual_seed(0)
    m = tiny_model(layers=(0, 1, 2, 3))
    xs = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 2, (8,))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    m.train()
    for _ in range(150):
        loss = masked_ce(m(xs)["margin"], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert any(abs(v) > 0 for v in m.controller.gates().values())
