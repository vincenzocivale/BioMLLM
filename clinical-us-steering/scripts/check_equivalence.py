"""STEP 5: with all gates at 0 the steered model equals the frozen backbone (real weights)."""
import torch

import _path  # noqa: F401
from common import build_model, device, fold_datasets, load_configs, load_manifest


def main():
    data_cfg, model_cfg, cfg = load_configs()
    df = load_manifest(data_cfg)
    dev = device()
    for bb in ("usfmae", "dinov2"):
        run_cfg = {"backbone": bb, "steering": True}
        model = build_model(data_cfg, model_cfg, run_cfg, df).to(dev).eval()
        _, _, ds = fold_datasets(df, data_cfg, model_cfg, run_cfg, 0, 0)
        x = torch.stack([ds[i]["image"] for i in range(16)]).to(dev)
        with torch.no_grad():
            ref = model.backbone(x)
            z = model.encode(x, list(model_cfg["queries"].values()) + [""])
        err = max((v - ref).abs().max().item() for v in z.values())
        print(f"{bb}: gates={list(model.controller.gates().values())} max|z_steered - z_backbone|={err:.2e}")
        assert err < 1e-5


if __name__ == "__main__":
    main()
