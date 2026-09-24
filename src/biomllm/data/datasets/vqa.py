"""VQA datasets: {"image": [3, H, W] float in [0, 1], "question": str, "answer": str, "id": str}."""

from __future__ import annotations

import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REPO = "flaviagiammarino/vqa-rad"
FILES = {
    "train": "data/train-00000-of-00001-eb8844602202be60.parquet",
    "test": "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet",
}


class VQARadYesNo(Dataset):
    """Closed-ended (yes/no) subset of VQA-RAD: 940 train / 251 test pairs, 512 x 512 by
    default. Restricted to yes/no so a first smoke eval can score exact-match accuracy
    without a full generation loop (see scripts/train_vqa_c0_vs_c3.py)."""

    def __init__(self, split: str = "train", image_size: int = 512):
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, FILES[split], repo_type="dataset")
        df = pd.read_parquet(path)
        self.df = df[df["answer"].isin(["yes", "no"])].reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        import numpy as np
        from PIL import Image

        row = self.df.iloc[i]
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1)
        image = F.interpolate(image[None], size=(self.image_size,) * 2, mode="bilinear",
                              align_corners=False, antialias=True)[0]
        return {"image": image, "question": row["question"], "answer": row["answer"], "id": str(i)}
