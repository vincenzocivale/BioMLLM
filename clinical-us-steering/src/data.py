"""BrEaST canonical manifest, leakage-safe folds and the image dataset."""
from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ATTRIBUTES = ["shape", "orientation", "margin", "echo_pattern", "posterior_features"]
MANIFEST_COLUMNS = ["sample_id", "patient_id", "lesion_id", "image_path", *ATTRIBUTES,
                    "birads", "malignancy", "source"]

_ECHO = {"anechoic": "anechoic", "hyperechoic": "hyperechoic",
         "complex cystic/solid": "complex_cystic_solid", "hypoechoic": "hypoechoic",
         "isoechoic": "isoechoic", "heterogeneous": "heterogeneous"}
_POSTERIOR = {"no": "none", "enhancement": "enhancement", "shadowing": "shadowing",
              "combined": "combined"}


def _margin(raw) -> str | None:
    if not isinstance(raw, str) or raw == "not applicable":
        return None
    if raw == "circumscribed":
        return "circumscribed"
    subtypes = raw.split(" - ", 1)[1].split("&")
    return "not_circumscribed_indistinct" if subtypes == ["indistinct"] else "not_circumscribed_other"


def _clean(raw, mapping=None):
    if not isinstance(raw, str) or raw in ("not applicable", "not available"):
        return None
    return mapping[raw] if mapping is not None else raw


def mask_box(mask_path: Path) -> list[int] | None:
    m = np.array(Image.open(mask_path).convert("L")) > 127
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def phash(path: Path) -> str:
    return str(imagehash.phash(Image.open(path).convert("L")))


def build_breast_manifest(cfg: dict) -> pd.DataFrame:
    raw = Path(cfg["raw_dir"])
    img_dir = raw / cfg["images_subdir"]
    clin = pd.read_excel(raw / cfg["clinical_file"])
    rows = []
    for r in clin.itertuples(index=False):
        sid = f"breast_{int(r.CaseID):03d}"
        mask = img_dir / r.Mask_tumor_filename if isinstance(r.Mask_tumor_filename, str) else None
        cls = r.Classification
        rows.append({
            "sample_id": sid,
            # BrEaST: one image per patient and one annotated lesion per image
            "patient_id": sid, "lesion_id": sid,
            "image_path": str(img_dir / r.Image_filename),
            "shape": _clean(r.Shape),
            "orientation": None,  # not annotated in BrEaST
            "margin": _margin(r.Margin),
            "echo_pattern": _clean(r.Echogenicity, _ECHO),
            "posterior_features": _clean(r.Posterior_features, _POSTERIOR),
            "birads": str(r.BIRADS),
            "malignancy": {"benign": 0, "malignant": 1}.get(cls),  # 'normal' -> null
            "source": cfg["source"],
            "margin_raw": _clean(r.Margin), "classification_raw": cls,
            "pathology": _clean(r.Diagnosis), "verification": _clean(r.Verification),
            "mask_path": str(mask) if mask is not None else None,
            "lesion_box": mask_box(mask) if mask is not None else None,
            "phash": phash(img_dir / r.Image_filename),
        })
    df = pd.DataFrame(rows)
    df["malignancy"] = df["malignancy"].astype("Int64")
    for a, classes in cfg["attributes"].items():
        bad = set(df[a].dropna()) - set(classes)
        if bad:
            raise ValueError(f"{a}: labels {bad} not in the configured label space")
    return df


def near_duplicate_groups(df: pd.DataFrame, max_dist: int) -> list[int]:
    """Union-find over perceptual-hash distance so near-duplicates share a split group."""
    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    patient_first = {}
    for i, p in enumerate(df["patient_id"]):
        parent[find(i)] = find(patient_first.setdefault(p, i))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if hashes[i] - hashes[j] <= max_dist:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(df))]


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int, max_dist: int) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = near_duplicate_groups(df, max_dist)
    strat = df["malignancy"].fillna(-1).astype(int)
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(df, strat, df["group_id"])):
        df.iloc[test_idx, df.columns.get_loc("fold")] = k
    return df


def partition(df: pd.DataFrame, fold: int, n_folds: int) -> pd.Series:
    """'test' = fold k, 'val' = fold k+1, 'train' = the rest."""
    part = pd.Series("train", index=df.index)
    part[df["fold"] == fold] = "test"
    part[df["fold"] == (fold + 1) % n_folds] = "val"
    return part


def dataset_stats(df: pd.DataFrame) -> dict:
    stats = {"n_images": len(df), "n_patients": int(df["patient_id"].nunique()),
             "n_lesions": int(df["lesion_id"].nunique()),
             "sources": df["source"].value_counts().to_dict(),
             "malignancy": df["malignancy"].map({0: "benign", 1: "malignant"})
             .fillna("null").value_counts().to_dict(),
             "birads": df["birads"].value_counts().to_dict(), "attributes": {}}
    for a in ATTRIBUTES:
        stats["attributes"][a] = {"missing_pct": round(100 * df[a].isna().mean(), 2),
                                  "counts": df[a].value_counts().to_dict()}
    stats["margin_raw"] = df["margin_raw"].value_counts().to_dict()
    if "fold" in df:
        stats["folds"] = {int(k): {"n": int((df.fold == k).sum()),
                                   "malignant": int((df[df.fold == k].malignancy == 1).sum())}
                          for k in sorted(df.fold.unique())}
    return stats


def label_matrix(df: pd.DataFrame, attributes: dict) -> np.ndarray:
    """(N, A) class indices, -1 where the label is missing."""
    out = np.full((len(df), len(attributes)), -1, dtype=np.int64)
    for j, (a, classes) in enumerate(attributes.items()):
        idx = {c: i for i, c in enumerate(classes)}
        out[:, j] = [idx[v] if isinstance(v, str) else -1 for v in df[a]]
    return out


class BreastDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, attributes: dict, image_cfg: dict, size: int,
                 mean, std, train: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.labels = label_matrix(self.df, attributes)
        self.cfg, self.size, self.train = image_cfg, size, train
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.images = [Image.open(p).convert("L") for p in self.df["image_path"]]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _box(self, i, w, h):
        box = self.df.at[i, "lesion_box"]
        if self.cfg["crop"] == "full" or box is None:
            return [0, 0, w, h]
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = x1 - x0, y1 - y0
        c = self.cfg["context"]
        x0, x1 = x0 - c * bw, x1 + c * bw
        y0, y1 = y0 - c * bh, y1 + (c + self.cfg["posterior_extra"]) * bh
        if self.train and self.cfg["train_jitter"] > 0:
            j = self.cfg["train_jitter"]
            s = 1 + self.rng.uniform(-j, j)
            dx, dy = self.rng.uniform(-j, j) * bw, self.rng.uniform(-j, j) * bh
            cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
            hw, hh = s * (x1 - x0) / 2, s * (y1 - y0) / 2
            x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
        return [max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))]

    def __getitem__(self, i):
        img = self.images[i]
        img = img.crop(self._box(i, *img.size))
        # pad to square before resizing: keeps the lesion's aspect ratio (shape, orientation)
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas.resize((self.size, self.size), Image.BICUBIC)
        if self.train and self.cfg["hflip"] and self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None].repeat(3, 1, 1)
        x = (x - self.mean) / self.std
        mal = self.df.at[i, "malignancy"]
        return {"image": x, "labels": torch.from_numpy(self.labels[i]),
                "malignancy": torch.tensor(-1 if pd.isna(mal) else int(mal)), "index": i}
