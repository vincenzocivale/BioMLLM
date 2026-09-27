import pandas as pd
import pytest

from data import assign_folds, label_matrix, partition
from utils import ROOT, load_yaml

MANIFEST = ROOT / load_yaml("configs/data.yaml")["manifest"]


def test_group_split_has_no_leakage():
    # synthetic: several images per patient, two of them near-identical across patients
    df = pd.DataFrame({"patient_id": [f"p{i // 3}" for i in range(60)],
                       "malignancy": [i // 3 % 2 for i in range(60)],
                       "phash": [f"{(i // 3) * 7919:016x}" for i in range(60)]})
    df.loc[5, "phash"] = df.loc[40, "phash"]  # near-duplicate of another patient's image
    df = assign_folds(df, 5, 17, 0)
    assert df.groupby("patient_id")["fold"].nunique().max() == 1
    assert df.loc[5, "fold"] == df.loc[40, "fold"]
    for k in range(5):
        part = partition(df, k, 5)
        sets = {s: set(df.patient_id[part == s]) for s in ("train", "val", "test")}
        assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"])
        assert not (sets["val"] & sets["test"])


@pytest.mark.skipif(not MANIFEST.exists(), reason="run scripts/prepare_breast.py first")
def test_manifest_split_and_labels():
    cfg = load_yaml("configs/data.yaml")
    df = pd.read_parquet(MANIFEST)
    assert df.groupby("patient_id")["fold"].nunique().max() == 1
    assert df.groupby("group_id")["fold"].nunique().max() == 1
    assert df["orientation"].isna().all()  # never inferred
    y = label_matrix(df, cfg["attributes"])
    assert ((y == -1) == df[list(cfg["attributes"])].isna().to_numpy()).all()
