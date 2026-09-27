"""STEP 1-2: canonical BrEaST manifest, statistics and leakage-safe folds.

(Replaces prepare_buscot.py: BUS-CoT is not available locally.)
"""
import json

import _path  # noqa: F401
from data import ATTRIBUTES, MANIFEST_COLUMNS, assign_folds, build_breast_manifest, dataset_stats
from utils import ROOT, load_yaml, save_json


def main():
    cfg = load_yaml("configs/data.yaml")
    df = build_breast_manifest(cfg)
    df = assign_folds(df, cfg["split"]["n_folds"], cfg["split"]["seed"],
                      cfg["split"]["phash_max_distance"])
    merged = df.groupby("group_id")["patient_id"].nunique()
    stats = dataset_stats(df)
    stats["near_duplicate_groups_merging_patients"] = int((merged > 1).sum())
    stats["n_split_groups"] = int(df["group_id"].nunique())
    # leakage check: a split group never spans two folds
    assert df.groupby("group_id")["fold"].nunique().max() == 1
    assert df.groupby("patient_id")["fold"].nunique().max() == 1

    out = ROOT / cfg["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df[MANIFEST_COLUMNS + [c for c in df.columns if c not in MANIFEST_COLUMNS]]
    df.to_parquet(out, index=False)
    save_json(stats, ROOT / cfg["stats_dir"] / "dataset_stats.json")
    print(json.dumps(stats, indent=2, default=str))
    print(f"manifest -> {out} ({len(df)} rows; attributes {ATTRIBUTES})")


if __name__ == "__main__":
    main()
