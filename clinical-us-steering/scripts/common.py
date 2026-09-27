"""Model/data construction shared by the training and evaluation scripts."""
import pandas as pd
import torch

import _path  # noqa: F401
from data import BreastDataset, partition
from steering import ClinicalSteeringModel
from text_encoder import FrozenTextEncoder
from utils import ROOT, load_yaml, setup_env
from vision_backbone import build_backbone


def load_configs():
    return (load_yaml("configs/data.yaml"), load_yaml("configs/model.yaml"),
            load_yaml("configs/train_steering.yaml"))


def supervised_attributes(data_cfg, df):
    """Attributes with at least one label; the others keep their query but get no head."""
    return {a: c for a, c in data_cfg["attributes"].items() if df[a].notna().any()}


def label_index(data_cfg, attrs):
    """Column of each attribute in the dataset label matrix."""
    order = list(data_cfg["attributes"])
    return {a: order.index(a) for a in attrs}


def build_model(data_cfg, model_cfg, run_cfg, df, pretrained=True):
    setup_env(model_cfg)
    backbone = build_backbone(model_cfg["backbones"][run_cfg["backbone"]], pretrained)
    text = FrozenTextEncoder(model_cfg["text_encoder"]["name"],
                             model_cfg["text_encoder"]["max_length"])
    return ClinicalSteeringModel(backbone, text, supervised_attributes(data_cfg, df),
                                 model_cfg["queries"], run_cfg["steering"], model_cfg["steering"],
                                 model_cfg["heads"]["dropout"])


def load_manifest(data_cfg):
    return pd.read_parquet(ROOT / data_cfg["manifest"])


def fold_datasets(df, data_cfg, model_cfg, run_cfg, fold, seed):
    bcfg = model_cfg["backbones"][run_cfg["backbone"]]
    part = partition(df, fold, data_cfg["split"]["n_folds"])
    mk = lambda d, train: BreastDataset(d, data_cfg["attributes"], data_cfg["image"],  # noqa: E731
                                        bcfg["image_size"], bcfg["mean"], bcfg["std"], train, seed)
    return part, {s: mk(df[part == s], s == "train") for s in ("train", "val", "test")}, \
        mk(df, False)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
