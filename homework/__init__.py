from . import data
from .data import VQADataset, benchmark

__all__ = ["BaseVLM", "VQADataset", "benchmark", "train", "load_vlm", "load_clip", "data"]


def __getattr__(name):
    if name == "BaseVLM":
        from .base_vlm import BaseVLM

        return BaseVLM
    if name == "train":
        from .finetune import train

        return train
    if name == "load_vlm":
        from .finetune import load

        return load
    if name == "load_clip":
        from .clip import load

        return load
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
