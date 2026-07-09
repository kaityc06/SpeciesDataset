#!/usr/bin/env python3
"""Validate BirdSnap coarse prompt coverage without importing SAM/Qwen."""

import importlib.util
import os
import sys
import types


DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DATASET_DIR, "config.py")


class DatasetConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_config():
    stub = types.ModuleType("species_segmentation")
    stub.DatasetConfig = DatasetConfig
    sys.modules.setdefault("species_segmentation", stub)

    spec = importlib.util.spec_from_file_location("birdsnap_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _load_config()
    mapping = config.BIRDSNAP_CLASS_MAPPING
    if mapping != {"Aves": "bird"}:
        print("BIRDSNAP_CLASS_MAPPING must be exactly {'Aves': 'bird'}")
        raise SystemExit(1)

    try:
        from datasets import get_dataset_config_info
    except ImportError:
        print("The datasets package is required to validate sasha/birdsnap.")
        raise SystemExit(1)

    info = get_dataset_config_info(config.BIRDSNAP_DATASET_PATH)
    if config.BIRDSNAP_SPLIT not in info.splits:
        print(f"Split {config.BIRDSNAP_SPLIT!r} not found in {config.BIRDSNAP_DATASET_PATH}.")
        raise SystemExit(1)
    total = info.splits[config.BIRDSNAP_SPLIT].num_examples

    features = info.features or {}
    missing_columns = [name for name in ("image", "label") if name not in features]
    if missing_columns:
        print("missing columns:", ", ".join(missing_columns))
        raise SystemExit(1)

    if total != config.TARGET_SAMPLES:
        print(f"expected {config.TARGET_SAMPLES} rows, found {total}")
        raise SystemExit(1)

    sample = {"label": "example"}
    if config.CONFIG.get_class(sample) != "Aves":
        print("CONFIG.get_class must return Aves for BirdSnap samples.")
        raise SystemExit(1)

    print(f"coverage: {total}/{total}")


if __name__ == "__main__":
    main()
