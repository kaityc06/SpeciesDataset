#!/usr/bin/env python3
"""Validate FishNet coarse prompt coverage and official row/image resolution."""

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

    spec = importlib.util.spec_from_file_location("fishnet_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _load_config()
    mapping = config.FISHNET_CLASS_MAPPING
    if mapping != {"Fish": "fish"}:
        print("FISHNET_CLASS_MAPPING must be exactly {'Fish': 'fish'}")
        raise SystemExit(1)

    rows = config._official_rows()
    if len(rows) != config.TARGET_SAMPLES:
        print(f"expected {config.TARGET_SAMPLES} rows, found {len(rows)}")
        raise SystemExit(1)

    image_index = config._build_image_index()
    bbox_index = config._load_bbox_index()
    if not bbox_index:
        print("FishNet bbox annotations parsed to an empty index.")
        raise SystemExit(1)

    missing_images = []
    bad_class = []
    bbox_rows = 0
    bbox_boxes = 0

    for idx, row in enumerate(rows):
        path = config._resolve_image_path(row, image_index)
        if not path:
            missing_images.append(
                f"{row.get('split')}:{row.get('Folder')}/{config._image_basename(row)}"
            )
            continue

        sample = config._sample_from_row(row, path, idx)
        if config.CONFIG.get_class(sample) != "Fish":
            bad_class.append(sample.get("image_id", str(idx)))

        boxes = config._raw_bboxes_for_row(row, bbox_index)
        if boxes:
            bbox_rows += 1
            bbox_boxes += len(boxes)

    if missing_images or bad_class:
        if missing_images:
            print(f"missing image resolutions: {len(missing_images)}")
            print("\n".join(missing_images[:20]))
        if bad_class:
            print(f"non-Fish class values: {len(bad_class)}")
            print("\n".join(bad_class[:20]))
        raise SystemExit(1)

    print(f"bbox coverage: {bbox_rows}/{len(rows)} rows, {bbox_boxes} boxes")
    print(f"coverage: {len(rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
