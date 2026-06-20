import os
import json
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
_TRAIN_META = os.path.join(_DATA_ROOT, "train.json")
_VAL_META = os.path.join(_DATA_ROOT, "val.json")


def _ensure_data():
    if not os.path.exists(_TRAIN_META):
        raise RuntimeError(
            f"FGVCx Fungi 2018 data not found at {_DATA_ROOT}.\n"
            "Run the download script first:\n"
            "  sbatch download_fungi.sh\n"
            "or manually:\n"
            f"  mkdir -p {_DATA_ROOT} && cd {_DATA_ROOT} && "
            "kaggle competitions download -c fungi-challenge-fgvc-2018 && "
            "unzip fungi-challenge-fgvc-2018.zip"
        )


def _fungi_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> Iterable:
    _ensure_data()

    if categories is not None and "Fungi" not in categories:
        return

    all_samples = []
    for split, meta_path in [("train", _TRAIN_META), ("val", _VAL_META)]:
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            data = json.load(f)
        cat_map = {c["id"]: c for c in data["categories"]}
        ann_by_img = {a["image_id"]: cat_map[a["category_id"]] for a in data["annotations"]}
        for img_meta in data["images"]:
            cat = ann_by_img.get(img_meta["id"], {})
            all_samples.append((split, img_meta, cat))

    shard_samples = all_samples[shard_id::total_shards]

    if shuffle:
        import random
        rng = random.Random(shuffle_seed)
        shard_samples = list(shard_samples)
        rng.shuffle(shard_samples)

    yielded = 0
    for split, img_meta, cat in shard_samples:
        if yielded >= num_samples:
            break

        file_name = img_meta["file_name"]
        img_path = os.path.join(_DATA_ROOT, split, file_name)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  Skipping {img_path}: {exc}")
            continue

        yield {
            "image":     pil_img,
            "species":   cat.get("name"),
            "file_name": file_name,
            "split":     split,
        }
        yielded += 1


FUNGI_CONFIG = DatasetConfig(
    name="FGVCx-Fungi-2018",
    load_fn=_fungi_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: "Fungi",
    class_mapping={
        "Fungi": ["fungus"],
    },
)
