import os
import json
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
_TRAIN_META = os.path.join(_DATA_ROOT, "train_metadata.json")


def _ensure_data():
    if not os.path.exists(_TRAIN_META):
        raise RuntimeError(
            f"Herbarium 2022 data not found at {_DATA_ROOT}.\n"
            "Run the download script first:\n"
            "  sbatch download_herbarium2022.sh\n"
            "or manually:\n"
            f"  mkdir -p {_DATA_ROOT} && cd {_DATA_ROOT} && "
            "kaggle competitions download -c herbarium-2022-fgvc9 && "
            "unzip herbarium-2022-fgvc9.zip"
        )


def _herbarium2022_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> Iterable:
    _ensure_data()

    with open(_TRAIN_META) as f:
        data = json.load(f)

    cat_map = {c["category_id"]: c for c in data["categories"]}
    ann_by_img = {a["image_id"]: cat_map[a["category_id"]] for a in data["annotations"]}

    all_images = data["images"]
    shard_imgs = all_images[shard_id::total_shards]

    if shuffle:
        import random
        rng = random.Random(shuffle_seed)
        shard_imgs = list(shard_imgs)
        rng.shuffle(shard_imgs)

    yielded = 0
    for img_meta in shard_imgs:
        if yielded >= num_samples:
            break

        cat = ann_by_img.get(img_meta["image_id"], {})
        kingdom = cat.get("kingdom")

        if categories is not None and kingdom not in categories:
            continue

        file_name = img_meta["file_name"]
        img_path = os.path.join(_DATA_ROOT, "train_images", file_name)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  Skipping {img_path}: {exc}")
            continue

        yield {
            "image":     pil_img,
            "species":   cat.get("scientificName"),
            "kingdom":   kingdom,
            "family":    cat.get("family"),
            "genus":     cat.get("genus"),
            "file_name": file_name,
        }
        yielded += 1


HERBARIUM2022_CONFIG = DatasetConfig(
    name="Herbarium-2022-FGVC9",
    load_fn=_herbarium2022_stream,
    get_image=lambda s: s["image"],
    get_prompt=lambda _: "plant",
)
