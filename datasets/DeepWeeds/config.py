import os
import csv
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_IMAGES_DIR = os.path.join(_DATA_DIR, "images")
_LABELS_CSV = os.path.join(_DATA_DIR, "labels.csv")

_LABEL_TO_SPECIES = {
    "0": "Chinee apple",
    "1": "Snake weed",
    "2": "Lantana",
    "3": "Prickly acacia",
    "4": "Siam weed",
    "5": "Parthenium",
    "6": "Rubber vine",
    "7": "Parkinsonia",
    "8": "Negative",
}


def _ensure_data():
    if not os.path.exists(_LABELS_CSV) or not os.path.isdir(_IMAGES_DIR):
        raise RuntimeError(
            f"DeepWeeds data not found at {_DATA_DIR}.\n"
            "Run the download script first:\n"
            "  sbatch download_deepweeds.sh\n"
            "or manually:\n"
            f"  mkdir -p {_IMAGES_DIR} && cd {_DATA_DIR} && "
            "gdown 1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj -O deepweeds.zip && "
            "unzip deepweeds.zip -d images/ && rm deepweeds.zip && "
            "wget https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"
        )


def _deepweeds_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> Iterable:
    _ensure_data()

    rows = []
    with open(_LABELS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = _LABEL_TO_SPECIES.get(row["Label"], "unknown")
            if species == "Negative":
                continue
            rows.append((row["Filename"], species))

    shard_rows = rows[shard_id::total_shards]

    if shuffle:
        import random
        rng = random.Random(shuffle_seed)
        shard_rows = list(shard_rows)
        rng.shuffle(shard_rows)

    yielded = 0
    for filename, species in shard_rows:
        if yielded >= num_samples:
            break

        img_path = os.path.join(_IMAGES_DIR, filename)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
            continue

        yield {
            "image":   pil_img,
            "species": species,
        }
        yielded += 1


DEEPWEEDS_CONFIG = DatasetConfig(
    name="DeepWeeds",
    load_fn=_deepweeds_stream,
    get_image=lambda s: s["image"],
    get_prompt=lambda s: "plant",
)
