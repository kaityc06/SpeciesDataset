import os
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


# ── Download via wildlife_datasets (Kaggle) ───────────────────────────────────
#
#   pip install wildlife-datasets kaggle
#
# Prerequisites:
#   1. Place Kaggle API credentials at ~/.kaggle/kaggle.json
#
# On first run, config calls AnimalCLEF2026.get_data(root) which invokes the
# Kaggle CLI to download and extract automatically.
#
# Species: loggerhead turtle, salamander, lynx, lizards (~15 k images, 1 450 individuals).
# Task: individual re-identification (not species classification).
#
# dataset.df columns (verify against actual): path, identity, species, ...
# Images live at: os.path.join(dataset.root, row["path"])

_ANIMALCLEF2026_DIR = "/mmfs1/gscratch/krishna/kaityc/lifeclef2026_cache/animalclef2026"


def _animalclef2026_stream(
    num_samples: int, shard_id: int = 0, total_shards: int = 1, categories=None
) -> Iterable:
    from wildlife_datasets.datasets import AnimalCLEF2026

    os.makedirs(_ANIMALCLEF2026_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(_ANIMALCLEF2026_DIR, "already_downloaded")):
        AnimalCLEF2026.get_data(_ANIMALCLEF2026_DIR)
    dataset = AnimalCLEF2026(_ANIMALCLEF2026_DIR)
    rows = dataset.df.iloc[shard_id::total_shards]
    if categories is not None:
        rows = rows[rows["species"].isin(categories)]
    rows = rows.head(num_samples)

    for _, row in rows.iterrows():
        img_path = os.path.join(dataset.root, row["path"])
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
            continue
        yield {
            "image":         pil_img,
            "individual_id": row.get("identity"),
            "species":       row.get("species"),
            "dataset":       row.get("dataset"),
        }


ANIMALCLEF2026_CONFIG = DatasetConfig(
    name="AnimalCLEF2026",
    load_fn=_animalclef2026_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: s.get("species"),
    class_mapping={
        "loggerhead turtle": ["turtle"],
        "lynx":              ["feline", "cat"],
        "salamander":        ["salamander"],
        "lizard":            ["lizard"],
    },
)
