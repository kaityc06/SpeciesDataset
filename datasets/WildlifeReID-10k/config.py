import os
import ast
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def _ensure_data():
    if not os.path.exists(os.path.join(_DATA_ROOT, "metadata.csv")):
        raise RuntimeError(
            f"WildlifeReID-10k data not found at {_DATA_ROOT}.\n"
            "Run the download script first:\n"
            "  sbatch download_wildlifereid.sh\n"
            "or manually:\n"
            f"  mkdir -p {_DATA_ROOT} && cd {_DATA_ROOT} && "
            "kaggle datasets download -d wildlifedatasets/wildlifereid-10k && "
            "unzip wildlifereid-10k.zip"
        )


_VALID_SPLITS = {"train", "test"}


def _make_wildlifereid_stream(split: str = None):
    if split is not None and split not in _VALID_SPLITS:
        raise ValueError(f"SPLIT must be one of {_VALID_SPLITS}, got {split!r}")

    def _wildlifereid_stream(
        num_samples: int,
        shard_id: int = 0,
        total_shards: int = 1,
        categories=None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ) -> Iterable:
        from wildlife_datasets.datasets import WildlifeReID10k

        _ensure_data()
        dataset = WildlifeReID10k(root=_DATA_ROOT)
        df = dataset.df

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        has_species = "species" in df.columns
        has_bbox = "bbox" in df.columns

        shard_df = df.iloc[shard_id::total_shards].reset_index(drop=True)
        if shuffle:
            shard_df = shard_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

        yielded = 0
        for _, row in shard_df.iterrows():
            if yielded >= num_samples:
                break
            species = row["species"] if has_species else None
            if categories is not None and species not in categories:
                continue
            img_path = os.path.join(_DATA_ROOT, row["path"])
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"  Skipping {img_path}: {exc}")
                continue
            bbox = None
            if has_bbox and row["bbox"] is not None:
                try:
                    coords = row["bbox"]
                    if isinstance(coords, str):
                        coords = ast.literal_eval(coords)
                    x, y, w, h = coords
                    iw, ih = pil_img.size
                    if w > 0 and h > 0:
                        bbox = [x / iw, y / ih, (x + w) / iw, (y + h) / ih]
                except Exception:
                    pass
            yield {
                "image":    pil_img,
                "species":  species,
                "identity": row["identity"],
                "bbox":     bbox,
            }
            yielded += 1

    return _wildlifereid_stream


_wildlifereid_split = os.environ.get("SPLIT", "train")

WILDLIFEREID_CONFIG = DatasetConfig(
    name="WildlifeReID-10k",
    load_fn=_make_wildlifereid_stream(_wildlifereid_split),
    get_image=lambda s: s["image"],
    get_class=lambda s: s.get("species"),
    get_bboxes=lambda s: [s["bbox"]] if s.get("bbox") else None,
    class_mapping={
        # Birds
        "bird":             ["bird"],
        "chicken":          ["chicken", "bird"],
        # Mammals — big cats
        "leopard":          ["leopard"],
        "tiger":            ["tiger"],
        "cat":              ["cat"],
        # Mammals — primates
        "chimpanzee":       ["chimpanzee", "primate"],
        "macaque":          ["monkey"],
        # Mammals — bears / pandas
        "polar bear":       ["bear"],
        "panda":            ["panda"],
        # Mammals — cetaceans
        "whale":            ["whale"],
        "dolphin":          ["dolphin"],
        # Mammals — pinnipeds
        "seal":             ["seal"],
        # Mammals — ungulates / herbivores
        "giraffe":          ["giraffe"],
        "zebra":            ["zebra"],
        "nyala":            ["antelope", "mammal"],
        "cow":              ["cow"],
        # Mammals — canids
        "dog":              ["dog"],
        # Mammals — hyena
        "hyena":            ["hyena", "mammal"],
        # Reptiles
        "sea turtle":       ["turtle"],
        # Fish / sharks
        "whaleshark":       ["whale shark", "shark", "fish"],
        "fish":             ["fish"],
        # Echinoderms
        "sea star":         ["sea star", "starfish"],
    },
)
