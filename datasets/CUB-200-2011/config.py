import os
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
_CUB_ROOT = os.path.join(_DATA_ROOT, "CUB_200_2011")


def _ensure_data():
    if not os.path.exists(os.path.join(_CUB_ROOT, "images.txt")):
        raise RuntimeError(
            f"CUB-200-2011 data not found at {_CUB_ROOT}.\n"
            "Run the download script first:\n"
            "  sbatch download_cub200.sh\n"
            "or manually:\n"
            f"  mkdir -p {_DATA_ROOT} && cd {_DATA_ROOT} && "
            "wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz && "
            "tar -xzf CUB_200_2011.tgz && rm CUB_200_2011.tgz"
        )


def _cub200_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> Iterable:
    _ensure_data()

    if categories is not None and "Aves" not in categories:
        return

    def _read_table(filename):
        rows = {}
        with open(os.path.join(_CUB_ROOT, filename)) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                rows[parts[0]] = parts[1]
        return rows

    images      = _read_table("images.txt")
    splits      = _read_table("train_test_split.txt")
    labels      = _read_table("image_class_labels.txt")
    class_names = _read_table("classes.txt")

    bbox_rows = {}
    with open(os.path.join(_CUB_ROOT, "bounding_boxes.txt")) as f:
        for line in f:
            parts = line.strip().split()
            bbox_rows[parts[0]] = (float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4]))

    all_ids = [img_id for img_id, rel_path in images.items()
               if splits.get(img_id, "0") == "1"]

    shard_ids = all_ids[shard_id::total_shards]

    if shuffle:
        import random
        rng = random.Random(shuffle_seed)
        shard_ids = list(shard_ids)
        rng.shuffle(shard_ids)

    yielded = 0
    for img_id in shard_ids:
        if yielded >= num_samples:
            break

        rel_path = images[img_id]
        img_path = os.path.join(_CUB_ROOT, "images", rel_path)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
            continue

        raw_class = class_names.get(labels[img_id], "unknown")
        # Strip numeric prefix "001.Black_footed_Albatross" -> "Black footed Albatross"
        if "." in raw_class:
            raw_class = raw_class.split(".", 1)[1]
        species = raw_class.replace("_", " ")

        x, y, bw, bh = bbox_rows[img_id]
        iw, ih = pil_img.size
        bbox_norm = [x / iw, y / ih, (x + bw) / iw, (y + bh) / ih]

        yield {
            "image":   pil_img,
            "species": species,
            "bbox":    bbox_norm,
        }
        yielded += 1


CUB200_CONFIG = DatasetConfig(
    name="CUB-200-2011",
    load_fn=_cub200_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: "Aves",
    get_bboxes=lambda s: [s["bbox"]],
    class_mapping={"Aves": "bird"},
)
