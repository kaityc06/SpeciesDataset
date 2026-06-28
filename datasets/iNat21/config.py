import os
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


_INAT21_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
_INAT21_CACHE_DIR = os.path.join(os.path.dirname(__file__), "inat21_cache")

_VALID_SPLITS = {"train", "train_mini", "val"}


def _split_paths(split: str):
    if split not in _VALID_SPLITS:
        raise ValueError(f"SPLIT must be one of {_VALID_SPLITS}, got {split!r}")
    json_path = os.path.join(_INAT21_CACHE_DIR, f"{split}.json")
    images_dir = os.path.join(_INAT21_CACHE_DIR, split)
    return json_path, images_dir


def _ensure_inat21_json(split: str):
    json_path, _ = _split_paths(split)
    if os.path.exists(json_path):
        return
    import tarfile as _tf
    import urllib.request as _ur
    os.makedirs(_INAT21_CACHE_DIR, exist_ok=True)
    url = f"{_INAT21_BASE}/{split}.json.tar.gz"
    print(f"Downloading iNat21 {split} annotation JSON ...")
    with _ur.urlopen(url) as resp:
        with _tf.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    with open(json_path, "wb") as out:
                        out.write(f.read())
                    break
    print(f"Cached to {json_path}")


def _make_inat21_stream(split: str):
    def _inat21_stream(num_samples: int, shard_id: int = 0, total_shards: int = 1, categories=None, shuffle: bool = False, shuffle_seed: int = 42) -> Iterable:
        import json
        import random
        import tarfile
        import urllib.request

        _ensure_inat21_json(split)
        json_path, images_dir = _split_paths(split)

        with open(json_path) as f:
            data = json.load(f)

        cat_map = {c["id"]: c for c in data["categories"]}
        ann_by_img = {a["image_id"]: cat_map[a["category_id"]] for a in data["annotations"]}

        all_images = data["images"]
        if shuffle:
            rng = random.Random(shuffle_seed)
            all_images = list(all_images)
            rng.shuffle(all_images)

        if os.path.isdir(images_dir):
            shard_imgs = all_images[shard_id::total_shards]
            shard_imgs = shard_imgs[:num_samples]
            for img_meta in shard_imgs:
                local_path = os.path.join(_INAT21_CACHE_DIR, img_meta["file_name"])
                cat = ann_by_img.get(img_meta["id"], {})
                if categories is not None and cat.get("supercategory") not in categories:
                    continue
                try:
                    pil_img = Image.open(local_path).convert("RGB")
                except Exception as exc:
                    print(f"  Skipping {local_path}: {exc}")
                    continue
                yield {
                    "image":         pil_img,
                    "supercategory": cat.get("supercategory"),
                    "species":       cat.get("name"),
                    "file_id":       os.path.basename(img_meta["file_name"]).rsplit(".", 1)[0],
                }
        else:
            if total_shards > 1:
                raise RuntimeError(
                    f"Multi-shard mode requires pre-extracted images at {images_dir}.\n"
                    f"Run:  cd {_INAT21_CACHE_DIR} && tar -xzf {split}.tar.gz"
                )
            fileid_to_meta = {}
            for img in all_images:
                file_id = os.path.basename(img["file_name"]).rsplit(".", 1)[0]
                cat = ann_by_img.get(img["id"], {})
                fileid_to_meta[file_id] = {
                    "supercategory": cat.get("supercategory"),
                    "species":       cat.get("name"),
                }

            # When shuffle=True, all_images is already shuffled; pre-select the
            # target file IDs so the tar stream is filtered rather than taken in order.
            if shuffle:
                candidate_imgs = all_images
                if categories is not None:
                    candidate_imgs = [
                        img for img in candidate_imgs
                        if fileid_to_meta.get(
                            os.path.basename(img["file_name"]).rsplit(".", 1)[0], {}
                        ).get("supercategory") in categories
                    ]
                target_ids = {
                    os.path.basename(img["file_name"]).rsplit(".", 1)[0]
                    for img in candidate_imgs[:num_samples]
                }
            else:
                target_ids = None

            url = f"{_INAT21_BASE}/{split}.tar.gz"
            yielded = 0
            with urllib.request.urlopen(url) as resp:
                with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                    for member in tar:
                        if yielded >= num_samples:
                            break
                        if not (member.isfile() and member.name.lower().endswith(".jpg")):
                            continue
                        file_id = os.path.basename(member.name).rsplit(".", 1)[0]
                        if target_ids is not None and file_id not in target_ids:
                            continue
                        meta = fileid_to_meta.get(file_id, {})
                        if target_ids is None and categories is not None and meta.get("supercategory") not in categories:
                            continue
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        try:
                            pil_img = Image.open(BytesIO(f.read())).convert("RGB")
                        except Exception:
                            continue
                        yield {
                            "image":         pil_img,
                            "supercategory": meta.get("supercategory"),
                            "species":       meta.get("species"),
                            "file_id":       file_id,
                        }
                        yielded += 1

    return _inat21_stream


_inat21_split = os.environ.get("SPLIT", "train")

INAT21_CONFIG = DatasetConfig(
    name=f"iNat21",
    load_fn=_make_inat21_stream(_inat21_split),
    get_image=lambda s: s["image"].convert("RGB") if s.get("image") else None,
    get_class=lambda s: s.get("supercategory"),
    class_mapping={
        "Animalia": ["animal"],
        "Amphibians": ["frog", "salamander"],
        "Arachnids": ["spider"],
        "Birds": ["bird"],
        "Fungi": ["fungus"],
        "Insects": ["insect"],
        "Mammals": ["mammal"],
        "Mollusks": ["shellfish", "snail", "slug", "octopus"],
        "Plants": ["plant"],
        "Ray-finned Fishes": ["fish"],
        "Reptiles": ["reptile", "turtle", "lizard", "crocodile"],
    },
)
