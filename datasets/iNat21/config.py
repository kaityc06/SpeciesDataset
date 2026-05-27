import os
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


_INAT21_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
_INAT21_CACHE_DIR = os.path.join(os.path.dirname(__file__), "inat21_cache")
_INAT21_JSON_PATH = os.path.join(_INAT21_CACHE_DIR, "train.json")
_INAT21_IMAGES_DIR = os.path.join(_INAT21_CACHE_DIR, "train")


def _ensure_inat21_json():
    if os.path.exists(_INAT21_JSON_PATH):
        return
    import tarfile as _tf
    import urllib.request as _ur
    os.makedirs(_INAT21_CACHE_DIR, exist_ok=True)
    url = f"{_INAT21_BASE}/train.json.tar.gz"
    print(f"Downloading iNat21 mini annotation JSON (~46 MB) ...")
    with _ur.urlopen(url) as resp:
        with _tf.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    with open(_INAT21_JSON_PATH, "wb") as out:
                        out.write(f.read())
                    break
    print(f"Cached to {_INAT21_JSON_PATH}")


def _inat21_stream(num_samples: int, shard_id: int = 0, total_shards: int = 1, categories=None) -> Iterable:
    import json
    import tarfile
    import urllib.request

    _ensure_inat21_json()

    with open(_INAT21_JSON_PATH) as f:
        data = json.load(f)

    cat_map = {c["id"]: c for c in data["categories"]}
    ann_by_img = {a["image_id"]: cat_map[a["category_id"]] for a in data["annotations"]}

    all_images = data["images"]

    if os.path.isdir(_INAT21_IMAGES_DIR):
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
                f"Multi-shard mode requires pre-extracted images at {_INAT21_IMAGES_DIR}.\n"
                f"Run:  cd {_INAT21_CACHE_DIR} && tar -xzf train.tar.gz"
            )
        fileid_to_meta = {}
        for img in all_images:
            file_id = os.path.basename(img["file_name"]).rsplit(".", 1)[0]
            cat = ann_by_img.get(img["id"], {})
            fileid_to_meta[file_id] = {
                "supercategory": cat.get("supercategory"),
                "species":       cat.get("name"),
            }

        url = f"{_INAT21_BASE}/train.tar.gz"
        yielded = 0
        with urllib.request.urlopen(url) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    if yielded >= num_samples:
                        break
                    if not (member.isfile() and member.name.lower().endswith(".jpg")):
                        continue
                    file_id = os.path.basename(member.name).rsplit(".", 1)[0]
                    meta = fileid_to_meta.get(file_id, {})
                    if categories is not None and meta.get("supercategory") not in categories:
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


INAT21_CONFIG = DatasetConfig(
    name="iNat21",
    load_fn=_inat21_stream,
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
