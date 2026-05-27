import os
import requests
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


_ENA24_JSON_URL    = "https://lilawildlife.blob.core.windows.net/lila-wildlife/ena24/ena24.json"
_ENA24_IMAGES_BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/ena24/images"
_ENA24_CACHE_DIR   = "/mmfs1/gscratch/krishna/kaityc/ena24_cache"
_ENA24_JSON_PATH   = os.path.join(_ENA24_CACHE_DIR, "ena24.json")
_ENA24_IMAGES_DIR  = os.path.join(_ENA24_CACHE_DIR, "images")


def load_image_from_url(url: str, timeout: int = 10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError) as e:
        print(f"  Skipping {url}: {e}")
        return None


def _ensure_ena24_json():
    if os.path.exists(_ENA24_JSON_PATH):
        return
    import urllib.request
    os.makedirs(_ENA24_CACHE_DIR, exist_ok=True)
    print(f"Downloading ENA24 metadata (~3.6 MB) ...")
    urllib.request.urlretrieve(_ENA24_JSON_URL, _ENA24_JSON_PATH)
    print(f"Cached to {_ENA24_JSON_PATH}")


def _ena24_stream(num_samples: int, shard_id: int = 0, total_shards: int = 1) -> Iterable:
    import json

    _ensure_ena24_json()

    with open(_ENA24_JSON_PATH) as f:
        data = json.load(f)

    _IGNORE = {"Human", "Vehicle"}
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    ignored_ids = {cid for cid, name in cat_map.items() if name in _IGNORE}

    anns_by_image: dict = {}
    for ann in data["annotations"]:
        if ann["category_id"] not in ignored_ids:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    all_images = data["images"]
    shard_imgs = all_images[shard_id::total_shards][:num_samples]

    for img_meta in shard_imgs:
        annotations = anns_by_image.get(img_meta["id"], [])
        if not annotations:
            continue

        iw = img_meta.get("width", 1)
        ih = img_meta.get("height", 1)

        ann = annotations[0]
        category = cat_map.get(ann["category_id"], "animal")
        x, y, w, h = ann["bbox"]
        bbox_norm = [x / iw, y / ih, (x + w) / iw, (y + h) / ih]
        bbox_norm = [max(0.0, min(1.0, v)) for v in bbox_norm]

        file_name = img_meta["file_name"]
        if os.path.isdir(_ENA24_IMAGES_DIR):
            local_path = os.path.join(_ENA24_IMAGES_DIR, file_name)
            try:
                pil_img = Image.open(local_path).convert("RGB")
            except Exception as e:
                print(f"  Skipping {local_path}: {e}")
                continue
        else:
            pil_img = load_image_from_url(f"{_ENA24_IMAGES_BASE}/{file_name}")
            if pil_img is None:
                continue

        yield {
            "image":    pil_img,
            "category": category,
            "bbox":     bbox_norm,
            "file_name": file_name,
        }


ENA24_CONFIG = DatasetConfig(
    name="ENA24",
    load_fn=_ena24_stream,
    get_image=lambda s: s["image"],
    get_prompt=lambda s: s["category"].replace("_", " ").replace("-", " ").lower(),
    get_bboxes=lambda s: [s["bbox"]],
)
