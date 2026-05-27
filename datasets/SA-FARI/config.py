import requests
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


SAFARI_GCS_BASE = (
    "https://storage.googleapis.com/cxl-public-camera-trap"
    "/sa_fari/sa_fari_train/JPEGImages_6fps"
)


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


def _load_safari_json():
    from huggingface_hub import hf_hub_download
    import json
    path = hf_hub_download(
        "facebook/SA-FARI", "annotation/sa_fari_train_ext.json", repo_type="dataset"
    )
    with open(path) as f:
        return json.load(f)


def _safari_stream(num_samples: int, categories=None) -> Iterable:
    data = _load_safari_json()

    cat_class = {}
    for c in data["categories"]:
        cls = c.get("Class")
        if cls and str(cls) != "nan":
            cat_class[c["id"]] = cls

    anns_by_video: dict = {}
    for ann in data["annotations"]:
        anns_by_video.setdefault(ann["video_id"], []).append(ann)

    yielded = 0
    for video in data["videos"]:
        if yielded >= num_samples:
            break

        annotations = anns_by_video.get(video["id"], [])
        if not annotations:
            continue

        ann = annotations[0]
        tax_class = cat_class.get(ann["category_id"])
        if categories is not None and tax_class not in categories:
            continue
        iw, ih = ann["width"], ann["height"]

        frame_idx = None
        bbox_norm = None
        for fi, bb in enumerate(ann["bboxes"] or []):
            if bb is None:
                continue
            x, y, w, h = bb
            if w <= 0 or h <= 0:
                continue
            if x <= 0 or y <= 0 or x + w >= iw or y + h >= ih:
                continue
            frame_idx = fi
            bbox_norm = [x / iw, y / ih, (x + w) / iw, (y + h) / ih]
            break
        if frame_idx is None:
            continue

        frame_path = video["file_names"][frame_idx]
        url = f"{SAFARI_GCS_BASE}/{frame_path}"
        pil_image = load_image_from_url(url)
        if pil_image is None:
            continue

        yield {
            "image":      pil_image,
            "tax_class":  tax_class,
            "noun_phrase": ann["noun_phrase"],
            "bbox":       bbox_norm,
            "video_name": video["video_name"],
        }
        yielded += 1


SAFARI_CONFIG = DatasetConfig(
    name="SA-FARI",
    load_fn=_safari_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: s.get("tax_class"),
    get_bboxes=lambda s: [s["bbox"]] if s.get("bbox") else None,
    class_mapping={
        "Mammalia": "mammal",
        "Aves": "bird",
        "Reptilia": "reptile",
        "Amphibia": "amphibian",
    },
)
