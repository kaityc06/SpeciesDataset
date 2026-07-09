import json
import os
import random
import re
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from typing import Iterable

from PIL import Image, ImageFile

from species_segmentation import DatasetConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATASET_DIR, "data")
IMAGE_CACHE_DIR = os.path.join(DATA_DIR, "images")
METADATA_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip"
)
BBOX_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "snapshotserengeti-v-2-0/SnapshotSerengetiBboxes_20190903.json.zip"
)
IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshotserengeti-unzipped/"
)

DEFAULT_TOTAL_SHARDS = 400
DEFAULT_PROCESS_SAMPLES = 250
TARGET_SAMPLES = 100_000
SNAPSHOT_SAMPLE_SEED = 42

SKIP_CATEGORIES = {"empty", "human", "vehicle", "blank", "unknown"}
PROMPT_OVERRIDES = {
    "aardvark": "aardvark",
    "aardwolf": "aardwolf",
    "baboon": "baboon",
    "bat": "bat",
    "batearedfox": "bat-eared fox",
    "buffalo": "buffalo",
    "bushbuck": "bushbuck",
    "caracal": "caracal",
    "cattle": "cattle",
    "cheetah": "cheetah",
    "civet": "civet",
    "dikdik": "dik-dik",
    "duiker": "duiker",
    "eland": "eland",
    "elephant": "elephant",
    "empty": "empty",
    "fire": "fire",
    "gazellegrants": "grant gazelle",
    "gazellethomsons": "thomson gazelle",
    "genet": "genet",
    "giraffe": "giraffe",
    "guineafowl": "guineafowl",
    "hare": "hare",
    "hartebeest": "hartebeest",
    "hippopotamus": "hippopotamus",
    "honeybadger": "honey badger",
    "human": "human",
    "hyenabrown": "brown hyena",
    "hyenaspotted": "spotted hyena",
    "hyenastriped": "striped hyena",
    "impala": "impala",
    "insectspider": "insect",
    "jackal": "jackal",
    "koribustard": "kori bustard",
    "kudu": "kudu",
    "leopard": "leopard",
    "lioncub": "lion cub",
    "lionfemale": "lion",
    "lionmale": "lion",
    "mongoose": "mongoose",
    "monkeyvervet": "vervet monkey",
    "ostrich": "ostrich",
    "otherbird": "bird",
    "pangolin": "pangolin",
    "porcupine": "porcupine",
    "reedbuck": "reedbuck",
    "reptiles": "reptile",
    "rhinoceros": "rhinoceros",
    "rodents": "rodent",
    "secretarybird": "secretary bird",
    "serval": "serval",
    "steenbok": "steenbok",
    "topi": "topi",
    "vulture": "vulture",
    "warthog": "warthog",
    "waterbuck": "waterbuck",
    "wildcat": "wildcat",
    "wilddog": "wild dog",
    "wildebeest": "wildebeest",
    "zebra": "zebra",
    "zorilla": "zorilla",
}

_metadata = None
_bbox_map = None


def _download(url: str, output_path: str) -> None:
    """Download an official Snapshot Serengeti metadata artifact if missing."""
    if os.path.exists(output_path):
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".part"
    print(f"Downloading {url} to {output_path} ...")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, output_path)


def _extract_first_json(zip_path: str, output_path: str) -> str:
    """Extract the first JSON member from a LILA zip artifact."""
    if os.path.exists(output_path):
        return output_path
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = next((n for n in zf.namelist() if n.lower().endswith(".json")), None)
        if member is None:
            raise RuntimeError(f"No JSON member found in {zip_path}")
        with zf.open(member) as src, open(output_path, "wb") as dst:
            dst.write(src.read())
    return output_path


def _metadata_path() -> str:
    zip_path = os.path.join(DATA_DIR, "SnapshotSerengeti_S1-11_v2_1.json.zip")
    json_path = os.path.join(DATA_DIR, "SnapshotSerengeti_S1-11_v2_1.json")
    _download(METADATA_URL, zip_path)
    return _extract_first_json(zip_path, json_path)


def _load_metadata():
    """Load Snapshot Serengeti COCO Camera Traps metadata."""
    global _metadata
    if _metadata is not None:
        return _metadata
    with open(_metadata_path(), encoding="utf-8") as f:
        _metadata = json.load(f)
    return _metadata


def _load_bbox_map():
    """Load optional bbox annotations keyed by source image id."""
    global _bbox_map
    if _bbox_map is not None:
        return _bbox_map
    zip_path = os.path.join(DATA_DIR, "SnapshotSerengetiBboxes_20190903.json.zip")
    json_path = os.path.join(DATA_DIR, "SnapshotSerengetiBboxes_20190903.json")
    try:
        _download(BBOX_URL, zip_path)
        _extract_first_json(zip_path, json_path)
        with open(json_path, encoding="utf-8") as f:
            bbox_data = json.load(f)
    except Exception as e:
        print(f"  Snapshot Serengeti bbox metadata unavailable: {e}")
        _bbox_map = {}
        return _bbox_map

    mapping = defaultdict(list)
    for ann in bbox_data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id is not None and ann.get("bbox"):
            mapping[str(image_id)].append(ann)
    _bbox_map = mapping
    return _bbox_map


def _camel_to_words(name: str) -> str:
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.replace("_", " ").replace("-", " ").strip().lower()


def _prompt_for_category(name: str) -> str:
    """Map Snapshot category labels to stable SAM text prompts."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return PROMPT_OVERRIDES.get(key, _camel_to_words(name))


def _image_url(file_name: str) -> str:
    return urllib.parse.urljoin(IMAGE_BASE_URL, file_name.replace("\\", "/"))


def _cached_image_path(file_name: str) -> str:
    rel = file_name.replace("\\", "/").lstrip("/")
    path = os.path.join(IMAGE_CACHE_DIR, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _download_image(file_name: str) -> str:
    """Cache Snapshot Serengeti images lazily by original relative path."""
    path = _cached_image_path(file_name)
    if os.path.exists(path):
        return path
    tmp_path = path + ".part"
    url = _image_url(file_name)
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)
    return path


def _normalize_bbox(bbox, image_size):
    """Convert COCO [x, y, w, h] boxes to normalized [x0, y0, x1, y1]."""
    if not bbox:
        return None
    iw, ih = image_size
    x, y, w, h = [float(v) for v in bbox]
    return [x / iw, y / ih, (x + w) / iw, (y + h) / ih]


def _source_metadata(image_info, annotations, category_record, bbox_annotations):
    """Keep the original LILA records that produced this processing sample."""
    return {
        "image": image_info,
        "annotations": annotations,
        "category": category_record,
        "bbox_annotations": bbox_annotations,
    }


def _build_candidate_rows(data, bbox_map, categories=None):
    """Build the fixed bbox-first random sample order before downloading images."""
    category_by_id = {str(c["id"]): c["name"] for c in data.get("categories", [])}
    category_record_by_id = {str(c["id"]): c for c in data.get("categories", [])}
    images_by_id = {str(img["id"]): img for img in data.get("images", [])}

    annotations_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id"))
        category_name = category_by_id.get(str(ann.get("category_id")), "")
        if category_name.lower() in SKIP_CATEGORIES:
            continue
        annotations_by_image[image_id].append(ann)

    with_bbox = []
    without_bbox = []
    for image_id in sorted(annotations_by_image):
        anns = annotations_by_image[image_id]
        image_info = images_by_id.get(image_id)
        if not image_info:
            continue
        file_name = image_info.get("file_name") or image_info.get("file")
        if not file_name:
            continue

        ann = anns[0]
        category_name = category_by_id.get(str(ann.get("category_id")), "")
        prompt = _prompt_for_category(category_name)
        if categories is not None and prompt not in categories:
            continue

        bbox_annotations = bbox_map.get(image_id, [])
        bbox = ann.get("bbox")
        if not bbox and bbox_annotations:
            bbox = bbox_annotations[0].get("bbox")

        row = {
            "image_id": image_id,
            "image_info": image_info,
            "annotations": anns,
            "annotation": ann,
            "category_name": category_name,
            "category_record": category_record_by_id.get(str(ann.get("category_id")), {}),
            "prompt": prompt,
            "file_name": file_name,
            "bbox": bbox,
            "bbox_annotations": bbox_annotations,
        }
        (with_bbox if bbox else without_bbox).append(row)

    rng = random.Random(SNAPSHOT_SAMPLE_SEED)
    rng.shuffle(with_bbox)
    rng.shuffle(without_bbox)
    return (with_bbox + without_bbox)[:TARGET_SAMPLES]


def _snapshot_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
) -> Iterable:
    data = _load_metadata()
    bbox_map = _load_bbox_map()
    candidate_rows = _build_candidate_rows(data, bbox_map, categories=categories)
    yielded = 0

    for row_index, row in enumerate(candidate_rows):
        if total_shards > 1 and row_index % total_shards != shard_id:
            continue
        if yielded >= num_samples:
            break

        file_name = row["file_name"]
        try:
            image_path = _download_image(file_name)
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping Snapshot Serengeti image {file_name}: {e}")
            continue

        yield {
            "image": image,
            "image_id": row["image_id"],
            "file_name": file_name,
            "species": row["category_name"],
            "common_name": row["category_name"],
            "label": row["category_name"],
            "category": row["category_name"],
            "prompt_class": row["prompt"],
            "bbox": _normalize_bbox(row["bbox"], image.size),
            "image_metadata": row["image_info"],
            "annotation_metadata": row["annotation"],
            "all_annotations_metadata": row["annotations"],
            "category_metadata": row["category_record"],
            "bbox_annotation_metadata": row["bbox_annotations"][0] if row["bbox_annotations"] else None,
            "all_bbox_annotations_metadata": row["bbox_annotations"],
            "original_metadata": _source_metadata(
                row["image_info"],
                row["annotations"],
                row["category_record"],
                row["bbox_annotations"],
            ),
        }
        yielded += 1


SNAPSHOT_SERENGETI_CONFIG = DatasetConfig(
    name="SnapshotSerengeti",
    load_fn=_snapshot_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: s.get("prompt_class"),
    get_bboxes=lambda s: [s["bbox"]] if s.get("bbox") else None,
    class_mapping={},
    get_prompt=lambda s: s.get("prompt_class"),
)

CONFIG = SNAPSHOT_SERENGETI_CONFIG
