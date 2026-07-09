import csv
import json
import os
import urllib.request
import zipfile
from typing import Iterable, List, Optional

from PIL import Image, ImageFile

from species_segmentation import DatasetConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATASET_DIR, "data")
FISHNET_REPO_ZIP_URL = "https://github.com/faixan-khan/FishNet/archive/refs/heads/main.zip"
FISHNET_IMAGE_FILE_ID = "1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3"
FISHNET_IMAGE_DRIVE_URL = (
    "https://drive.google.com/file/d/"
    f"{FISHNET_IMAGE_FILE_ID}/view?usp=sharing"
)

FISHNET_REPO_DIR = os.path.join(DATA_DIR, "FishNet-main")
FISHNET_ANNS_DIR = os.path.join(FISHNET_REPO_DIR, "anns")
FISHNET_BBOX_ZIP = os.path.join(FISHNET_REPO_DIR, "bbox.zip")
FISHNET_BBOX_DIR = os.path.join(FISHNET_REPO_DIR, "bbox")

DEFAULT_TOTAL_SHARDS = 379
DEFAULT_PROCESS_SAMPLES = 250
TARGET_SAMPLES = 94_532

FISHNET_CLASS_MAPPING = {"Fish": "fish"}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
TRAIT_COLUMNS = (
    "Troph",
    "FeedingPath",
    "Tropical",
    "Temperate",
    "Subtropical",
    "Boreal",
    "Polar",
    "freshwater",
    "saltwater",
    "brackish",
)
TAXONOMY_COLUMNS = (
    "species",
    "SpecCode",
    "Genus",
    "Subfamily",
    "Family",
    "Order",
    "Class",
    "SuperClass",
    "NewOrder",
    "source",
    "Folder",
)
TAXONOMY_OUTPUT_KEYS = {
    "species": "species",
    "SpecCode": "spec_code",
    "Genus": "genus",
    "Subfamily": "subfamily",
    "Family": "family",
    "Order": "order",
    "Class": "class_name",
    "SuperClass": "super_class",
    "NewOrder": "new_order",
    "source": "source",
    "Folder": "folder",
}

_rows_cache = None
_image_index = None
_bbox_index = None


def _download(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".part"
    print(f"Downloading {url} to {path} ...")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)


def _safe_extract(zip_path: str, output_dir: str) -> None:
    """Extract official archives without allowing zip path traversal."""
    base = os.path.abspath(output_dir)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(output_dir, member))
            if not target.startswith(base + os.sep) and target != base:
                raise RuntimeError(f"Unsafe zip member path: {member}")
        zf.extractall(output_dir)


def _ensure_annotations() -> None:
    """Ensure the official FishNet repo metadata and bbox archive are local."""
    train_path = os.path.join(FISHNET_ANNS_DIR, "train.csv")
    test_path = os.path.join(FISHNET_ANNS_DIR, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(FISHNET_BBOX_ZIP):
        return

    repo_zip = os.path.join(DATA_DIR, "FishNet-main.zip")
    _download(FISHNET_REPO_ZIP_URL, repo_zip)
    _safe_extract(repo_zip, DATA_DIR)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise RuntimeError(f"Could not find FishNet train/test CSVs under {FISHNET_ANNS_DIR}")
    if not os.path.exists(FISHNET_BBOX_ZIP):
        raise RuntimeError(f"Could not find FishNet bbox.zip under {FISHNET_REPO_DIR}")


def _image_roots() -> List[str]:
    candidates = [
        os.path.join(DATA_DIR, "Image_Library"),
        os.path.join(DATA_DIR, "Images"),
        os.path.join(DATA_DIR, "images"),
        os.path.join(FISHNET_REPO_DIR, "Images"),
    ]
    return sorted({path for path in candidates if os.path.isdir(path)})


def _ensure_images() -> None:
    if _image_roots():
        return
    image_zip = os.path.join(DATA_DIR, "fishnet_images.zip")
    if not os.path.exists(image_zip):
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "FishNet images are distributed from Google Drive. Install gdown "
                f"or place the downloaded zip from {FISHNET_IMAGE_DRIVE_URL} at "
                f"{image_zip}."
            ) from exc
        print(f"Downloading FishNet images to {image_zip} ...")
        gdown.download(id=FISHNET_IMAGE_FILE_ID, output=image_zip, quiet=False)
    _safe_extract(image_zip, DATA_DIR)
    if not _image_roots():
        raise RuntimeError(f"Could not find FishNet image directory under {DATA_DIR}")


def _read_split(split: str) -> List[dict]:
    """Read one official FishNet annotation CSV and keep its split provenance."""
    path = os.path.join(FISHNET_ANNS_DIR, f"{split}.csv")
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["split"] = split
            rows.append(row)
    return rows


def _official_rows() -> List[dict]:
    global _rows_cache
    if _rows_cache is not None:
        return _rows_cache
    _ensure_annotations()
    _rows_cache = _read_split("train") + _read_split("test")
    return _rows_cache


def _all_image_paths() -> List[str]:
    _ensure_images()
    paths = []
    for root in _image_roots():
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.abspath(os.path.join(dirpath, filename)))
    return sorted(set(paths))


def _build_image_index():
    """Build relative-path and basename indexes for FishNet image resolution."""
    global _image_index
    if _image_index is not None:
        return _image_index

    roots = _image_roots()
    index = {}
    basename_index = {}
    for path in _all_image_paths():
        for root in roots:
            try:
                rel = os.path.relpath(path, root)
            except ValueError:
                continue
            if not rel.startswith(".."):
                index[rel.replace(os.sep, "/")] = path
        basename_index.setdefault(os.path.basename(path), []).append(path)
    _image_index = {"rel": index, "basename": basename_index}
    return _image_index


def _image_basename(row: dict) -> str:
    return os.path.basename(str(row.get("image") or "").split("?", 1)[0])


def _resolve_image_path(row: dict, image_index=None) -> Optional[str]:
    image_index = image_index or _build_image_index()
    folder = str(row.get("Folder") or "").strip().strip("/")
    basename = _image_basename(row)
    if not basename:
        return None

    if folder:
        rel = f"{folder}/{basename}"
        path = image_index["rel"].get(rel)
        if path:
            return path
        for root in _image_roots():
            candidate = os.path.join(root, folder, basename)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    matches = image_index["basename"].get(basename) or []
    if len(matches) == 1:
        return matches[0]
    return None


def _ensure_bbox_annotations() -> str:
    _ensure_annotations()
    marker = os.path.join(FISHNET_BBOX_DIR, "all_family")
    if not os.path.isdir(marker):
        _safe_extract(FISHNET_BBOX_ZIP, FISHNET_BBOX_DIR)
    if not os.path.isdir(marker):
        raise RuntimeError(f"Could not find all_family after extracting {FISHNET_BBOX_ZIP}")
    return marker


def _load_bbox_index():
    """Parse all official FishNet bbox text files into an image-keyed index."""
    global _bbox_index
    if _bbox_index is not None:
        return _bbox_index

    bbox_root = _ensure_bbox_annotations()
    index = {}
    for dirpath, _, filenames in os.walk(bbox_root):
        for filename in filenames:
            if not filename.lower().endswith(".txt"):
                continue
            path = os.path.join(dirpath, filename)
            folder = os.path.basename(os.path.dirname(path))
            stem = os.path.splitext(filename)[0]
            boxes = []
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        x0, y0, x1, y1 = [float(v) for v in parts[-4:]]
                    except ValueError:
                        continue
                    boxes.append([x0, y0, x1, y1])
            if boxes:
                index[(folder, stem)] = boxes
                index.setdefault(("", stem), boxes)
    _bbox_index = index
    return _bbox_index


def _raw_bboxes_for_row(row: dict, bbox_index=None):
    bbox_index = bbox_index or _load_bbox_index()
    folder = str(row.get("Folder") or "").strip()
    stem = os.path.splitext(_image_basename(row))[0]
    return bbox_index.get((folder, stem)) or bbox_index.get(("", stem)) or []


def _normalise_bboxes(raw_bboxes, image_size):
    iw, ih = image_size
    bboxes = []
    for x0, y0, x1, y1 in raw_bboxes:
        if max(x0, y0, x1, y1) > 1.0:
            x0, x1 = x0 / iw, x1 / iw
            y0, y1 = y0 / ih, y1 / ih
        bboxes.append([
            max(0.0, min(1.0, x0)),
            max(0.0, min(1.0, y0)),
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
        ])
    return bboxes


def _clean_value(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value and value.lower() != "nan" else None


def _row_id(row: dict, fallback_index: int) -> str:
    return _clean_value(row.get("Unnamed: 0")) or _clean_value(row.get("")) or str(fallback_index)


def _trait_data(row: dict):
    """Return the official FishNet trait columns as a compact metadata dict."""
    traits = {}
    for key in TRAIT_COLUMNS:
        value = _clean_value(row.get(key))
        if value is not None:
            traits[key] = value
    return traits


def _original_row_metadata(row: dict):
    """Preserve every non-empty column from the official FishNet CSV row."""
    return {
        key: value
        for key, value in row.items()
        if _clean_value(value) is not None
    }


def _sample_from_row(row: dict, path: str, row_index: int, image=None, bboxes=None) -> dict:
    """Convert one official FishNet row into the shared SpeciesDataset sample shape."""
    sample = {
        "image_id": f"FishNet_{_row_id(row, row_index)}",
        "file_name": os.path.relpath(path, DATA_DIR),
        "split": _clean_value(row.get("split")) or "",
        "original_metadata": _original_row_metadata(row),
        "trait_data": _trait_data(row),
        "bboxes": bboxes or [],
        "bbox": bboxes[0] if bboxes else [],
        "bbox_count": len(bboxes or []),
    }
    if image is not None:
        sample["image"] = image

    for key in TAXONOMY_COLUMNS:
        output_key = TAXONOMY_OUTPUT_KEYS.get(key, key)
        sample[output_key] = _clean_value(row.get(key)) or ""

    scientific_name = _clean_value(row.get("species"))
    sample["scientific_name"] = scientific_name or ""
    return sample


def _fishnet_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
) -> Iterable:
    if categories is not None and "Fish" not in categories and "fish" not in categories:
        return

    image_index = _build_image_index()
    bbox_index = _load_bbox_index()
    yielded = 0

    for row_index, row in enumerate(_official_rows()):
        if total_shards > 1 and row_index % total_shards != shard_id:
            continue
        if yielded >= num_samples:
            break

        path = _resolve_image_path(row, image_index)
        if not path:
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

        raw_bboxes = _raw_bboxes_for_row(row, bbox_index)
        bboxes = _normalise_bboxes(raw_bboxes, image.size)
        yield _sample_from_row(row, path, row_index, image=image, bboxes=bboxes)
        yielded += 1


FISHNET_CONFIG = DatasetConfig(
    name="FishNet",
    load_fn=_fishnet_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: "Fish",
    get_bboxes=lambda s: s.get("bboxes"),
    class_mapping=FISHNET_CLASS_MAPPING,
)

CONFIG = FISHNET_CONFIG
