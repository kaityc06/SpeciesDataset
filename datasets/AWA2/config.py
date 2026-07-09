import os
import shutil
import urllib.request
import zipfile
from functools import lru_cache
from typing import Iterable

from PIL import Image, ImageFile

from species_segmentation import DatasetConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATASET_DIR, "data")
AWA2_DATA_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
AWA2_BASE_URL = "https://cvml.ista.ac.at/AwA2/AwA2-base.zip"

DEFAULT_TOTAL_SHARDS = 150
DEFAULT_PROCESS_SAMPLES = 250
TARGET_SAMPLES = 37_322

AWA2_CLASS_MAPPING = {
    "antelope": "antelope",
    "grizzly+bear": "grizzly bear",
    "killer+whale": "killer whale",
    "beaver": "beaver",
    "dalmatian": "dalmatian",
    "persian+cat": "persian cat",
    "horse": "horse",
    "german+shepherd": "german shepherd",
    "blue+whale": "blue whale",
    "siamese+cat": "siamese cat",
    "skunk": "skunk",
    "mole": "mole",
    "tiger": "tiger",
    "hippopotamus": "hippopotamus",
    "leopard": "leopard",
    "moose": "moose",
    "spider+monkey": "spider monkey",
    "humpback+whale": "humpback whale",
    "elephant": "elephant",
    "gorilla": "gorilla",
    "ox": "ox",
    "fox": "fox",
    "sheep": "sheep",
    "seal": "seal",
    "chimpanzee": "chimpanzee",
    "hamster": "hamster",
    "squirrel": "squirrel",
    "rhinoceros": "rhinoceros",
    "rabbit": "rabbit",
    "bat": "bat",
    "giraffe": "giraffe",
    "wolf": "wolf",
    "chihuahua": "chihuahua",
    "rat": "rat",
    "weasel": "weasel",
    "otter": "otter",
    "buffalo": "buffalo",
    "zebra": "zebra",
    "giant+panda": "giant panda",
    "deer": "deer",
    "bobcat": "bobcat",
    "pig": "pig",
    "lion": "lion",
    "mouse": "mouse",
    "polar+bear": "polar bear",
    "collie": "collie",
    "walrus": "walrus",
    "raccoon": "raccoon",
    "cow": "cow",
    "dolphin": "dolphin",
}


def _read_indexed_table(path: str):
    """Read AwA2 files whose rows are '<index> <value>'."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                rows[int(parts[0])] = parts[1]
    return rows


def _read_class_list(path: str):
    """Read AwA2 train/test class lists while tolerating absent base metadata."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _read_attribute_matrix(path: str, class_names, attribute_names, cast):
    """Map each class to its original AwA2 attribute vector."""
    if not os.path.exists(path):
        return {}
    matrix = {}
    with open(path, encoding="utf-8") as f:
        for class_name, line in zip(class_names, f):
            values = line.strip().split()
            matrix[class_name] = {
                attr: cast(value)
                for attr, value in zip(attribute_names, values)
            }
    return matrix


@lru_cache(maxsize=1)
def _awa2_metadata(root: str):
    """Load all class-level AwA2 metadata distributed outside JPEGImages."""
    class_rows = _read_indexed_table(os.path.join(root, "classes.txt"))
    predicate_rows = _read_indexed_table(os.path.join(root, "predicates.txt"))
    class_names = [class_rows[i] for i in sorted(class_rows)]
    attribute_names = [predicate_rows[i] for i in sorted(predicate_rows)]
    train_classes = _read_class_list(os.path.join(root, "trainclasses.txt"))
    test_classes = _read_class_list(os.path.join(root, "testclasses.txt"))
    return {
        "class_index": {name: idx for idx, name in class_rows.items()},
        "attribute_names": attribute_names,
        "attribute_binary": _read_attribute_matrix(
            os.path.join(root, "predicate-matrix-binary.txt"),
            class_names,
            attribute_names,
            int,
        ),
        "attribute_continuous": _read_attribute_matrix(
            os.path.join(root, "predicate-matrix-continuous.txt"),
            class_names,
            attribute_names,
            float,
        ),
        "train_classes": train_classes,
        "test_classes": test_classes,
    }


def _class_split(class_name: str, metadata: dict) -> str | None:
    """Return the official zero-shot split label when AwA2 base metadata exists."""
    if class_name in metadata["train_classes"]:
        return "train"
    if class_name in metadata["test_classes"]:
        return "test"
    return None


def _license_metadata(root: str, rel_path: str):
    """Preserve the per-image Flickr license file distributed with AwA2-base."""
    parts = rel_path.split(os.sep)
    class_name = parts[1] if parts and parts[0] == "JPEGImages" and len(parts) > 1 else parts[0]
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    license_rel = os.path.join("licenses", class_name, f"{stem}.txt")
    license_path = os.path.join(root, license_rel)
    if not os.path.exists(license_path):
        return None, None
    with open(license_path, encoding="utf-8", errors="replace") as f:
        return license_rel, f.read().strip()


def _expected_size(response) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError:
            return None
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            return int(content_length)
        except ValueError:
            return None
    return None


def _download(url: str, path: str, max_attempts: int = 5) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".part"
    for attempt in range(1, max_attempts + 1):
        resume_at = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if resume_at else "wb"
        print(f"Downloading {url} to {path} (attempt {attempt}/{max_attempts}) ...")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                if resume_at and getattr(response, "status", None) != 206:
                    mode = "wb"
                    resume_at = 0
                    print("  Server did not resume partial download; restarting.")
                expected_size = _expected_size(response)
                with open(tmp_path, mode) as f:
                    shutil.copyfileobj(response, f, length=1024 * 1024)
            if expected_size is not None and os.path.getsize(tmp_path) < expected_size:
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got {os.path.getsize(tmp_path)} out of {expected_size} bytes",
                    None,
                )
            os.replace(tmp_path, path)
            return
        except Exception as e:
            if attempt == max_attempts:
                raise
            print(f"  Download interrupted: {e}. Retrying with partial file if possible.")


def _extract(zip_path: str, marker_dir: str) -> None:
    if os.path.isdir(marker_dir):
        return
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)


def _find_awa_root() -> str:
    candidates = [
        os.path.join(DATA_DIR, "Animals_with_Attributes2"),
        os.path.join(DATA_DIR, "AWA2"),
        DATA_DIR,
    ]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "JPEGImages")):
            return root
    for dirpath, dirnames, _ in os.walk(DATA_DIR):
        if "JPEGImages" in dirnames:
            return dirpath
    raise RuntimeError(
        "Could not find AwA2 JPEGImages. Expected AwA2-data.zip extracted under "
        f"{DATA_DIR}."
    )


def _ensure_awa2():
    try:
        _find_awa_root()
        return
    except RuntimeError:
        pass
    data_zip = os.path.join(DATA_DIR, "AwA2-data.zip")
    base_zip = os.path.join(DATA_DIR, "AwA2-base.zip")
    _download(AWA2_DATA_URL, data_zip)
    _download(AWA2_BASE_URL, base_zip)
    _extract(data_zip, os.path.join(DATA_DIR, "Animals_with_Attributes2", "JPEGImages"))
    _extract(base_zip, os.path.join(DATA_DIR, "Animals_with_Attributes2"))


def _image_paths(root: str):
    jpeg_root = os.path.join(root, "JPEGImages")
    paths = []
    for dirpath, _, filenames in os.walk(jpeg_root):
        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def _awa2_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
) -> Iterable:
    _ensure_awa2()
    root = _find_awa_root()
    metadata = _awa2_metadata(root)
    yielded = 0
    for image_index, path in enumerate(_image_paths(root)):
        if total_shards > 1 and image_index % total_shards != shard_id:
            continue
        if yielded >= num_samples:
            break
        class_name = os.path.basename(os.path.dirname(path))
        prompt = AWA2_CLASS_MAPPING[class_name]
        if categories is not None and class_name not in categories:
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue
        rel_path = os.path.relpath(path, root)
        image_id = os.path.splitext(os.path.basename(path))[0]
        split = _class_split(class_name, metadata) or ""
        license_rel, license_text = _license_metadata(root, rel_path)
        license_rel = license_rel or ""
        license_text = license_text or ""
        original_metadata = {
            "class_name": class_name,
            "class_index": metadata["class_index"].get(class_name),
            "split": split,
            "attribute_binary": metadata["attribute_binary"].get(class_name, {}),
            "attribute_continuous": metadata["attribute_continuous"].get(class_name, {}),
            "license_file": license_rel,
            "license_text": license_text,
        }
        yield {
            "image": image,
            "image_id": image_id,
            "file_name": rel_path,
            "species": prompt,
            "common_name": prompt,
            "label": class_name,
            "class_name": class_name,
            "class_index": metadata["class_index"].get(class_name),
            "prompt_class": prompt,
            "split": split,
            "attribute_binary": metadata["attribute_binary"].get(class_name, {}),
            "attribute_continuous": metadata["attribute_continuous"].get(class_name, {}),
            "license_file": license_rel,
            "license_text": license_text,
            "original_metadata": original_metadata,
        }
        yielded += 1


AWA2_CONFIG = DatasetConfig(
    name="AWA2",
    load_fn=_awa2_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: s.get("class_name"),
    class_mapping=AWA2_CLASS_MAPPING,
)

CONFIG = AWA2_CONFIG
