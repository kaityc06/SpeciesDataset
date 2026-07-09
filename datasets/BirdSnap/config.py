import io
import os
from typing import Iterable

from PIL import Image, ImageFile

from species_segmentation import DatasetConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATASET_DIR, "data")
BIRDSNAP_DATASET_PATH = "sasha/birdsnap"
BIRDSNAP_SPLIT = "train"

DEFAULT_TOTAL_SHARDS = 160
DEFAULT_PROCESS_SAMPLES = 250
TARGET_SAMPLES = 39_860

BIRDSNAP_CLASS_MAPPING = {"Aves": "bird"}


def _clean_label(label: str) -> str:
    return str(label).replace("_", " ").strip()


def _serializable_value(value):
    """Convert Hugging Face metadata values into parquet-safe Python values."""
    if isinstance(value, Image.Image):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_serializable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable_value(val) for key, val in value.items()}
    return str(value)


def _original_metadata(sample: dict):
    """Preserve every non-image field exposed by the sasha/birdsnap stream."""
    metadata = {}
    for key, value in sample.items():
        if key == "image":
            continue
        converted = _serializable_value(value)
        if converted is not None:
            metadata[key] = converted
    return metadata


def _decode_image(image_data):
    """Decode a Hugging Face image record while tolerating truncated JPEG tails."""
    try:
        if isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, dict):
            if image_data.get("bytes") is not None:
                image = Image.open(io.BytesIO(image_data["bytes"]))
            elif image_data.get("path"):
                image = Image.open(image_data["path"])
            else:
                return None
        else:
            return None
        image.load()
        return image.convert("RGB")
    except Exception as e:
        print(f"  Skipping BirdSnap image decode failure: {e}")
        return None


def _birdsnap_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
) -> Iterable:
    if categories is not None and "Aves" not in categories:
        return

    from datasets import Image as HFImage
    from datasets import load_dataset

    dataset = load_dataset(BIRDSNAP_DATASET_PATH, split=BIRDSNAP_SPLIT, streaming=True)
    dataset = dataset.cast_column("image", HFImage(decode=False))
    yielded = 0
    for image_index, sample in enumerate(dataset):
        if total_shards > 1 and image_index % total_shards != shard_id:
            continue
        if yielded >= num_samples:
            break

        image = _decode_image(sample.get("image"))
        if image is None:
            continue

        original_metadata = _original_metadata(sample)
        label = sample.get("label")
        species = _clean_label(label)
        file_name = original_metadata.get("file_name") or getattr(image, "filename", None)
        row = {
            "image": image,
            "image_id": f"BirdSnap_{image_index:06d}",
            "file_name": file_name or f"BirdSnap_{image_index:06d}",
            "species": species,
            "common_name": species,
            "label": label,
            "split": BIRDSNAP_SPLIT,
            "original_metadata": original_metadata,
        }
        yield row
        yielded += 1


BIRDSNAP_CONFIG = DatasetConfig(
    name="BirdSnap",
    load_fn=_birdsnap_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: "Aves",
    class_mapping=BIRDSNAP_CLASS_MAPPING,
)

CONFIG = BIRDSNAP_CONFIG
