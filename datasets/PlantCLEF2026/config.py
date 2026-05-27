import os
import csv
import requests
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


# ── Download (no auth required) ───────────────────────────────────────────────
#
# PlantCLEF2026 reuses the PlantCLEF2024 single-plant training set plus the
# PlantCLEF2025 pseudoquadrat complementary set. The task is multi-label
# identification of all plant species in 50×50 cm vegetation plot images.
#
#   Training metadata CSV (~750 MB):
#     wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.csv
#   Training images, 800 px max side (~160 GB):
#     wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar
#   Training images, full resolution (~281 GB):
#     wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.tar
#   Complementary unlabeled cover images (~170 GB):
#     wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2025/pseudoquadrats_without_labels_complementary_training_set/PlantCLEF2025_pseudoquadrats_without_labels_complementary_training_set.tar
#   Test set (vegetation plots):
#     wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2025/vegetation_plot_test_data/PlantCLEF2025test.tar
#
# Expected layout after extraction:
#   $PLANTCLEF2026_DIR/PlantCLEF2024singleplanttrainingdata.csv
#   $PLANTCLEF2026_DIR/PlantCLEF2024singleplanttrainingdata/<image_name>
#
# ~1.4 M images, 7 800 vascular plant species from SW Europe (Pl@ntNet).
# Key CSV columns: image_backup_url, gbif_species_id, species, genus, family.
# Verify remaining column names from the actual CSV header before use.

_PLANTCLEF2026_BASE = (
    "https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data"
)
_PLANTCLEF2026_DIR = "/mmfs1/gscratch/krishna/kaityc/lifeclef2026_cache/plantclef2026"
_PLANTCLEF2026_CSV = os.path.join(
    _PLANTCLEF2026_DIR, "PlantCLEF2024singleplanttrainingdata.csv"
)
_PLANTCLEF2026_IMG_DIR = os.path.join(
    _PLANTCLEF2026_DIR, "images_max_side_800"
)


def _ensure_plantclef2026_csv():
    if os.path.exists(_PLANTCLEF2026_CSV):
        return
    import urllib.request

    os.makedirs(_PLANTCLEF2026_DIR, exist_ok=True)
    url = f"{_PLANTCLEF2026_BASE}/PlantCLEF2024singleplanttrainingdata.csv"
    print("Downloading PlantCLEF2026 training metadata CSV (~750 MB) …")
    urllib.request.urlretrieve(url, _PLANTCLEF2026_CSV)
    print(f"Cached to {_PLANTCLEF2026_CSV}")


def _plantclef2026_stream(
    num_samples: int, shard_id: int = 0, total_shards: int = 1
) -> Iterable:
    _ensure_plantclef2026_csv()

    with open(_PLANTCLEF2026_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    for row in rows[shard_id::total_shards][:num_samples]:
        image_name = row.get("image_name") or row.get("filename") or row.get("file_name")
        species_id = row.get("species_id")
        local_path = os.path.join(_PLANTCLEF2026_IMG_DIR, species_id, image_name) if (image_name and species_id) else None

        if local_path and os.path.exists(local_path):
            try:
                pil_img = Image.open(local_path).convert("RGB")
            except Exception as e:
                print(f"  Skipping {local_path}: {e}")
                continue
        else:
            backup_url = row.get("image_backup_url")
            if not backup_url:
                continue
            try:
                resp = requests.get(backup_url, timeout=10)
                resp.raise_for_status()
                pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                print(f"  Skipping {backup_url}: {e}")
                continue

        yield {
            "image":        pil_img,
            "species":      row.get("species") or row.get("classname") or row.get("scientific_name"),
            "species_id":   row.get("classid") or row.get("species_id"),
            "gbif_id":      row.get("gbif_species_id"),
            "genus":        row.get("genus"),
            "family":       row.get("family"),
        }


PLANTCLEF2026_CONFIG = DatasetConfig(
    name="PlantCLEF2026",
    load_fn=_plantclef2026_stream,
    get_image=lambda s: s["image"],
    get_prompt=lambda s: "plant",
)
