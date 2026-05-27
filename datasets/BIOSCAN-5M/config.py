import os
from typing import Iterable

from species_segmentation import DatasetConfig


BIOSCAN_METADATA_URL = (
    "https://zenodo.org/records/11973457/files/"
    "BIOSCAN_5M_Insect_Dataset_metadata_MultiTypes.zip"
)
BIOSCAN_METADATA_DIR  = "/mmfs1/gscratch/krishna/kaityc/bioscan_cache/bioscan5m/metadata/csv"
BIOSCAN_METADATA_PATH = os.path.join(BIOSCAN_METADATA_DIR, "BIOSCAN_5M_Insect_Dataset_metadata.csv")


def _ensure_bioscan_metadata():
    if os.path.exists(BIOSCAN_METADATA_PATH):
        return
    import urllib.request
    import zipfile

    os.makedirs(BIOSCAN_METADATA_DIR, exist_ok=True)
    zip_path = os.path.join(BIOSCAN_METADATA_DIR, "metadata_MultiTypes.zip")
    print(f"Downloading BIOSCAN-5M metadata to {zip_path} ...")
    urllib.request.urlretrieve(BIOSCAN_METADATA_URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(BIOSCAN_METADATA_DIR)
    os.remove(zip_path)
    print(f"Metadata ready at {BIOSCAN_METADATA_PATH}")


def _bioscan_stream(num_samples: int, categories=None) -> Iterable:
    import pandas as pd
    from datasets import load_dataset

    _ensure_bioscan_metadata()

    TAXONOMY_COLS = ["processid", "split", "class", "order", "family", "genus", "species"]

    hf_iter = iter(load_dataset("bioscan-ml/BIOSCAN-5M", split="train", streaming=True))
    yielded = 0

    for chunk in pd.read_csv(BIOSCAN_METADATA_PATH, usecols=TAXONOMY_COLS, chunksize=10_000):
        if yielded >= num_samples:
            break
        for _, row in chunk[chunk["split"] == "train"].iterrows():
            if yielded >= num_samples:
                break
            hf_sample = next(hf_iter, None)
            if hf_sample is None:
                return
            cls = row["class"] if pd.notna(row.get("class")) else None
            if categories is not None and cls not in categories:
                continue
            yield {
                "image":   hf_sample["image"],
                "class":   cls,
                "order":   row["order"]   if pd.notna(row.get("order"))   else None,
                "family":  row["family"]  if pd.notna(row.get("family"))  else None,
                "genus":   row["genus"]   if pd.notna(row.get("genus"))   else None,
                "species": row["species"] if pd.notna(row.get("species")) else None,
            }
            yielded += 1


BIOSCAN_CONFIG = DatasetConfig(
    name="BIOSCAN-5M",
    load_fn=_bioscan_stream,
    get_image=lambda s: s["image"].convert("RGB") if s.get("image") else None,
    get_class=lambda s: s.get("class"),
    class_mapping={
        "Insecta": "insect",
        "Arachnida": "spider or mite",
        "Malacostraca": "shellfish",
        "Collembola": "springtail",
        "Diplopoda": "millipede",
        "Chilopoda": "centipede",
    },
)
