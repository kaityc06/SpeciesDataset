---
license: cc-by-4.0
task_categories:
- image-segmentation
language:
- en
tags:
- wildlife
- camera-trap
- segmentation
- masks
- rle
- coco
- snapshot-serengeti
size_categories:
- 10K<n<100K
pretty_name: Snapshot Serengeti Masked
---

# Snapshot Serengeti Masked

Segmentation masks for the [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti/) dataset, stored as RLE-encoded masks in a single Parquet file.

## File

| File | Description |
| --- | --- |
| `masks.parquet` | 35,730 rows — one per accepted mask — with RLE mask, score, source image metadata, prompt class, raw category label, normalized bounding box when available, and preserved LILA image/category/annotation metadata |

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `dataset` | str | Always `SnapshotSerengeti` |
| `text_prompt` | str | Text prompt used to generate the mask, e.g. `zebra`, `wildebeest`, `bird`, `thomson gazelle` |
| `mask_rle_counts` | str | RLE-encoded mask counts in COCO format |
| `mask_rle_height` | int | Height used for RLE decoding |
| `mask_rle_width` | int | Width used for RLE decoding |
| `mask_score` | float | Mask confidence score |
| `image_width` | int | Original image width |
| `image_height` | int | Original image height |
| `image_id` | str | Snapshot Serengeti image identifier |
| `file_name` | str | Original Snapshot Serengeti image path relative to the LILA image root |
| `species` | str | Raw Snapshot Serengeti category label copied into the shared species field |
| `common_name` | str | Raw Snapshot Serengeti category label copied into the shared common-name field |
| `label` | str | Raw Snapshot Serengeti category label copied into the shared label field |
| `category` | str | Raw Snapshot Serengeti category label |
| `prompt_class` | str | Normalized class prompt used for segmentation |
| `bbox` | str | Bounding box `[x0, y0, x1, y1]` normalized to `[0, 1]`, or null/empty if unavailable |
| `image_metadata` | str | JSON-encoded original LILA image record |
| `annotation_metadata` | str | JSON-encoded primary Snapshot Serengeti annotation record used for this sample |
| `all_annotations_metadata` | str | JSON-encoded list of Snapshot Serengeti annotation records associated with the source image after excluded categories are removed |
| `category_metadata` | str | JSON-encoded original LILA category record for the primary annotation |
| `bbox_annotation_metadata` | str | JSON-encoded primary Snapshot Serengeti bbox annotation record when available |
| `all_bbox_annotations_metadata` | str | JSON-encoded list of Snapshot Serengeti bbox annotation records associated with the source image |
| `original_metadata` | str | JSON-encoded bundle of the original LILA image record, category annotation records, category record, and bbox annotation records |

## Loading the dataset

Load the Parquet file directly from Hugging Face.

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/suryadv/Snapshot-Serengeti_masked/masks.parquet"
)

print(len(df))
print(df.columns.tolist())
```

## Retrieving the original image

The `file_name` column stores the source image path under the Snapshot Serengeti LILA image root.

```python
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image

IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshotserengeti-unzipped/"
)

row = df.iloc[0]

file_name = row["file_name"].replace("\\", "/")
url = IMAGE_BASE_URL + quote(file_name, safe="/")

img = Image.open(
    BytesIO(requests.get(url, timeout=30).content)
).convert("RGB")

img
```

## Retrieving an image by category

Look up an example image corresponding to a particular Snapshot Serengeti category.

```python
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image

IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshotserengeti-unzipped/"
)

category = "zebra"

row = df[
    df["category"].str.lower() == category.lower()
].iloc[0]

file_name = row["file_name"].replace("\\", "/")
url = IMAGE_BASE_URL + quote(file_name, safe="/")

img = Image.open(
    BytesIO(requests.get(url, timeout=30).content)
).convert("RGB")

img
```

## Decoding a mask

Decode a COCO-format RLE mask into a binary NumPy array.

```python
from pycocotools import mask as mask_utils

row = df.iloc[0]

rle = {
    "counts": row["mask_rle_counts"],
    "size": [
        row["mask_rle_height"],
        row["mask_rle_width"],
    ],
}

binary_mask = mask_utils.decode(rle)

print(binary_mask.shape)
```

The decoded mask is a NumPy array with shape `(H, W)` and dtype `uint8`.

## License

This dataset is released under **CC-BY-4.0**.

Please also consult the original Snapshot Serengeti / LILA dataset terms when using the source imagery or metadata.
