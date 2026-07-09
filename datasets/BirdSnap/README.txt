---
license: other
task_categories:
- image-segmentation
language:
- en
tags:
- wildlife
- birds
- fine-grained
- segmentation
- masks
- rle
- coco
- birdsnap
- flickr
size_categories:
- 10K<n<100K
pretty_name: BirdSnap Masked
---

# BirdSnap Masked

Segmentation masks for the [BirdSnap](https://huggingface.co/datasets/sasha/birdsnap) dataset, stored as RLE-encoded masks in a single Parquet file.

## File

| File | Description |
| --- | --- |
| `masks.parquet` | 37,693 rows — one per accepted mask — with RLE mask, score, source image identifier, normalized bird species/common-name label, raw BirdSnap label, split, and preserved source row metadata |

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `dataset` | str | Always `BirdSnap` |
| `text_prompt` | str | Text prompt used to generate the mask, always `bird` |
| `mask_rle_counts` | str | RLE-encoded mask counts in COCO format |
| `mask_rle_height` | int | Height used for RLE decoding |
| `mask_rle_width` | int | Width used for RLE decoding |
| `mask_score` | float | Mask confidence score |
| `image_width` | int | Original image width |
| `image_height` | int | Original image height |
| `image_id` | str | BirdSnap source stream identifier, e.g. `BirdSnap_000000` |
| `file_name` | str | Source file name when exposed by the Hugging Face row, otherwise the same synthetic BirdSnap identifier as `image_id` |
| `species` | str | Bird species/common name normalized from the raw label, with underscores replaced by spaces |
| `common_name` | str | Bird species/common name normalized from the raw label, with underscores replaced by spaces |
| `label` | str | Raw BirdSnap label from the Hugging Face dataset, e.g. `Greater_White_fronted_Goose` |
| `split` | str | Source Hugging Face split, always `train` |
| `original_metadata` | str | JSON-encoded non-image metadata preserved from the source Hugging Face row |

## Loading the dataset

Load the Parquet file directly from Hugging Face.

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/suryadv/BirdSnap_masked/masks.parquet"
)

print(len(df))
print(df.columns.tolist())
```

## Retrieving the original image

The `image_id` column stores the source row index in the `sasha/birdsnap` Hugging Face train stream.

```python
from itertools import islice

from datasets import load_dataset

row = df.iloc[0]

source_index = int(row["image_id"].rsplit("_", 1)[1])

source = load_dataset(
    "sasha/birdsnap",
    split="train",
    streaming=True,
)

sample = next(islice(source, source_index, None))
img = sample["image"].convert("RGB")

img
```

## Retrieving an image by species

Look up an example image corresponding to a particular BirdSnap species/common name.

```python
from itertools import islice

from datasets import load_dataset

species = "Wilsons Phalarope"

row = df[
    df["species"].str.lower() == species.lower()
].iloc[0]

source_index = int(row["image_id"].rsplit("_", 1)[1])

source = load_dataset(
    "sasha/birdsnap",
    split="train",
    streaming=True,
)

sample = next(islice(source, source_index, None))
img = sample["image"].convert("RGB")

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

Please consult the original BirdSnap / Hugging Face dataset terms when using the source imagery or metadata.

If you use BirdSnap, please also cite the original BirdSnap CVPR 2014 paper.
