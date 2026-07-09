---
license: other
task_categories:
- image-segmentation
language:
- en
tags:
- wildlife
- animals
- segmentation
- masks
- rle
- coco
- awa2
- animals-with-attributes
- attributes
- zero-shot
size_categories:
- 10K<n<100K
pretty_name: AWA2 Masked
---

# AWA2 Masked

Segmentation masks for the [Animals with Attributes 2](https://cvml.ista.ac.at/AwA2/) dataset, stored as RLE-encoded masks in a single Parquet file.

## File

| File | Description |
| --- | --- |
| `masks.parquet` | 53,557 rows — one per accepted mask — with RLE mask, score, source image metadata, normalized class prompt, raw AwA2 class label, official zero-shot split, class attribute metadata, and per-image license metadata |

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `dataset` | str | Always `AWA2` |
| `text_prompt` | str | Text prompt used to generate the mask, e.g. `grizzly bear`, `blue whale`, `giant panda`, `german shepherd` |
| `mask_rle_counts` | str | RLE-encoded mask counts in COCO format |
| `mask_rle_height` | int | Height used for RLE decoding |
| `mask_rle_width` | int | Width used for RLE decoding |
| `mask_score` | float | Mask confidence score |
| `image_width` | int | Original image width |
| `image_height` | int | Original image height |
| `image_id` | str | AwA2 image identifier derived from the source image filename |
| `file_name` | str | Original AwA2 image path relative to the `Animals_with_Attributes2` archive root |
| `species` | str | Normalized AwA2 class prompt copied into the shared species field |
| `common_name` | str | Normalized AwA2 class prompt copied into the shared common-name field |
| `label` | str | Raw AwA2 class folder label |
| `class_name` | str | Raw AwA2 class folder label |
| `class_index` | int | Official AwA2 class index from `classes.txt` |
| `prompt_class` | str | Normalized class prompt used for segmentation |
| `split` | str | Official AwA2 zero-shot split, either `train` or `test` |
| `attribute_binary` | str | JSON-encoded class-level binary vector for the 85 AwA2 attributes |
| `attribute_continuous` | str | JSON-encoded class-level continuous vector for the 85 AwA2 attributes |
| `license_file` | str | Path to the per-image AwA2 license metadata file |
| `license_text` | str | Contents of the per-image AwA2 license metadata file |
| `original_metadata` | str | JSON-encoded bundle of class label, class index, split, class attributes, and per-image license metadata |

## Loading the dataset

Load the Parquet file directly from Hugging Face.

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/suryadv/AWA2_masked/masks.parquet"
)

print(len(df))
print(df.columns.tolist())
```

## Retrieving the original image

The `file_name` column stores the source image path under the AwA2 image archive root.

```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from PIL import Image

AWA2_DATA_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"

source_dir = Path("awa2_source")
source_dir.mkdir(exist_ok=True)

zip_path = source_dir / "AwA2-data.zip"
if not zip_path.exists():
    urlretrieve(AWA2_DATA_URL, zip_path)

row = df.iloc[0]

file_name = row["file_name"].replace("\\", "/")
member = f"Animals_with_Attributes2/{file_name}"

with ZipFile(zip_path) as zf:
    with zf.open(member) as f:
        img = Image.open(f).convert("RGB")

img
```

## Retrieving an image by class

Look up an example image corresponding to a particular AwA2 class prompt.

```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from PIL import Image

AWA2_DATA_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"

source_dir = Path("awa2_source")
source_dir.mkdir(exist_ok=True)

zip_path = source_dir / "AwA2-data.zip"
if not zip_path.exists():
    urlretrieve(AWA2_DATA_URL, zip_path)

class_prompt = "giant panda"

row = df[
    df["prompt_class"].str.lower() == class_prompt.lower()
].iloc[0]

file_name = row["file_name"].replace("\\", "/")
member = f"Animals_with_Attributes2/{file_name}"

with ZipFile(zip_path) as zf:
    with zf.open(member) as f:
        img = Image.open(f).convert("RGB")

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

AwA2 source images have per-image licenses. This dataset preserves the per-image AwA2 license metadata in `license_file` and `license_text`.

Please also consult the original Animals with Attributes 2 terms when using the source imagery or metadata.
