# FishNet

Source: FishNet ICCV 2023 project data. The config downloads annotation files from the public FishNet GitHub repo and expects the image archive from the FishNet Google Drive link.

Scope: all 94,532 images, processed as 379 shards with up to 250 samples per shard.

Cache: `datasets/FishNet/data/`

SAM3 prompt mapping: all samples use `fish`; bounding boxes are used if the annotation files expose bbox columns.

If the Google Drive image download does not work automatically, install `gdown` in the species conda env or place the FishNet image zip at:

`datasets/FishNet/data/fishnet_images.zip`

Visualizations:

- `outputs/report.html`
- `outputs/dedup_report.html`

Processing output:

- `outputs/masks.parquet`
