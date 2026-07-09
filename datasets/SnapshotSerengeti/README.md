# Snapshot Serengeti

Source: LILA Snapshot Serengeti v2.1 metadata and image files from <https://lila.science/datasets/snapshot-serengeti/>.

Scope: deterministic 100k subset of non-empty, non-human wildlife images, processed as 400 shards with 250 samples per shard.

Cache: `datasets/SnapshotSerengeti/data/`

SAM3 prompt mapping: category names are normalized to common animal prompts. Broad labels map as `otherBird` -> `bird`, `reptiles` -> `reptile`, and `rodents` -> `rodent`. Bounding boxes are used when available from the Snapshot Serengeti bbox metadata.

Visualizations:

- `outputs/report.html`
- `outputs/dedup_report.html`

Processing output:

- `outputs/masks.parquet`
