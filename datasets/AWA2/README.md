# AWA2

Source: official Animals with Attributes 2 downloads from <https://cvml.ista.ac.at/AwA2/>.

Scope: all 37,322 images, processed as 150 shards with up to 250 samples per shard.

Cache: `datasets/AWA2/data/`

SAM3 prompt mapping: class folder names are normalized into natural prompts, for example `grizzly+bear` -> `grizzly bear`, `blue+whale` -> `blue whale`, `persian+cat` -> `persian cat`, and `german+shepherd` -> `german shepherd`.

Visualizations:

- `outputs/report.html`
- `outputs/dedup_report.html`

Processing output:

- `outputs/masks.parquet`
