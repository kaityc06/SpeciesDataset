#!/usr/bin/env python
"""Snapshot Serengeti-only full processing runner.

This file intentionally lives under datasets/SnapshotSerengeti instead of
changing species_segmentation.py. The failed full run showed a Snapshot-specific
streaming parquet issue:

- The shared runner writes parquet incrementally.
- Snapshot Serengeti's metadata field "bbox" can be None for some accepted masks
  and a JSON string for later accepted masks.
- If the first written batch has only None for bbox, PyArrow infers bbox as
  type null. When a later batch has a real bbox string, the ParquetWriter raises
  "Table schema does not match schema used to create file".

The fix here keeps the core numeric mask fields numeric, but writes all
provenance and dataset metadata columns as nullable strings. That makes bbox
stable as string whether a particular row has a value or not.

This runner also avoids the shared runner's eager shard merge. A parquet shard
file exists while its job is still running, so checking only for file existence
can merge partial output. This script writes a ".done" marker only after a shard
finishes successfully, then merges once all 400 done markers exist.
"""

import argparse
import os
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CONFIG, DEFAULT_PROCESS_SAMPLES, DEFAULT_TOTAL_SHARDS
from duplicate_detection import DuplicateDetector
from species_segmentation import (
    DEDUP_THRESHOLD,
    _dataset_output_dir,
    _init_processor,
    _make_stream,
    merge_shards,
    process_sample,
)


NUMERIC_COLUMNS = {
    "mask_rle_height",
    "mask_rle_width",
    "mask_score",
    "image_width",
    "image_height",
}

QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN_REQUIRED_FILES = (
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
)
QWEN_CACHE_POLL_SECONDS = 30
QWEN_LOAD_ATTEMPTS = 5
QWEN_LOAD_RETRY_SECONDS = 30


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _stable_table(records: list[dict], schema: pa.Schema | None = None) -> pa.Table:
    """Build an Arrow table whose non-numeric columns have stable string type."""
    df = pd.DataFrame(records)
    for column in df.columns:
        if column not in NUMERIC_COLUMNS:
            df[column] = df[column].astype("string")

    table = pa.Table.from_pandas(df, preserve_index=False)
    if schema is not None:
        table = table.cast(schema)
    return table


def _write_done_marker(marker_path: str, n_processed: int, n_masks: int) -> None:
    """Write this only after the shard has closed its parquet writer cleanly."""
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"samples={n_processed}\n")
        f.write(f"masks={n_masks}\n")


def _qwen_cache_file_ready(snapshot_dir: str, filename: str) -> bool:
    path = os.path.join(snapshot_dir, filename)
    if not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) <= 0:
            return False
        if os.path.islink(path):
            target = os.path.realpath(path)
            return os.path.exists(target) and os.path.getsize(target) > 0
    except OSError:
        return False
    return True


def _missing_qwen_cache_files(snapshot_dir: str) -> list[str]:
    return [
        filename
        for filename in QWEN_REQUIRED_FILES
        if not _qwen_cache_file_ready(snapshot_dir, filename)
    ]


def _wait_for_qwen_cache(timeout_seconds: int = 900) -> str:
    """Avoid transient failures while many array tasks read the shared HF cache."""
    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(ROOT, ".hf_cache")), "hub"
    )
    model_dir = os.path.join(cache_root, "models--Qwen--Qwen3.5-4B")
    refs_main = os.path.join(model_dir, "refs", "main")
    deadline = time.time() + timeout_seconds
    last_missing: list[str] = []

    while time.time() < deadline:
        if os.path.exists(refs_main):
            with open(refs_main, "r", encoding="utf-8") as f:
                revision = f.read().strip()
            snapshot_dir = os.path.join(model_dir, "snapshots", revision)
            if not os.path.isdir(snapshot_dir):
                last_missing = [snapshot_dir]
            else:
                last_missing = _missing_qwen_cache_files(snapshot_dir)
            if not last_missing:
                print(f"[snapshot_stable] Using local {QWEN_MODEL_ID} snapshot: {snapshot_dir}")
                return snapshot_dir
        else:
            last_missing = [refs_main]

        print(
            f"[snapshot_stable] Waiting for local {QWEN_MODEL_ID} cache: "
            f"missing {', '.join(last_missing)}"
        )
        time.sleep(QWEN_CACHE_POLL_SECONDS)

    raise RuntimeError(
        f"Timed out waiting for local {QWEN_MODEL_ID} cache; "
        f"still missing: {', '.join(last_missing)}"
    )


def _reset_qwen_checker(qwen_checker) -> None:
    qwen_checker._model = None
    qwen_checker._processor = None


def _warm_qwen_checker(config, qwen_model_path: str) -> None:
    """Load Qwen from the validated local snapshot before processing samples."""
    qwen_checker = getattr(config, "_qwen_checker", None)
    if qwen_checker is None:
        return

    qwen_checker._model_name = qwen_model_path
    for attempt in range(1, QWEN_LOAD_ATTEMPTS + 1):
        try:
            print(
                f"[snapshot_stable] Warming Qwen checker from local snapshot "
                f"(attempt {attempt}/{QWEN_LOAD_ATTEMPTS})"
            )
            qwen_checker._load()
            return
        except (OSError, RuntimeError) as e:
            _reset_qwen_checker(qwen_checker)
            if attempt == QWEN_LOAD_ATTEMPTS:
                raise
            print(
                f"[snapshot_stable] Qwen warmup failed: {e}; "
                f"retrying in {QWEN_LOAD_RETRY_SECONDS}s"
            )
            time.sleep(QWEN_LOAD_RETRY_SECONDS)


def _try_claim_merge(out_dir: str, total_shards: int, shard_id: int) -> str | None:
    """Create an exclusive lock so only one completed shard performs the merge."""
    lock_path = os.path.join(out_dir, f"masks_{total_shards:05d}.merge.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None

    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
        lock_file.write(f"claimed_by_shard={shard_id}\n")
    return lock_path


def _maybe_merge_completed_shards(out_dir: str, total_shards: int, shard_id: int) -> None:
    """Merge only after every shard has written a success marker."""
    done_markers = [
        os.path.join(out_dir, f"masks_{i:05d}_of_{total_shards:05d}.parquet.done")
        for i in range(total_shards)
    ]
    if not all(os.path.exists(path) for path in done_markers):
        return

    lock_path = _try_claim_merge(out_dir, total_shards, shard_id)
    if lock_path is None:
        print("[snapshot_stable] Merge already claimed by another shard")
        return

    try:
        merge_shards(out_dir, delete_shards=True)
        for marker_path in done_markers:
            if os.path.exists(marker_path):
                os.remove(marker_path)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def process_snapshot_stable(
    num_samples: int | None,
    shard_id: int,
    total_shards: int,
    dedup_threshold: float,
) -> None:
    """Snapshot Serengeti processing loop with stable parquet and done markers."""
    out_dir = _dataset_output_dir(CONFIG)
    parquet_output = os.path.join(
        out_dir, f"masks_{shard_id:05d}_of_{total_shards:05d}.parquet"
    )
    done_marker = f"{parquet_output}.done"

    # Each rerun owns its shard path. Removing only this shard avoids touching
    # unrelated output while ensuring failed partial files do not survive.
    if os.path.exists(parquet_output):
        print(f"[snapshot_stable] Removing existing shard: {parquet_output}")
        os.remove(parquet_output)
    if os.path.exists(done_marker):
        print(f"[snapshot_stable] Removing existing done marker: {done_marker}")
        os.remove(done_marker)

    limit = num_samples if num_samples is not None else sys.maxsize
    print(
        "[snapshot_stable] SnapshotSerengeti  |  "
        f"shard={shard_id}/{total_shards}  |  "
        f"samples={'all' if num_samples is None else num_samples}  |  "
        f"parquet={parquet_output}"
    )

    qwen_model_path = _wait_for_qwen_cache()
    processor = _init_processor(CONFIG)
    _warm_qwen_checker(CONFIG, qwen_model_path)
    ds_stream = _make_stream(
        CONFIG,
        limit,
        shard_id=shard_id,
        total_shards=total_shards,
    )
    duplicate_detector = DuplicateDetector(threshold=dedup_threshold)

    writer = None
    total_masks = 0
    n_processed = 0
    n_dupes = 0
    start = time.time()

    it = iter(ds_stream)
    try:
        for i in range(limit):
            sample = next(it)
            records, render_info = process_sample(
                processor,
                sample,
                CONFIG,
                sample_idx=i,
                qwen_crop=True,
                qwen_isolate_mask=True,
                duplicate_detector=duplicate_detector,
            )
            if render_info is None and not records:
                n_dupes += 1
                continue
            if records:
                table = _stable_table(records, writer.schema if writer else None)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_output, table.schema)
                writer.write_table(table)
                total_masks += len(records)
            n_processed += 1
    except StopIteration:
        pass
    finally:
        if hasattr(it, "close"):
            it.close()
        if writer is not None:
            writer.close()

    elapsed = time.time() - start
    h, m, s = int(elapsed) // 3600, (int(elapsed) % 3600) // 60, int(elapsed) % 60
    rate = n_processed / elapsed if elapsed > 0 else 0
    print(f"Dedup: {n_dupes} duplicate(s) skipped")
    if total_masks:
        print(
            f"Done: {n_processed} samples  {total_masks} masks  "
            f"{h}h {m}m {s}s  ({rate:.2f} img/s)  -> {parquet_output}"
        )
    else:
        print(f"Done: {n_processed} samples  no masks accepted  {h}h {m}m {s}s")

    _write_done_marker(done_marker, n_processed, total_masks)
    _maybe_merge_completed_shards(out_dir, total_shards, shard_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=_env_int("SLURM_ARRAY_TASK_ID", 0))
    parser.add_argument(
        "--total-shards",
        type=int,
        default=_env_int("TOTAL_SHARDS", _env_int("SLURM_ARRAY_TASK_COUNT", DEFAULT_TOTAL_SHARDS)),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=float(os.environ.get("DEDUP_THRESHOLD", DEDUP_THRESHOLD)),
    )
    args = parser.parse_args()

    process_snapshot_stable(
        num_samples=args.num_samples or DEFAULT_PROCESS_SAMPLES,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        dedup_threshold=args.dedup_threshold,
    )


if __name__ == "__main__":
    main()
