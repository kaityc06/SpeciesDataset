import argparse
import os
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import CONFIG, DEFAULT_PROCESS_SAMPLES, DEFAULT_TOTAL_SHARDS
import species_segmentation
from species_segmentation import process_dataset as _process_dataset, test_config
from test_duplicate_detection import dedup_visual_demo


DATASET_LABEL = os.path.basename(os.path.dirname(__file__))
QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN_REQUIRED_FILES = (
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
)
QWEN_CACHE_POLL_SECONDS = int(os.environ.get("QWEN_CACHE_POLL_SECONDS", "30"))
QWEN_CACHE_TIMEOUT_SECONDS = int(os.environ.get("QWEN_CACHE_TIMEOUT_SECONDS", "900"))
QWEN_LOAD_ATTEMPTS = int(os.environ.get("QWEN_LOAD_ATTEMPTS", "5"))
QWEN_LOAD_RETRY_SECONDS = int(os.environ.get("QWEN_LOAD_RETRY_SECONDS", "30"))

_ORIGINAL_QWEN_LOAD = species_segmentation.QwenMaskChecker._load
_ORIGINAL_MERGE_SHARDS = species_segmentation.merge_shards
_LOCAL_PATCHES_INSTALLED = False


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


def _wait_for_qwen_cache(model_name: str = QWEN_MODEL_ID) -> str:
    local_snapshot = os.environ.get("QWEN_LOCAL_SNAPSHOT")
    if local_snapshot:
        missing = _missing_qwen_cache_files(local_snapshot)
        if missing:
            raise RuntimeError(
                f"QWEN_LOCAL_SNAPSHOT is incomplete: {local_snapshot}; "
                f"missing {', '.join(missing)}"
            )
        print(f"[{DATASET_LABEL}] Using QWEN_LOCAL_SNAPSHOT: {local_snapshot}")
        return local_snapshot

    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(ROOT, ".hf_cache")), "hub"
    )
    model_dir = os.path.join(cache_root, "models--" + model_name.replace("/", "--"))
    refs_main = os.path.join(model_dir, "refs", "main")
    deadline = time.time() + QWEN_CACHE_TIMEOUT_SECONDS
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
                print(f"[{DATASET_LABEL}] Using local {model_name} snapshot: {snapshot_dir}")
                return snapshot_dir
        else:
            last_missing = [refs_main]

        print(f"[{DATASET_LABEL}] Waiting for local {model_name} cache: missing {', '.join(last_missing)}")
        time.sleep(QWEN_CACHE_POLL_SECONDS)

    raise RuntimeError(
        f"Timed out waiting for local {model_name} cache; "
        f"still missing: {', '.join(last_missing)}"
    )


def _patched_qwen_load(self):
    if self._model is not None:
        return
    if self._model_name != QWEN_MODEL_ID and not os.path.isdir(self._model_name):
        return _ORIGINAL_QWEN_LOAD(self)

    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

    for attempt in range(1, QWEN_LOAD_ATTEMPTS + 1):
        model_path = self._model_name if os.path.isdir(self._model_name) else _wait_for_qwen_cache(self._model_name)
        missing = _missing_qwen_cache_files(model_path) if os.path.isdir(model_path) else []
        if missing:
            raise RuntimeError(f"Local Qwen snapshot is incomplete: {model_path}; missing {', '.join(missing)}")
        try:
            print(f"[{DATASET_LABEL}] Loading Qwen3.5 model from {model_path} (attempt {attempt}/{QWEN_LOAD_ATTEMPTS})")
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path, device_map="auto"
            ).eval()
            print(f"[{DATASET_LABEL}] Qwen3.5 ready.")
            return
        except (OSError, RuntimeError) as e:
            self._model = None
            self._processor = None
            if attempt == QWEN_LOAD_ATTEMPTS:
                raise
            print(f"[{DATASET_LABEL}] Qwen load failed: {e}; retrying in {QWEN_LOAD_RETRY_SECONDS}s")
            time.sleep(QWEN_LOAD_RETRY_SECONDS)


def _infer_total_shards(shard_files: list[str]) -> int | None:
    totals = set()
    for path in shard_files:
        name = os.path.basename(path)
        if "_of_" not in name:
            continue
        try:
            totals.add(int(name.rsplit("_of_", 1)[1].split(".parquet", 1)[0]))
        except ValueError:
            continue
    return totals.pop() if len(totals) == 1 else None


def _parquet_readable(path: str) -> bool:
    """Return whether a shard/final parquet has a readable Arrow schema."""
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        import pyarrow.parquet as pq

        pq.read_schema(path)
        return True
    except Exception as e:
        print(f"[{DATASET_LABEL}] Unreadable parquet: {path}: {e}")
        return False


def _existing_output_is_final(output_path: str) -> bool:
    if not os.path.exists(output_path):
        return False
    if _parquet_readable(output_path):
        print(f"[{DATASET_LABEL}] Merge output already exists; skipping merge: {output_path}")
        return True
    backup_path = f"{output_path}.invalid_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"[{DATASET_LABEL}] Moving invalid merge output to {backup_path}")
    os.replace(output_path, backup_path)
    return False


def _patched_merge_shards(dataset_dir: str, output_path: str = None, delete_shards: bool = False) -> None:
    import glob

    if output_path is None:
        output_path = os.path.join(dataset_dir, "masks.parquet")
    if _existing_output_is_final(output_path):
        return

    shard_files = sorted(glob.glob(os.path.join(dataset_dir, "masks_*_of_*.parquet")))
    total_shards = _infer_total_shards(shard_files)
    if delete_shards and total_shards is not None:
        if len(shard_files) < total_shards:
            print(f"[{DATASET_LABEL}] Waiting to merge: {len(shard_files)}/{total_shards} shard files exist")
            return
        unreadable = [path for path in shard_files if not _parquet_readable(path)]
        if unreadable:
            print(f"[{DATASET_LABEL}] Waiting to merge: {len(unreadable)} unreadable shard(s)")
            return
        # Older full runs did not create marker files, so repair merges use the
        # complete readable shard set as the source of truth.

    lock_path = os.path.join(dataset_dir, "masks.merge.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[{DATASET_LABEL}] Merge already claimed by another shard")
        return

    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"pid={os.getpid()}\n")
        if _existing_output_is_final(output_path):
            return
        shard_files = sorted(glob.glob(os.path.join(dataset_dir, "masks_*_of_*.parquet")))
        total_shards = _infer_total_shards(shard_files)
        if delete_shards and total_shards is not None and len(shard_files) < total_shards:
            print(f"[{DATASET_LABEL}] Waiting to merge: {len(shard_files)}/{total_shards} shard files exist")
            return
        unreadable = [path for path in shard_files if not _parquet_readable(path)]
        if unreadable:
            print(f"[{DATASET_LABEL}] Waiting to merge: {len(unreadable)} unreadable shard(s)")
            return
        _ORIGINAL_MERGE_SHARDS(dataset_dir, output_path=output_path, delete_shards=delete_shards)
    except FileExistsError:
        print(f"[{DATASET_LABEL}] Merge output already exists; skipping merge: {output_path}")
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def _install_dataset_local_patches() -> None:
    global _LOCAL_PATCHES_INSTALLED
    if _LOCAL_PATCHES_INSTALLED:
        return
    species_segmentation.QwenMaskChecker._load = _patched_qwen_load
    species_segmentation.merge_shards = _patched_merge_shards
    _LOCAL_PATCHES_INSTALLED = True


def _shard_paths(config, shard_id: int, total_shards: int) -> tuple[str, str]:
    out_dir = species_segmentation._dataset_output_dir(config)
    shard_path = os.path.join(out_dir, f"masks_{shard_id:05d}_of_{total_shards:05d}.parquet")
    return out_dir, f"{shard_path}.done"


def _write_done_marker(marker_path: str) -> None:
    tmp_marker = f"{marker_path}.tmp.{os.getpid()}"
    with open(tmp_marker, "w", encoding="utf-8") as f:
        f.write(f"finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write(f"dataset_local_patch={DATASET_LABEL}\n")
    os.replace(tmp_marker, marker_path)


def process_dataset(config, *args, shard_id: int = 0, total_shards: int = 1, **kwargs) -> None:
    _install_dataset_local_patches()
    out_dir = None
    done_marker = None
    if total_shards > 1:
        out_dir, done_marker = _shard_paths(config, shard_id, total_shards)
        if os.path.exists(done_marker):
            os.remove(done_marker)
    _process_dataset(config, *args, shard_id=shard_id, total_shards=total_shards, **kwargs)
    if done_marker is not None:
        _write_done_marker(done_marker)
        species_segmentation.merge_shards(out_dir, delete_shards=True)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test-all", "test-qwen", "dedup", "process"], default="test-all")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dedup-threshold", type=float, default=float(os.environ.get("DEDUP_THRESHOLD", 0.99)))
    parser.add_argument("--shard-id", type=int, default=_env_int("SLURM_ARRAY_TASK_ID", 0))
    parser.add_argument(
        "--total-shards",
        type=int,
        default=_env_int("TOTAL_SHARDS", _env_int("SLURM_ARRAY_TASK_COUNT", DEFAULT_TOTAL_SHARDS)),
    )
    args = parser.parse_args()
    _install_dataset_local_patches()

    if args.mode in {"test-all", "test-qwen"}:
        test_config(
            CONFIG,
            qwen_crop=True,
            qwen_isolate_mask=True,
            num_samples=args.num_samples or 30,
            dedup_threshold=args.dedup_threshold,
        )
    if args.mode in {"test-all", "dedup"}:
        dedup_visual_demo(CONFIG, num_images=args.num_samples or 30, threshold=args.dedup_threshold)
    if args.mode == "process":
        process_dataset(
            CONFIG,
            qwen_isolate_mask=True,
            qwen_crop=True,
            shard_id=args.shard_id,
            total_shards=args.total_shards,
            num_samples=args.num_samples or DEFAULT_PROCESS_SAMPLES,
            dedup_threshold=args.dedup_threshold,
        )


if __name__ == "__main__":
    main()
