import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from config import INAT21_CONFIG
import species_segmentation


_SHARED_QWEN_LOCK = "/mmfs1/gscratch/krishna/kaityc/qwen_hf_download.lock"


def _configure_runtime_paths() -> None:
    output_dir = os.environ.get("INAT21_OUTPUT_DIR")
    if output_dir:
        output_dir = os.path.abspath(output_dir)

        def _inat21_output_dir(_config):
            os.makedirs(output_dir, exist_ok=True)
            return output_dir

        species_segmentation._dataset_output_dir = _inat21_output_dir

    qwen_lock_path = os.environ.get("QWEN_HF_DOWNLOAD_LOCK")
    if qwen_lock_path:
        import filelock

        qwen_lock_path = os.path.abspath(qwen_lock_path)
        os.makedirs(os.path.dirname(qwen_lock_path), exist_ok=True)
        original_file_lock = filelock.FileLock

        def _runtime_file_lock(lock_file, *args, **kwargs):
            if os.path.abspath(os.fspath(lock_file)) == _SHARED_QWEN_LOCK:
                lock_file = qwen_lock_path
            return original_file_lock(lock_file, *args, **kwargs)

        filelock.FileLock = _runtime_file_lock


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "process"], default=os.environ.get("MODE", "process"))
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=_env_int("TEST_NUM_SAMPLES", 10))
    parser.add_argument("--dedup-threshold", type=float, default=_env_float("DEDUP_THRESHOLD", 0.99))
    parser.add_argument("--shuffle-seed", type=int, default=_env_int("SHUFFLE_SEED", 42))
    parser.add_argument("--shard-id", type=int, default=_env_int("SLURM_ARRAY_TASK_ID", 0))
    parser.add_argument(
        "--total-shards",
        type=int,
        default=_env_int("TOTAL_SHARDS", _env_int("SLURM_ARRAY_TASK_COUNT", 1)),
    )
    args = parser.parse_args()
    _configure_runtime_paths()

    num_samples_env = os.environ.get("NUM_SAMPLES")
    num_samples = args.num_samples if args.num_samples is not None else int(num_samples_env or 250)

    if args.mode == "test":
        species_segmentation.test_config(
            INAT21_CONFIG,
            qwen_crop=True,
            qwen_isolate_mask=True,
            num_samples=args.test_samples,
            dedup_threshold=args.dedup_threshold,
            shuffle=True,
            shuffle_seed=args.shuffle_seed,
        )
        return

    species_segmentation.process_dataset(
        INAT21_CONFIG,
        qwen_isolate_mask=True,
        qwen_crop=True,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        num_samples=num_samples,
        dedup_threshold=args.dedup_threshold,
        shuffle=False,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()
