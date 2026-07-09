import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from config import INAT21_CONFIG
from species_segmentation import process_dataset, test_config


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

    num_samples_env = os.environ.get("NUM_SAMPLES")
    num_samples = args.num_samples if args.num_samples is not None else int(num_samples_env or 250)

    if args.mode == "test":
        test_config(
            INAT21_CONFIG,
            qwen_crop=True,
            qwen_isolate_mask=True,
            num_samples=args.test_samples,
            dedup_threshold=args.dedup_threshold,
            shuffle=True,
            shuffle_seed=args.shuffle_seed,
        )
        return

    process_dataset(
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
