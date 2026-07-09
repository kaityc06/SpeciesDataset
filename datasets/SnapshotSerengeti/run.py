import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from config import CONFIG, DEFAULT_PROCESS_SAMPLES, DEFAULT_TOTAL_SHARDS
from species_segmentation import process_dataset, test_config
from test_duplicate_detection import dedup_visual_demo


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
