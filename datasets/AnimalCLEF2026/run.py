import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import ANIMALCLEF2026_CONFIG
from species_segmentation import process_dataset, test_config
from test_duplicate_detection import dedup_visual_demo

shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
total_shards = int(os.environ.get("TOTAL_SHARDS", os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
num_samples_env = os.environ.get("NUM_SAMPLES")
num_samples = int(num_samples_env) if num_samples_env else None

"""dedup_visual_demo(
    ANIMALCLEF2026_CONFIG,
    num_images=30,
    threshold=0.99,
)"""

test_config(
    ANIMALCLEF2026_CONFIG,
    qwen_crop=True,
    qwen_isolate_mask=True,
    num_samples=10,
    dedup_threshold=0.99,
    categories={"loggerhead turtle"},
)

"""
process_dataset(
    ANIMALCLEF2026_CONFIG,
    qwen_isolate_mask=True,
    qwen_crop=True,
    shard_id=shard_id,
    total_shards=total_shards,
    num_samples=250, # per shard
    dedup_threshold=0.99
)
"""

os._exit(0)
