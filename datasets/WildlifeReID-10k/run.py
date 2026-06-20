import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import WILDLIFEREID_CONFIG
from species_segmentation import process_dataset, test_config
from test_duplicate_detection import dedup_visual_demo

shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
total_shards = int(os.environ.get("TOTAL_SHARDS", os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
num_samples_env = os.environ.get("NUM_SAMPLES")
num_samples = int(num_samples_env) if num_samples_env else None

"""dedup_visual_demo(
    WILDLIFEREID_CONFIG,
    num_images=30,
    threshold=0.99,
)"""

"""test_config(
    WILDLIFEREID_CONFIG,
    qwen_crop=True,
    qwen_isolate_mask=True,
    num_samples=30,
    dedup_threshold=0.99,
    shuffle=True,
    categories={
        "chimpanzee",
        "whaleshark",
        "sea star",
        "leopard"
    }
)"""

process_dataset(
    WILDLIFEREID_CONFIG,
    qwen_isolate_mask=True,
    qwen_crop=True,
    shard_id=shard_id,
    total_shards=total_shards,
    dedup_threshold=0.99,
)

os._exit(0)