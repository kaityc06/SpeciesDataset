import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import IWILDCAM_CONFIG
from species_segmentation import process_dataset, test_config

shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
total_shards = int(os.environ.get("TOTAL_SHARDS", os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
num_samples_env = os.environ.get("NUM_SAMPLES")
num_samples = int(num_samples_env) if num_samples_env else None

"""test_config(
    IWILDCAM_CONFIG,
    qwen_crop=True,
    qwen_isolate_mask=True,
    num_samples=30,
    dedup_threshold=0.99,
    shuffle=True,
    categories={
        "panthera onca",
        "panthera pardus",
        "panthera leo",
        "acinonyx jubatus",
        "equus quagga",
        "giraffa camelopardalis",
        "loxodonta africana",
    }
)"""

process_dataset(
    IWILDCAM_CONFIG,
    qwen_isolate_mask=True,
    qwen_crop=True,
    shard_id=shard_id,
    total_shards=total_shards,
    num_samples=250, # per shard
    dedup_threshold=0.99,
)
os._exit(0)
