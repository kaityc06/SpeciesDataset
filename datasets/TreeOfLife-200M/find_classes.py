"""
Scan all 200M TreeOfLife samples to collect every unique taxonomic class.
Reads only metadata fields — never opens image files.
Results are written to outputs/all_classes.txt after every checkpoint_interval samples.
"""

import os
import sys
import time
from collections import Counter
from datasets import load_dataset

CHECKPOINT_INTERVAL = 1_000_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_classes.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_results(class_counts: Counter, n_total: int, n_missing: int):
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"# Scanned {n_total:,} samples  ({n_missing:,} with no class)\n")
        f.write(f"# {len(class_counts)} unique classes\n\n")
        for cls, count in class_counts.most_common():
            f.write(f"{cls}\t{count}\n")


def main():
    print("Loading TreeOfLife-200M in streaming mode (metadata only)...")
    ds = load_dataset(
        "imageomics/TreeOfLife-200M",
        split="train",
        streaming=True,
    )

    class_counts: Counter = Counter()
    n_total = 0
    n_missing = 0
    t0 = time.time()

    for sample in ds:
        cls = sample.get("class")
        if cls:
            class_counts[cls] += 1
        else:
            n_missing += 1
        n_total += 1

        if n_total % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = n_total / elapsed
            print(
                f"  {n_total:>12,} samples  |  {len(class_counts)} classes  |"
                f"  {rate:,.0f} samples/s  |  elapsed {elapsed/60:.1f} min",
                flush=True,
            )
            save_results(class_counts, n_total, n_missing)

    save_results(class_counts, n_total, n_missing)
    elapsed = time.time() - t0
    print(f"\nDone. {n_total:,} samples scanned in {elapsed/60:.1f} min.")
    print(f"Found {len(class_counts)} unique classes. Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
