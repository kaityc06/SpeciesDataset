#!/usr/bin/env python3
"""Validate that every AwA2 class has an explicit prompt mapping."""

import ast
import os
from typing import Dict, List


DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_PATH = os.path.join(DATASET_DIR, "data", "Animals_with_Attributes2", "classes.txt")
CONFIG_PATH = os.path.join(DATASET_DIR, "config.py")


def _read_classes() -> List[str]:
    classes = []
    with open(CLASSES_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                classes.append(parts[1])
    return classes


def _read_mapping() -> Dict[str, str]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        module = ast.parse(f.read(), filename=CONFIG_PATH)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AWA2_CLASS_MAPPING":
                    return ast.literal_eval(node.value)
    raise RuntimeError("AWA2_CLASS_MAPPING not found in config.py")


def main() -> None:
    classes = _read_classes()
    mapping = _read_mapping()
    class_set = set(classes)
    mapping_set = set(mapping)
    missing = sorted(class_set - mapping_set)
    extra = sorted(mapping_set - class_set)
    empty = sorted(key for key, value in mapping.items() if not value)

    if missing or extra or empty:
        if missing:
            print("missing:", ", ".join(missing))
        if extra:
            print("extra:", ", ".join(extra))
        if empty:
            print("empty:", ", ".join(empty))
        raise SystemExit(1)

    print(f"coverage: {len(classes)}/{len(classes)}")


if __name__ == "__main__":
    main()
