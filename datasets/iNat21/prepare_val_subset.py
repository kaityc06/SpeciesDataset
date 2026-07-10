#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


INAT21_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
SUPERCATEGORY_ORDER = [
    "Animalia",
    "Amphibians",
    "Arachnids",
    "Birds",
    "Fungi",
    "Insects",
    "Mammals",
    "Mollusks",
    "Plants",
    "Ray-finned Fishes",
    "Reptiles",
]


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _safe_join(root: str, relative_path: str) -> str:
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, relative_path))
    if os.path.commonpath([root_abs, target]) != root_abs:
        raise ValueError(f"Refusing to write outside {root_abs}: {relative_path}")
    return target


def _manifest_path(data_dir: str, split: str, target_size: int, seed: int, explicit: str = None) -> str:
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        return os.path.join(os.path.dirname(__file__), explicit)
    return os.path.join(data_dir, f"{split}_species_balanced_subset_{target_size}_seed{seed}.json")


def ensure_split_json(split: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, f"{split}.json")
    if os.path.exists(json_path):
        print(f"[inat21_subset] Using existing JSON: {json_path}")
        return json_path

    url = f"{INAT21_BASE}/{split}.json.tar.gz"
    tmp_path = json_path + ".tmp"
    print(f"[inat21_subset] Downloading {url}")
    with urllib.request.urlopen(url) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(tmp_path, json_path)
                print(f"[inat21_subset] Cached JSON: {json_path}")
                return json_path

    raise RuntimeError(f"No JSON file found in {url}")


def load_annotations(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _supercategory_order(groups: dict) -> list:
    known = [name for name in SUPERCATEGORY_ORDER if name in groups]
    extra = sorted((name for name in groups if name not in SUPERCATEGORY_ORDER), key=str.casefold)
    return known + extra


def select_balanced_subset(data: dict, target_size: int, seed: int) -> Tuple[List[dict], Dict, List[str]]:
    cat_by_id = {cat["id"]: cat for cat in data["categories"]}
    category_id_by_image = {}
    for ann in data["annotations"]:
        category_id_by_image.setdefault(ann["image_id"], ann["category_id"])

    groups = defaultdict(lambda: defaultdict(list))
    missing_category = 0
    for img in data["images"]:
        cat = cat_by_id.get(category_id_by_image.get(img["id"]))
        if not cat or not cat.get("supercategory"):
            missing_category += 1
            continue

        record = dict(img)
        record["file_name"] = _norm_path(record["file_name"])
        record["category_id"] = cat.get("id")
        record["category_name"] = cat.get("name")
        record["supercategory"] = cat.get("supercategory")
        species_key = cat.get("name") or str(cat.get("id"))
        record["species_key"] = species_key
        groups[record["supercategory"]][species_key].append(record)

    if missing_category:
        print(f"[inat21_subset] Skipped {missing_category} image(s) without category metadata")
    if not groups:
        raise RuntimeError("No annotated supercategory groups found")

    group_order = _supercategory_order(groups)
    total_available = sum(
        len(images)
        for species_groups in groups.values()
        for images in species_groups.values()
    )
    if target_size > total_available:
        raise ValueError(f"Requested {target_size} images, but only {total_available} are available")

    for name in group_order:
        for species_key in groups[name]:
            groups[name][species_key].sort(key=lambda img: (img["file_name"], str(img.get("id", ""))))

    base_quota = target_size // len(group_order)
    remainder = target_size % len(group_order)
    selected_by_group = {}
    desired_counts = {}
    group_stats = {}

    for idx, name in enumerate(group_order):
        desired = base_quota + (1 if idx < remainder else 0)
        desired_counts[name] = desired
        species_groups = groups[name]
        species_keys = sorted(species_groups)

        if len(species_keys) > desired:
            shuffled_species = list(species_keys)
            random.Random(f"{seed}:{name}:species").shuffle(shuffled_species)
            chosen_species = sorted(shuffled_species[:desired])
            per_species_take = {species_key: 1 for species_key in chosen_species}
            mode = "sample_species_one_each"
        else:
            chosen_species = species_keys
            per_species_base = desired // len(chosen_species)
            per_species_remainder = desired % len(chosen_species)
            per_species_take = {
                species_key: per_species_base
                for species_key in chosen_species
            }
            remainder_order = list(chosen_species)
            random.Random(f"{seed}:{name}:remainder").shuffle(remainder_order)
            for species_key in remainder_order[:per_species_remainder]:
                per_species_take[species_key] += 1
            mode = "all_species_even_images"

        selected = []
        for species_key in chosen_species:
            images = list(species_groups[species_key])
            random.Random(f"{seed}:{name}:{species_key}:images").shuffle(images)
            take = per_species_take[species_key]
            if len(images) < take:
                raise RuntimeError(
                    f"{name} species {species_key} has {len(images)} image(s), needs {take}"
                )
            selected.extend(images[:take])

        if len(selected) != desired:
            raise RuntimeError(f"Selected {len(selected)} images for {name}, expected {desired}")

        allocation_hist = Counter(per_species_take.values())
        group_stats[name] = {
            "target_images": desired,
            "source_species": len(species_keys),
            "source_images": sum(len(species_groups[species_key]) for species_key in species_keys),
            "sampling_mode": mode,
            "selected_species": len(chosen_species),
            "species_image_allocation_histogram": {
                str(k): allocation_hist[k]
                for k in sorted(allocation_hist)
            },
        }
        selected_by_group[name] = selected

    selected = []
    for name in group_order:
        selected.extend(selected_by_group[name])

    file_names = [img["file_name"] for img in selected]
    if len(file_names) != len(set(file_names)):
        raise RuntimeError("Selected subset contains duplicate file names")
    if len(selected) != target_size:
        raise RuntimeError(f"Selected {len(selected)} images, expected {target_size}")

    counts = {name: len(selected_by_group[name]) for name in group_order}
    selected_species_total = sum(group_stats[name]["selected_species"] for name in group_order)
    species_allocation_histogram = Counter()
    for name in group_order:
        for per_species_images, species_count in group_stats[name]["species_image_allocation_histogram"].items():
            species_allocation_histogram[per_species_images] += species_count
    return selected, {
        "desired": desired_counts,
        "selected": counts,
        "sampling_method": "species_balanced_within_supercategory",
        "selected_species_total": selected_species_total,
        "species_allocation_histogram": dict(sorted(species_allocation_histogram.items(), key=lambda kv: int(kv[0]))),
        "supercategory_stats": group_stats,
    }, group_order


def write_manifest(path: str, split: str, target_size: int, seed: int, json_path: str, selected: list, counts: dict, group_order: list, data: dict) -> None:
    manifest = {
        "dataset": "iNat21",
        "split": split,
        "target_size": target_size,
        "seed": seed,
        "source_json": os.path.basename(json_path),
        "total_source_images": len(data.get("images", [])),
        "sampling_method": counts.get("sampling_method"),
        "supercategory_order": group_order,
        "supercategory_counts": counts["selected"],
        "desired_supercategory_counts": counts["desired"],
        "selected_species_total": counts.get("selected_species_total"),
        "species_allocation_histogram": counts.get("species_allocation_histogram"),
        "supercategory_sampling_stats": counts.get("supercategory_stats"),
        "images": selected,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
    print(f"[inat21_subset] Wrote manifest: {path}")
    print(f"[inat21_subset] Selected counts: {dict(Counter(img['supercategory'] for img in selected))}")
    print(f"[inat21_subset] Selected species: {counts.get('selected_species_total')}")
    print(f"[inat21_subset] Species allocation histogram: {counts.get('species_allocation_histogram')}")


def extract_selected_images(split: str, data_dir: str, selected: list, force: bool = False) -> None:
    selected_names = {_norm_path(img["file_name"]) for img in selected}
    existing = {
        name for name in selected_names
        if os.path.exists(os.path.join(data_dir, name)) and os.path.getsize(os.path.join(data_dir, name)) > 0
    }
    pending = set(selected_names if force else selected_names - existing)
    if not pending:
        print(f"[inat21_subset] All {len(selected_names)} selected images already exist under {data_dir}")
        return

    url = f"{INAT21_BASE}/{split}.tar.gz"
    extracted = set()
    print(f"[inat21_subset] Streaming {url}")
    print(f"[inat21_subset] Extracting {len(pending)} selected image(s) into {data_dir}")
    with urllib.request.urlopen(url) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not pending:
                    break
                if not member.isfile():
                    continue
                member_name = _norm_path(member.name)
                if member_name not in pending:
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue

                dest = _safe_join(data_dir, member_name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp_dest = dest + ".tmp"
                with open(tmp_dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(tmp_dest, dest)
                pending.remove(member_name)
                extracted.add(member_name)
                if len(extracted) % 250 == 0:
                    print(f"[inat21_subset] Extracted {len(extracted)} image(s)")

    if pending:
        examples = sorted(pending)[:10]
        raise RuntimeError(f"Missing {len(pending)} selected image(s) in tar stream; examples: {examples}")

    print(
        f"[inat21_subset] Done. existing={len(existing)} "
        f"extracted={len(extracted)} total={len(selected_names)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced iNat21 validation subset.")
    parser.add_argument("--split", default=os.environ.get("SPLIT", "val"), choices=["val"])
    parser.add_argument("--target-size", type=int, default=int(os.environ.get("INAT21_VAL_SUBSET_SIZE", 2500)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("INAT21_VAL_SUBSET_SEED", 42)))
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("INAT21_DATA_DIR", os.path.join(os.path.dirname(__file__), "data")),
    )
    parser.add_argument("--manifest", default=os.environ.get("INAT21_VAL_SUBSET_MANIFEST"))
    parser.add_argument("--no-extract", action="store_true", help="Only write the manifest; do not extract images.")
    parser.add_argument("--force-extract", action="store_true", help="Overwrite selected images even if they already exist.")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    manifest_path = _manifest_path(data_dir, args.split, args.target_size, args.seed, args.manifest)
    json_path = ensure_split_json(args.split, data_dir)
    data = load_annotations(json_path)
    selected, counts, group_order = select_balanced_subset(data, args.target_size, args.seed)
    write_manifest(manifest_path, args.split, args.target_size, args.seed, json_path, selected, counts, group_order, data)

    if args.no_extract:
        print("[inat21_subset] Skipping image extraction because --no-extract was set")
        return

    extract_selected_images(args.split, data_dir, selected, force=args.force_extract)


if __name__ == "__main__":
    main()
