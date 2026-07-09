import os
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


_INAT21_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
_INAT21_DATA_DIR = os.environ.get(
    "INAT21_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)

_VALID_SPLITS = {"train", "train_mini", "val"}


def _env_int(name: str, default=None):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _split_paths(split: str):
    if split not in _VALID_SPLITS:
        raise ValueError(f"SPLIT must be one of {_VALID_SPLITS}, got {split!r}")
    json_path = os.path.join(_INAT21_DATA_DIR, f"{split}.json")
    images_dir = os.path.join(_INAT21_DATA_DIR, split)
    return json_path, images_dir


def _ensure_inat21_json(split: str):
    json_path, _ = _split_paths(split)
    if os.path.exists(json_path):
        return
    import tarfile as _tf
    import urllib.request as _ur
    os.makedirs(_INAT21_DATA_DIR, exist_ok=True)
    url = f"{_INAT21_BASE}/{split}.json.tar.gz"
    print(f"Downloading iNat21 {split} annotation JSON ...")
    with _ur.urlopen(url) as resp:
        with _tf.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    with open(json_path, "wb") as out:
                        out.write(f.read())
                    break
    print(f"Cached to {json_path}")


def _val_subset_target(split: str):
    if split != "val":
        return None
    target = _env_int("INAT21_VAL_SUBSET_SIZE")
    if target is None or target <= 0:
        return None
    return target


def _val_subset_manifest_path(split: str, target: int):
    explicit = os.environ.get("INAT21_VAL_SUBSET_MANIFEST")
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        return os.path.join(os.path.dirname(__file__), explicit)

    seed = _env_int("INAT21_VAL_SUBSET_SEED", 42)
    return os.path.join(_INAT21_DATA_DIR, f"{split}_subset_{target}_seed{seed}.json")


def _load_subset_images(split: str):
    target = _val_subset_target(split)
    if target is None:
        return None, None

    import json

    manifest_path = _val_subset_manifest_path(split, target)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"iNat21 val subset manifest not found: {manifest_path}\n"
            "Run prepare_val_subset.py first, or unset INAT21_VAL_SUBSET_SIZE."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    if manifest.get("split") != split:
        raise ValueError(f"{manifest_path} is for split {manifest.get('split')!r}, not {split!r}")
    if len(images) != target:
        raise ValueError(f"{manifest_path} has {len(images)} images, expected {target}")
    return images, manifest_path


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _category_for_image(img_meta: dict, ann_by_img: dict):
    cat = ann_by_img.get(img_meta.get("id"))
    if cat is not None:
        return cat
    return {
        "id": img_meta.get("category_id"),
        "name": img_meta.get("category_name") or img_meta.get("species"),
        "supercategory": img_meta.get("supercategory"),
    }


def _sample_from_image(pil_img: Image.Image, img_meta: dict, cat: dict, split: str, subset_manifest: str = None):
    file_name = _norm_path(img_meta["file_name"])
    sample = {
        "image": pil_img,
        "split": split,
        "image_id": img_meta.get("id"),
        "file_name": file_name,
        "supercategory": cat.get("supercategory"),
        "species": cat.get("name"),
        "category_id": cat.get("id"),
        "category": cat.get("name"),
        "file_id": os.path.basename(file_name).rsplit(".", 1)[0],
    }
    if subset_manifest is not None:
        sample["subset_manifest"] = os.path.basename(subset_manifest)
    return sample


def _make_inat21_stream(split: str):
    def _inat21_stream(num_samples: int, shard_id: int = 0, total_shards: int = 1, categories=None, shuffle: bool = False, shuffle_seed: int = 42) -> Iterable:
        import json
        import random
        import tarfile
        import urllib.request

        _ensure_inat21_json(split)
        json_path, images_dir = _split_paths(split)

        with open(json_path) as f:
            data = json.load(f)

        cat_map = {c["id"]: c for c in data["categories"]}
        ann_by_img = {a["image_id"]: cat_map[a["category_id"]] for a in data["annotations"]}

        subset_images, subset_manifest = _load_subset_images(split)
        all_images = subset_images if subset_images is not None else data["images"]
        if shuffle:
            rng = random.Random(shuffle_seed)
            all_images = list(all_images)
            rng.shuffle(all_images)

        if os.path.isdir(images_dir):
            shard_imgs = all_images[shard_id::total_shards]
            shard_imgs = shard_imgs[:num_samples]
            for img_meta in shard_imgs:
                file_name = _norm_path(img_meta["file_name"])
                local_path = os.path.join(_INAT21_DATA_DIR, file_name)
                cat = _category_for_image(img_meta, ann_by_img)
                if categories is not None and cat.get("supercategory") not in categories:
                    continue
                try:
                    pil_img = Image.open(local_path).convert("RGB")
                except Exception as exc:
                    print(f"  Skipping {local_path}: {exc}")
                    continue
                yield _sample_from_image(pil_img, img_meta, cat, split, subset_manifest)
        else:
            if total_shards > 1:
                raise RuntimeError(
                    f"Multi-shard mode requires pre-extracted images at {images_dir}.\n"
                    "Run prepare_val_subset.py for the val subset, or extract the split tarball first."
                )
            path_to_img = {}
            for img in all_images:
                path_to_img[_norm_path(img["file_name"])] = img

            # When shuffle=True, all_images is already shuffled; pre-select the
            # target file IDs so the tar stream is filtered rather than taken in order.
            if shuffle:
                candidate_imgs = all_images
                if categories is not None:
                    candidate_imgs = [
                        img for img in candidate_imgs
                        if _category_for_image(img, ann_by_img).get("supercategory") in categories
                    ]
                target_paths = {
                    _norm_path(img["file_name"])
                    for img in candidate_imgs[:num_samples]
                }
            else:
                target_paths = None

            url = f"{_INAT21_BASE}/{split}.tar.gz"
            yielded = 0
            with urllib.request.urlopen(url) as resp:
                with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                    for member in tar:
                        if yielded >= num_samples:
                            break
                        if not (member.isfile() and member.name.lower().endswith(".jpg")):
                            continue
                        member_name = _norm_path(member.name)
                        if target_paths is not None and member_name not in target_paths:
                            continue
                        img_meta = path_to_img.get(member_name)
                        if img_meta is None:
                            continue
                        cat = _category_for_image(img_meta, ann_by_img)
                        if categories is not None and cat.get("supercategory") not in categories:
                            continue
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        try:
                            pil_img = Image.open(BytesIO(f.read())).convert("RGB")
                        except Exception:
                            continue
                        yield _sample_from_image(pil_img, img_meta, cat, split, subset_manifest)
                        yielded += 1

    return _inat21_stream


_inat21_split = os.environ.get("SPLIT", "train")

INAT21_CONFIG = DatasetConfig(
    name=f"iNat21",
    load_fn=_make_inat21_stream(_inat21_split),
    get_image=lambda s: s["image"].convert("RGB") if s.get("image") else None,
    get_class=lambda s: s.get("supercategory"),
    class_mapping={
        "Animalia": ["animal"],
        "Amphibians": ["frog", "salamander"],
        "Arachnids": ["spider"],
        "Birds": ["bird"],
        "Fungi": ["fungus"],
        "Insects": ["insect"],
        "Mammals": ["mammal"],
        "Mollusks": ["shellfish", "snail", "slug", "octopus"],
        "Plants": ["plant"],
        "Ray-finned Fishes": ["fish"],
        "Reptiles": ["reptile", "turtle", "lizard", "crocodile"],
    },
)
