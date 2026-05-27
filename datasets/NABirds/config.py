import os
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


NABIRDS_DIR = "/mmfs1/gscratch/krishna/kaityc/nabirds"
NABIRDS_URL = (
    "https://www.dropbox.com/scl/fi/yas70u9uzkeyzrmrfwcru/nabirds.tar.gz"
    "?rlkey=vh0uduhckom5jyp73igjugqtr&st=khk3jsbq&dl=1"
)


def _ensure_nabirds():
    if os.path.exists(os.path.join(NABIRDS_DIR, "images.txt")):
        return
    import urllib.request
    import tarfile

    os.makedirs(NABIRDS_DIR, exist_ok=True)
    tar_path = os.path.join(NABIRDS_DIR, "nabirds.tar.gz")
    print(f"Downloading NABirds to {tar_path} ...")
    urllib.request.urlretrieve(NABIRDS_URL, tar_path)
    print("Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(NABIRDS_DIR)
    os.remove(tar_path)
    nested = os.path.join(NABIRDS_DIR, "nabirds")
    if os.path.isdir(nested):
        import shutil
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), NABIRDS_DIR)
        os.rmdir(nested)
    print(f"NABirds ready at {NABIRDS_DIR}")


def _nabirds_stream(num_samples: int, categories=None) -> Iterable:
    _ensure_nabirds()
    if categories is not None and "Aves" not in categories:
        return

    def _read_table(filename):
        rows = {}
        with open(os.path.join(NABIRDS_DIR, filename)) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                rows[parts[0]] = parts[1]
        return rows

    images      = _read_table("images.txt")
    splits      = _read_table("train_test_split.txt")
    labels      = _read_table("image_class_labels.txt")
    class_names = _read_table("classes.txt")

    bbox_rows = {}
    with open(os.path.join(NABIRDS_DIR, "bounding_boxes.txt")) as f:
        for line in f:
            parts = line.strip().split()
            bbox_rows[parts[0]] = (float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4]))

    yielded = 0
    for img_id, rel_path in images.items():
        if yielded >= num_samples:
            break
        if splits.get(img_id, "0") != "1":
            continue

        img_path = os.path.join(NABIRDS_DIR, "images", rel_path)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
            continue

        x, y, bw, bh = bbox_rows[img_id]
        iw, ih = pil_img.size
        bbox_norm = [x / iw, y / ih, (x + bw) / iw, (y + bh) / ih]

        yield {
            "image":   pil_img,
            "species": class_names.get(labels[img_id], "unknown"),
            "bbox":    bbox_norm,
        }
        yielded += 1


NABIRDS_CONFIG = DatasetConfig(
    name="NABirds",
    load_fn=_nabirds_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: "Aves",
    get_bboxes=lambda s: [s["bbox"]],
    class_mapping={"Aves": "bird"},
)
