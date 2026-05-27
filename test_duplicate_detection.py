"""Duplicate detection visual report — called from test_config() in species_segmentation.py.

Loads images from the given DatasetConfig, injects known duplicates, runs them through
DuplicateDetector, and writes a self-contained HTML report showing which images were flagged.
"""

import base64
import io
import os

import numpy as np
from PIL import Image, ImageFilter

from duplicate_detection import DuplicateDetector


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Duplicate Detection Report</title>
<style>
  body  { font-family: sans-serif; background: #1a1a1a; color: #eee;
          margin: 0; padding: 20px; }
  h1   { margin-bottom: 4px; }
  .stats { background: #2a2a2a; border-radius: 6px; padding: 12px 16px;
           margin-bottom: 20px; font-size: 14px; line-height: 1.7; }
  .grid { display: flex; flex-wrap: wrap; gap: 14px; }
  .card { background: #2a2a2a; border-radius: 8px; padding: 10px;
          width: 320px; box-sizing: border-box; }
  .card img { width: 100%; border-radius: 4px; display: block; }
  .badge { display: inline-block; border-radius: 4px; padding: 2px 8px;
           font-size: 12px; font-weight: bold; margin-top: 6px; }
  .unique         { background: #1e7e34; color: #fff; }
  .duplicate      { background: #b71c1c; color: #fff; }
  .caught         { background: #e65100; color: #fff; margin-left: 4px; }
  .missed         { background: #f9a825; color: #000; margin-left: 4px; }
  .false-positive { background: #f9a825; color: #000; margin-left: 4px; }
  .sim { font-size: 11px; color: #ccc; margin-top: 3px; }
</style>
</head><body>
"""

_HTML_FOOT = "</div></body></html>\n"


def _pil_to_b64(img: Image.Image, max_side: int = 300) -> str:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    thumb = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _write_html_report(results: list, output_path: str, threshold: float) -> None:
    n_unique = sum(1 for r in results if not r["is_duplicate"])
    n_dup    = sum(1 for r in results if r["is_duplicate"])
    n_inject = sum(1 for r in results if r["injected"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_HTML_HEAD)
        f.write("<h1>Duplicate Detection Report — iNat21</h1>\n")
        f.write(f'<div class="stats">'
                f"Threshold: <b>{threshold}</b> &nbsp;|&nbsp; "
                f"Images processed: <b>{len(results)}</b> &nbsp;|&nbsp; "
                f"Unique: <b>{n_unique}</b> &nbsp;|&nbsp; "
                f"Duplicates caught: <b>{n_dup}</b> "
                f"(of which <b>{n_inject}</b> were intentionally injected)"
                f"</div>\n")
        f.write('<div class="grid">\n')

        for r in results:
            b64 = _pil_to_b64(r["image"])
            badge_cls = "duplicate" if r["is_duplicate"] else "unique"
            badge_txt = "DUPLICATE" if r["is_duplicate"] else "UNIQUE"
            if r["injected"] and r["is_duplicate"]:
                inject_badge = f'<span class="badge caught">{r["injected"]} — CAUGHT</span>'
            elif r["injected"] and not r["is_duplicate"]:
                inject_badge = f'<span class="badge caught">{r["injected"]}</span><span class="badge missed">MISSED</span>'
            elif not r["injected"] and r["is_duplicate"]:
                inject_badge = '<span class="badge false-positive">FALSE POSITIVE</span>'
            else:
                inject_badge = ""
            if r["is_duplicate"] or r["injected"]:
                sim_line = f'<div class="sim">cosine similarity: {r["similarity"]:.4f}</div>'
            else:
                sim_line = ""

            f.write(
                f'<div class="card">'
                f'<img src="{b64}">'
                f'<span class="badge {badge_cls}">{badge_txt}</span>{inject_badge}'
                f'{sim_line}'
                f'</div>\n'
            )

        f.write(_HTML_FOOT)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def dedup_visual_demo(config, num_images: int = 20, threshold: float = 0.99):
    from species_segmentation import _dataset_output_dir, _make_stream
    out_subdir = _dataset_output_dir(config)
    output = os.path.join(out_subdir, "dedup_report.html")

    print(f"[dedup] Loading {num_images} images from {config.name} …")
    real_images = []
    for sample in _make_stream(config, num_images):
        img = config.get_image(sample)
        if img is None:
            continue
        real_images.append(img)
        print(f"  loaded {len(real_images)}/{num_images}", end="\r")
    print(f"\nLoaded {len(real_images)} images.")

    if len(real_images) < 3:
        print("Not enough images loaded.")
        return

    # Build stream: real images + injected duplicates inserted 2 positions after their originals.
    # Injections must be listed in ascending order of src_idx so the cumulative offset stays correct.
    N = len(real_images)

    def _bright(img):
        from PIL import ImageEnhance
        return ImageEnhance.Brightness(img).enhance(1.5)

    def _crop(img):
        w, h = img.size
        return img.crop((w // 8, h // 8, 7 * w // 8, 7 * h // 8)).resize((w, h))

    injections = [
        (0,        "EXACT COPY",   lambda img: img.copy()),
        (N // 5,   "BRIGHTENED",   _bright),
        (N // 3,   "BLURRED",      lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
        (N // 2,   "CROPPED",      _crop),
        (2*N // 3, "RESIZED 75%",  lambda img: img.resize((int(img.width * 0.75), int(img.height * 0.75)))),
    ]

    # Each injection shifts all subsequent source indices by 1, so insert_at = src_idx + k + gap.
    gap = 2
    stream_items = [(img, None) for img in real_images]
    for k, (src_idx, tag, transform) in enumerate(injections):
        insert_at = src_idx + k + gap
        stream_items.insert(insert_at, (transform(real_images[src_idx]), tag))

    detector = DuplicateDetector(threshold=threshold)
    results = []
    for img, injected in stream_items:
        is_dup, sim = detector.check_and_add(img)
        results.append({"image": img, "injected": injected,
                        "is_duplicate": is_dup, "similarity": sim})
        status = "DUPLICATE" if is_dup else "unique"
        tag = f" [{injected}]" if injected else ""
        print(f"  {status:<10}  sim={sim:.4f}{tag}")

    os.makedirs(out_subdir, exist_ok=True)
    if os.path.exists(output):
        print(f"[dedup] Removing existing file: {output}")
        os.remove(output)
    _write_html_report(results, output, threshold)

    n_injected = sum(1 for r in results if r["injected"])
    caught = sum(1 for r in results if r["is_duplicate"] and r["injected"])
    n_dup = sum(1 for r in results if r["is_duplicate"])
    print(f"\nReport written → {output}")
    print(f"Summary: {len(results)} images processed, {n_dup} duplicates flagged, "
          f"{caught}/{n_injected} injected duplicates caught")


