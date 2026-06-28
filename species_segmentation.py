import base64
import io
import os
import sys
import time
import torch
import sam3.sam3
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
from sam3.sam3.agent.helpers.rle import robust_rle_encode
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Iterable
from PIL import Image


# ---------------------------------------------------------------------------
# Qwen3.5 mask quality checker
# ---------------------------------------------------------------------------

QWEN_MODEL = "Qwen/Qwen3.5-4B"
QWEN_QUALITY_PROMPT = (
    "You are given two images:\n"
    "- Image 1: the original photograph.\n"
    "- Image 2: the same photograph with a segmentation mask overlaid in semi-transparent red.\n\n"
    "The mask was generated using the text prompt: \"{text_prompt}\".\n\n"
    "Task:\n"
    "Determine whether the red mask accurately covers the object(s) described by the text prompt.\n\n"
    "Evaluation criteria:\n"
    "- The mask should cover most (≥95%) of the target object(s).\n"
    "- The mask should not include large background regions (≤10% excess area).\n"
    "- If the object is partially occluded or ambiguous, judge based on visible regions only.\n\n"
    "Instructions:\n"
    "- Carefully compare Image 1 and Image 2.\n"
    "- Explain any missing regions or extra coverage.\n"
    "- Be conservative: if unsure, answer \"No\".\n\n"
    "Output format (strictly follow):\n"
    "Analysis: <1-3 sentences describing coverage and any errors>\n"
    "Verdict: Yes or No\n"
)


QWEN_ISOLATION_PROMPT = (
    "You are given a single image where only a segmented region is visible.\n"
    "All other pixels have been blacked out and contain no useful information.\n\n"
    "Determine whether the visible (non-black) region clearly contains: \"{text_prompt}\".\n\n"
    "Guidelines:\n"
    "- Treat black regions as if they do not exist.\n"
    "- The object must be visually present and recognizable.\n"
    "- Valid objects must show texture, consistent boundaries, and meaningful visual features (not just shape alone).\n"
    "- Reject the object if it is heavily pixelated, noisy, blurry, or artifact-heavy.\n"
    "- Do not infer or guess based on context alone.\n"
    "- If the object is ambiguous or unclear, answer \"No\".\n"
    "- Be conservative in your judgment.\n\n"
    "Output format (strictly follow):\n"
    "Analysis: <1–2 sentences explaining your reasoning>\n"
    "Verdict: Yes or No\n"
)


_qwen_checker_singleton: "QwenMaskChecker | None" = None


def get_qwen_checker(device: str = "cuda") -> "QwenMaskChecker":
    global _qwen_checker_singleton
    if _qwen_checker_singleton is None:
        _qwen_checker_singleton = QwenMaskChecker(device=device)
    return _qwen_checker_singleton


class QwenMaskChecker:
    """Lazy-loads Qwen3.5 and evaluates mask quality for a masked image."""

    def __init__(self, model_name: str = QWEN_MODEL, device: str = "cuda"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        from filelock import FileLock
        lock_path = "/mmfs1/gscratch/krishna/kaityc/qwen_hf_download.lock"
        with FileLock(lock_path, timeout=600):
            print(f"Loading Qwen3.5 model: {self._model_name} …")
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self._model_name, device_map="auto"
            ).eval()
        print("Qwen3.5 ready.")

    def is_quality(self, original_pil: Image.Image, masked_pil: "Image.Image | None", text_prompt: str = "species subject") -> tuple[bool, str]:
        """Return (is_high_quality, explanation) from Qwen.

        When masked_pil is None, operates in isolation mode: original_pil should
        be the mask-isolated image (subject pixels at original colour, background
        black) and a single-image prompt is used.
        """
        self._load()
        if masked_pil is None:
            prompt = QWEN_ISOLATION_PROMPT.format(text_prompt=text_prompt)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            prompt = QWEN_QUALITY_PROMPT.format(text_prompt=text_prompt)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_pil},
                        {"type": "image", "image": masked_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False, # TODO: can change
        )

        # Move all tensors to the model device, but keep image_grid_thw on CPU.
        # Qwen only uses it for .prod(-1).tolist() (a Python list), never in a
        # CUDA kernel directly. Moving it to CUDA triggers an integer reduction
        # bug.
        image_grid_thw = inputs.pop("image_grid_thw")
        inputs = inputs.to(self._model.device)
        inputs["image_grid_thw"] = image_grid_thw

        # Disable the global SAM3 bfloat16 autocast only for the forward pass.
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
            out_ids = self._model.generate(**inputs, max_new_tokens=4096)

        generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        answer = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        lower = answer.lower()
        is_valid = "verdict: yes" in lower
        return is_valid, answer


def render_masks_to_pil(pil_image: Image.Image, tagged_masks) -> Image.Image:
    """Composite mask overlays onto the image and return as a PIL Image."""
    img_w, img_h = pil_image.size
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pil_image)
    for i, (mask, score, size_tag) in enumerate(tagged_masks):
        mask_np = mask.cpu().numpy()
        alpha = 0.25 if size_tag != "valid" else 0.5
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[..., 0] = 1.0  # red
        colored_mask[..., 3] = mask_np * alpha
        ax.imshow(colored_mask)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def crop_for_verification(
    pil_image: Image.Image,
    mask,
    bboxes=None,
    padding: float = 0.10,
):
    """Crop an image and mask to the region of interest for Qwen verification.

    If *bboxes* are provided, the first box (normalised [x0, y0, x1, y1]) is
    used as the crop region.  Otherwise the bounding box is derived from the
    non-zero pixels of *mask*.  In both cases *padding* (fraction of box size)
    is added on each side and clamped to the image boundary.

    Returns ``(cropped_pil, cropped_mask_tensor)``.
    """
    img_w, img_h = pil_image.size
    mask_np = mask.cpu().numpy()

    if bboxes:
        x0, y0, x1, y1 = bboxes[0]
        px0 = int(x0 * img_w)
        py0 = int(y0 * img_h)
        px1 = int(x1 * img_w)
        py1 = int(y1 * img_h)
    else:
        ys, xs = np.where(mask_np)
        if len(ys) == 0:
            return pil_image, mask
        px0, py0 = int(xs.min()), int(ys.min())
        px1, py1 = int(xs.max()), int(ys.max())

    pad_x = int((px1 - px0) * padding)
    pad_y = int((py1 - py0) * padding)
    px0 = max(0, px0 - pad_x)
    py0 = max(0, py0 - pad_y)
    px1 = min(img_w, px1 + pad_x)
    py1 = min(img_h, py1 + pad_y)

    cropped_pil = pil_image.crop((px0, py0, px1, py1))
    cropped_mask = torch.from_numpy(mask_np[py0:py1, px0:px1])
    return cropped_pil, cropped_mask


def render_mask_isolation_pil(pil_image: Image.Image, mask) -> Image.Image:
    """Return the image with all non-mask pixels blacked out."""
    img_np = np.array(pil_image)
    mask_bool = mask.cpu().numpy().astype(bool)
    result = np.zeros_like(img_np)
    result[mask_bool] = img_np[mask_bool]
    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# Parquet export helpers
# ---------------------------------------------------------------------------

def mask_to_rle(mask_tensor) -> dict:
    """Encode a 2-D mask tensor to a compressed COCO RLE dict with string counts."""
    mask_bool = mask_tensor.bool()
    if mask_bool.ndim == 2:
        mask_bool = mask_bool.unsqueeze(0)
    return robust_rle_encode(mask_bool)[0]


def _sample_metadata(sample: dict) -> dict:
    """Extract all serializable (non-image, non-tensor) fields from a sample dict."""
    import json
    result = {}
    for k, v in sample.items():
        if isinstance(v, Image.Image):
            continue
        if isinstance(v, torch.Tensor):
            continue
        if isinstance(v, (list, dict)):
            result[k] = json.dumps(v)
        else:
            result[k] = v
    return result


def save_masks_to_parquet(records: list, output_path: str) -> None:
    """Write accepted mask records to a Parquet file.

    Schema per row:
      dataset, text_prompt                        — provenance
      mask_rle_counts, mask_rle_height/width      — compressed COCO RLE
      mask_score                                  — mask quality
      image_width, image_height                   — original image dimensions
      <all other non-image sample fields>         — dataset-specific metadata

    Only samples with at least one accepted mask are included.
    Samples with multiple accepted masks produce one row per mask.
    """
    import pandas as pd
    if not records:
        print(f"No mask records to save → {output_path}")
        return
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} mask records → {output_path}")


# ---------------------------------------------------------------------------
# HTML report writer
# ---------------------------------------------------------------------------

class HtmlReport:
    """Context manager that writes results as an HTML file with embedded images."""

    _HEADER = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Segmentation Results</title>
<style>
  body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 16px; }}
  .card {{ background: #2a2a2a; border-radius: 8px; margin: 12px 0; padding: 12px; }}
  .card h3 {{ margin: 0 0 8px; font-size: 14px; word-break: break-word; }}
  .card img {{ max-width: 100%; border-radius: 4px; }}
</style>
</head><body>
<h1>{title}</h1>
"""
    _FOOTER = "</body></html>\n"

    def __init__(self, path: str, title: str = "Segmentation Results"):
        self._path = path
        self._title = title
        self._fh = None

    def __enter__(self):
        self._fh = open(self._path, "w", encoding="utf-8")
        self._fh.write(self._HEADER.format(title=self._title))
        return self

    def __exit__(self, *_):
        self._fh.write(self._FOOTER)
        self._fh.close()

    def add_figure(self, fig, title: str):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self._fh.write(
            f'<div class="card"><h3>{safe_title}</h3>'
            f'<img src="data:image/png;base64,{b64}"></div>\n'
        )


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """All dataset-specific settings in one place."""
    # Human-readable name used in log output
    name: str

    # How to produce an iterable of raw sample dicts.
    # If load_fn is set it is called with (num_samples: int) and must return an
    # iterable; dataset_path / dataset_name / split / streaming are ignored.
    # If load_fn is None, load_dataset(dataset_path, ...) is used instead.
    load_fn: Optional[Callable[[int], Iterable]] = None
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None   # HuggingFace config/subset name
    split: str = "train"
    streaming: bool = True

    # Field accessors — callables that accept a raw sample dict.
    # Returning None signals "not available / skip."
    get_image: Callable = None    # → PIL Image
    get_class: Callable = None    # → taxonomic class string for class_mapping lookup

    # If set, returns the SAM3 text prompt directly (bypasses class_mapping).
    # Useful when there is no class field or when all samples share a known prompt.
    get_prompt: Optional[Callable] = None

    # If set, returns a list of bounding boxes in normalized [x0, y0, x1, y1] format.
    # When bboxes are available they are used as the geometric prompt for SAM3.
    # The text prompt (get_class / get_prompt) is still used alongside if available,
    # since add_geometric_prompt benefits from language context.
    # Return None or [] to fall back to text-only prompting for that sample.
    get_bboxes: Optional[Callable] = None

    # Maps get_class() output to a SAM3 text prompt (e.g. "Aves" → "bird").
    class_mapping: Optional[dict] = None




MIN_MASK_AREA = 0.001
MAX_MASK_AREA = 0.80
DEDUP_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# SAM3 prompt helpers
# ---------------------------------------------------------------------------

def add_text_prompt(processor, inference_state, prompt):
    """Set a text prompt. Runs inference internally; call get_results() to read output."""
    processor.set_text_prompt(state=inference_state, prompt=prompt)


def add_bbox_prompts(processor, inference_state, bboxes_xyxy_norm):
    """Add one or more bounding boxes as geometric prompts.

    bboxes_xyxy_norm: list of [x0, y0, x1, y1] each normalised to [0, 1].
    Each box is converted to [cx, cy, w, h] (SAM3 format) before being added.
    Boxes accumulate in the state; inference re-runs after each addition so
    the final state reflects all boxes combined.
    If no text prompt has been set, SAM3 falls back to a "visual" language token.
    """
    for x0, y0, x1, y1 in bboxes_xyxy_norm:
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w  = x1 - x0
        h  = y1 - y0
        processor.add_geometric_prompt([cx, cy, w, h], label=True, state=inference_state)


def get_results(inference_state):
    """Extract (mask, score) pairs from the current inference state."""
    scores = inference_state.get("scores", [])
    masks  = inference_state.get("masks", [])
    return [(masks[i].squeeze(), scores[i].item()) for i in range(len(scores))]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def classify_mask_size(mask):
    area_frac = mask.float().mean().item()
    if area_frac < MIN_MASK_AREA:
        return "too small"
    if area_frac > MAX_MASK_AREA:
        return "too large"
    return "valid"


def render_results(pil_image, tagged_masks, report, title, input_bboxes=None):
    """Render image with predicted masks overlaid and (optionally) input bboxes.

    tagged_masks:  list of (mask, score, size_tag)
    input_bboxes:  list of [x0, y0, x1, y1] normalised — drawn as dashed rectangles
                   so you can compare the prompt region to the predicted mask.
    """
    img_w, img_h = pil_image.size
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pil_image)

    for i, (mask, score, size_tag) in enumerate(tagged_masks):
        mask_np = mask.cpu().numpy()
        alpha = 0.25 if size_tag != "valid" else 0.5
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[..., 0] = 1.0  # red
        colored_mask[..., 3] = mask_np * alpha
        ax.imshow(colored_mask)

        ys, xs = np.where(mask_np)
        if len(ys) > 0:
            cy, cx = ys.mean(), xs.mean()
            label = f"{score:.2f}" if size_tag == "valid" else f"{score:.2f} ({size_tag})"
            ax.text(cx, cy, label, color="white", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor=(1, 0, 0), alpha=0.7))

    if input_bboxes:
        for x0, y0, x1, y1 in input_bboxes:
            rect = plt.Rectangle(
                (x0 * img_w, y0 * img_h),
                (x1 - x0) * img_w,
                (y1 - y0) * img_h,
                linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)

    ax.axis("off")
    report.add_figure(fig, title)


# ---------------------------------------------------------------------------
# Core processing (dataset-agnostic)
# ---------------------------------------------------------------------------

def process_image(
    processor, pil_image, cls_prompt, bboxes,
    qwen_checker=None, use_qwen_crop=False, qwen_crop_padding=0.10, use_qwen_isolate=False,
) -> dict:
    """Run SAM3 with text, bbox, or both prompts depending on what is available.

    cls_prompt:         text prompt string, or None
    bboxes:             list of [x0, y0, x1, y1] normalised boxes, or None / []
    qwen_checker:       optional QwenMaskChecker; if provided, evaluates each
                        valid mask individually
    use_qwen_crop:      if True, crop the image to the ROI before Qwen sees it
    qwen_crop_padding:  pixels of padding around the crop box
    use_qwen_isolate:   if True, black out non-mask pixels and send one image;
                        if False, send original + red-overlay image (default)

    Returns a dict:
      {"mask_records": list[dict], "tagged": list[(mask, score, size_tag)], "title": str}
    mask_records contains one entry per accepted mask; tagged covers all masks for rendering.
    """
    has_text = cls_prompt is not None
    has_bbox = bool(bboxes)

    if not has_text and not has_bbox:
        return {"mask_records": [], "tagged": [], "title": "No prompt available, skipping"}

    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)

    if has_text:
        add_text_prompt(processor, inference_state, cls_prompt)
    if has_bbox:
        add_bbox_prompts(processor, inference_state, bboxes)

    all_masks = get_results(inference_state)

    prompt_desc = " + ".join(filter(None, [cls_prompt, "bbox" if has_bbox else None]))
    if not all_masks:
        return {"mask_records": [], "tagged": [], "title": f"{prompt_desc} — No mask returned, skipping"}

    if has_bbox:
        best_mask, best_score = max(all_masks, key=lambda x: x[1])
        tagged = [(best_mask, best_score, classify_mask_size(best_mask))]
    else:
        tagged = [(mask, score, classify_mask_size(mask)) for mask, score in all_masks]

    has_valid = any(tag == "valid" for _, _, tag in tagged)
    title = prompt_desc if has_valid else f"{prompt_desc} — No valid masks, skipping"

    size_discarded = [(score, size_tag) for _, score, size_tag in tagged if size_tag != "valid"]
    if size_discarded:
        title = title + "\nSize-discarded: " + " | ".join(
            f"score={score:.2f}: {size_tag}" for score, size_tag in size_discarded
        )

    mask_records = []

    if qwen_checker is not None and has_valid:
        verified_tagged = []
        discarded_tagged = []
        kept_reasons = []
        discarded_reasons = []
        text_prompt = cls_prompt or "species subject" # TODO: change this so there is not fallback
        for mask, score, size_tag in tagged:
            if size_tag != "valid":
                verified_tagged.append((mask, score, size_tag))
                continue
            # Optionally crop both the image and mask to the region of interest.
            if use_qwen_crop:
                verify_orig, verify_mask = crop_for_verification(
                    pil_image, mask, bboxes=bboxes, padding=qwen_crop_padding
                )
            else:
                verify_orig, verify_mask = pil_image, mask
            # Build input for Qwen: isolated single image or original + overlay.
            if use_qwen_isolate:
                verify_img = render_mask_isolation_pil(verify_orig, verify_mask)
                quality_ok, qwen_reason = qwen_checker.is_quality(
                    verify_img, None, text_prompt=text_prompt
                )
            else:
                masked_pil = render_masks_to_pil(verify_orig, [(verify_mask, score, size_tag)])
                quality_ok, qwen_reason = qwen_checker.is_quality(
                    verify_orig, masked_pil, text_prompt=text_prompt
                )
            print(f"  Qwen verdict (score={score:.2f}): {qwen_reason}")
            if quality_ok:
                verified_tagged.append((mask, score, size_tag))
                kept_reasons.append(f"score={score:.2f}: {qwen_reason}")
                mask_records.append({
                    "mask": mask, "score": score, "size_tag": size_tag,
                })
            else:
                discarded_tagged.append((mask, score, "discarded"))
                discarded_reasons.append(f"score={score:.2f}: {qwen_reason}")
        # Include discarded masks in render so they appear in the HTML.
        tagged = verified_tagged + discarded_tagged
        has_valid = any(tag == "valid" for _, _, tag in verified_tagged)
        if discarded_tagged:
            if not has_valid:
                title = f"{prompt_desc} — all masks discarded by Qwen"
            else:
                title = f"{title} — {len(discarded_tagged)} mask(s) discarded by Qwen"
        reason_lines = []
        if kept_reasons:
            reason_lines.append("Kept: " + " | ".join(kept_reasons))
        if discarded_reasons:
            reason_lines.append("Discarded: " + " | ".join(discarded_reasons))
        if reason_lines:
            title = title + "\n" + "\n".join(reason_lines)
    else:
        for mask, score, size_tag in tagged:
            if size_tag == "valid":
                mask_records.append({
                    "mask": mask, "score": score, "size_tag": size_tag,
                })

    return {"mask_records": mask_records, "tagged": tagged, "title": title}


def process_sample(processor, sample, config, sample_idx: int = 0,
                   qwen_crop: bool = False, qwen_isolate_mask: bool = False,
                   qwen_crop_padding: float = 0.10,
                   duplicate_detector=None) -> tuple:
    """Return (records, render_info).

    records      — list of parquet row dicts for accepted masks (empty if none).
    render_info  — dict with keys pil_image, tagged, title, bboxes for HTML
                   rendering; None if the sample was skipped entirely.
    duplicate_detector — optional DuplicateDetector; when provided, images that
                   are near-duplicates of a previously seen image are skipped
                   (returns ([], None)).  Non-duplicate images are registered
                   with the detector before processing continues.
    """
    pil_image = config.get_image(sample)
    if pil_image is None:
        return [], None

    iw, ih = pil_image.size
    if iw < 100 or ih < 100:
        print(f"  [size] Skipping image {iw}x{ih} at sample {sample_idx} (< 100px on a dimension)")
        return [], None

    if duplicate_detector is not None:
        if duplicate_detector.is_duplicate_and_add(pil_image):
            print(f"  [dedup] Skipping duplicate image at sample {sample_idx}")
            return [], None

    if config.get_prompt is not None:
        cls_prompts = [config.get_prompt(sample) or None]
    else:
        raw_class = config.get_class(sample) if config.get_class else None
        if raw_class and config.class_mapping:
            if raw_class not in config.class_mapping:
                return [], None
            mapping_val = config.class_mapping[raw_class]
        else:
            mapping_val = raw_class
        # class_mapping values may be a list of candidates or a single string.
        cls_prompts = mapping_val if isinstance(mapping_val, list) else [mapping_val]

    bboxes = config.get_bboxes(sample) if config.get_bboxes else None

    result = None
    cls_prompt = None
    for candidate in cls_prompts:
        result = process_image(
            processor, pil_image, candidate, bboxes,
            qwen_checker=config._qwen_checker,
            use_qwen_crop=qwen_crop,
            qwen_crop_padding=qwen_crop_padding,
            use_qwen_isolate=qwen_isolate_mask,
        )
        cls_prompt = candidate
        if result["mask_records"]:
            break  # at least one mask accepted — stop trying

    render_info = {
        "pil_image": pil_image,
        "tagged": result["tagged"],
        "title": result["title"],
        "bboxes": bboxes,
    }

    if not result["mask_records"]:
        return [], render_info

    iw, ih = pil_image.size
    sample_meta = _sample_metadata(sample)

    records = []
    for mr in result["mask_records"]:
        rle = mask_to_rle(mr["mask"])
        records.append({
            "dataset": config.name,
            "text_prompt": cls_prompt or "",
            "mask_rle_counts": rle["counts"],
            "mask_rle_height": int(rle["size"][0]),
            "mask_rle_width": int(rle["size"][1]),
            "mask_score": float(mr["score"]),
            "image_width": iw,
            "image_height": ih,
            **sample_meta,
        })
    return records, render_info


def _init_processor(config: DatasetConfig):
    """Load SAM3 model and Qwen checker; enter bfloat16 autocast for the process lifetime."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    config._qwen_checker = get_qwen_checker(device=device)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.3)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    return processor


def _make_stream(config: DatasetConfig, num_samples: int, shard_id: int = 0, total_shards: int = 1, categories=None, shuffle: bool = False, shuffle_seed: int = 42):
    """Build the sample iterable for the given config and sample limit."""
    if config.load_fn is not None:
        import inspect
        params = inspect.signature(config.load_fn).parameters
        kwargs = {}
        if "shard_id" in params:
            kwargs["shard_id"] = shard_id
            kwargs["total_shards"] = total_shards
        if "categories" in params:
            kwargs["categories"] = categories
        if "shuffle" in params:
            kwargs["shuffle"] = shuffle
            kwargs["shuffle_seed"] = shuffle_seed
        return config.load_fn(int(num_samples * 1.1), **kwargs)
    load_kwargs = dict(split=config.split, streaming=config.streaming)
    if config.dataset_name:
        load_kwargs["name"] = config.dataset_name
    ds = load_dataset(config.dataset_path, **load_kwargs)
    return ds


def _dataset_output_dir(config: DatasetConfig) -> str:
    """Return (and create) <SpeciesDataset>/datasets/<dataset_name>/outputs, anchored to this file."""
    import os
    import re
    safe_name = re.sub(r"[^\w\-]", "_", config.name)
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "datasets", safe_name, "outputs")
    os.makedirs(path, exist_ok=True)
    return path


def test_config(
    config: DatasetConfig,
    num_samples: int = 30,
    qwen_crop: bool = False,
    qwen_isolate_mask: bool = False,
    qwen_crop_padding: float = 0.10,
    categories: set = None,
    dedup_threshold: float = DEDUP_THRESHOLD,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 1000,
) -> None:
    """Process num_samples samples and write an HTML visualization.

    Writes report.html to <SpeciesDataset>/<dataset_name>/outputs/.
    If categories is given, only samples whose get_class() value is in the set are kept.
    """
    import os
    from duplicate_detection import DuplicateDetector

    out_subdir = _dataset_output_dir(config)
    output_html = os.path.join(out_subdir, "report.html")
    print(f"[test_config] {config.name}  |  samples={num_samples}  |  html={output_html}")
    try:
        os.remove(output_html)
        print(f"[test_config] Removing existing file: {output_html}")
    except FileNotFoundError:
        pass

    processor = _init_processor(config)
    ds_stream = _make_stream(config, num_samples, categories=categories, shuffle=shuffle, shuffle_seed=shuffle_seed)
    if shuffle:
        if hasattr(ds_stream, "shuffle"):
            ds_stream = ds_stream.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
        else:
            import random
            ds_stream = list(ds_stream)
            random.seed(shuffle_seed)
            random.shuffle(ds_stream)
    duplicate_detector = DuplicateDetector(threshold=dedup_threshold)

    n_skipped = 0
    with HtmlReport(output_html, title=config.name) as report:
        it = iter(ds_stream)
        i = 0
        n_scanned = 0
        try:
            while i < num_samples:
                sample = next(it)
                n_scanned += 1
                if categories is not None:
                    raw_class = config.get_class(sample) if config.get_class else None
                    if raw_class not in categories:
                        continue
                _, render_info = process_sample(processor, sample, config, sample_idx=i,
                                                qwen_crop=qwen_crop, qwen_isolate_mask=qwen_isolate_mask,
                                                qwen_crop_padding=qwen_crop_padding,
                                                duplicate_detector=duplicate_detector)
                if render_info is None:
                    n_skipped += 1
                else:
                    i += 1
                    render_results(render_info["pil_image"], render_info["tagged"],
                                   report, render_info["title"], input_bboxes=render_info["bboxes"])
        except StopIteration:
            pass
        finally:
            if hasattr(it, "close"):
                it.close()

    print(f"[test_config] {n_skipped} sample(s) skipped out of {i + n_skipped} processed ({n_scanned} scanned)")


def process_dataset(
    config: DatasetConfig,
    qwen_crop: bool = False,
    qwen_isolate_mask: bool = False,
    qwen_crop_padding: float = 0.10,
    num_samples: Optional[int] = None,
    shard_id: int = 0,
    total_shards: int = 1,
    dedup_threshold: float = DEDUP_THRESHOLD,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 1000,
) -> None:
    """Process the dataset and save accepted mask records to Parquet.

    When total_shards > 1, each call writes to a separate shard file:
      <SpeciesDataset>/<dataset_name>/outputs/masks_00042_of_00200.parquet
    Otherwise writes to masks.parquet.
    num_samples limits per-shard when sharding, or total otherwise.
    Near-duplicate images are always skipped via HOG cosine similarity (DEDUP_THRESHOLD).
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os
    from duplicate_detection import DuplicateDetector

    out_subdir = _dataset_output_dir(config)
    if total_shards > 1:
        parquet_output = os.path.join(
            out_subdir, f"masks_{shard_id:05d}_of_{total_shards:05d}.parquet"
        )
    else:
        parquet_output = os.path.join(out_subdir, "masks.parquet")

    # Write to a .tmp path and atomically rename after close, so the final
    # path only appears once the file is fully written (parquet footer included).
    # This prevents merge_shards from reading an incomplete shard.
    parquet_tmp = parquet_output + ".tmp"

    if os.path.exists(parquet_output):
        print(f"[process_dataset] Removing existing file: {parquet_output}")
        os.remove(parquet_output)
    if os.path.exists(parquet_tmp):
        os.remove(parquet_tmp)

    limit = num_samples if num_samples is not None else sys.maxsize
    shard_tag = f"shard={shard_id}/{total_shards}  |  " if total_shards > 1 else ""
    print(f"[process_dataset] {config.name}  |  {shard_tag}samples={'all' if num_samples is None else num_samples}  |  parquet={parquet_output}")

    processor = _init_processor(config)
    ds_stream = _make_stream(config, limit, shard_id=shard_id, total_shards=total_shards, shuffle=shuffle, shuffle_seed=shuffle_seed)
    if shuffle:
        if hasattr(ds_stream, "shuffle"):
            ds_stream = ds_stream.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
        else:
            import random
            ds_stream = list(ds_stream)
            random.seed(shuffle_seed)
            random.shuffle(ds_stream)
    duplicate_detector = DuplicateDetector(threshold=dedup_threshold)

    writer = None
    total = 0
    n_processed = 0
    n_dupes = 0
    start = time.time()

    it = iter(ds_stream)
    try:
        for i in range(limit):
            sample = next(it)
            records, render_info = process_sample(
                processor, sample, config, sample_idx=i,
                qwen_crop=qwen_crop, qwen_isolate_mask=qwen_isolate_mask,
                qwen_crop_padding=qwen_crop_padding,
                duplicate_detector=duplicate_detector,
            )
            if render_info is None and not records:
                n_dupes += 1
                continue
            if records:
                table = pa.Table.from_pandas(pd.DataFrame(records), preserve_index=False)
                # Promote all-null columns to large_string so schema stays consistent
                # across batches where optional fields (e.g. basis_of_record) happen to be all-None.
                null_cols = [f.name for f in table.schema if pa.types.is_null(f.type)]
                for col in null_cols:
                    table = table.set_column(
                        table.schema.get_field_index(col), col,
                        table.column(col).cast(pa.large_utf8()),
                    )
                if writer is None:
                    writer = pq.ParquetWriter(parquet_tmp, table.schema)
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                total += len(records)
            n_processed += 1
    except StopIteration:
        pass
    finally:
        if hasattr(it, "close"):
            it.close()
        if writer is not None:
            writer.close()
            os.rename(parquet_tmp, parquet_output)

    elapsed = time.time() - start
    h, m, s = int(elapsed) // 3600, (int(elapsed) % 3600) // 60, int(elapsed) % 60
    rate = n_processed / elapsed if elapsed > 0 else 0
    print(f"Dedup: {n_dupes} duplicate(s) skipped")
    if total:
        print(f"Done: {n_processed} samples  {total} masks  {h}h {m}m {s}s  ({rate:.2f} img/s)  → {parquet_output}")
    else:
        print(f"Done: {n_processed} samples  no masks accepted  {h}h {m}m {s}s")

    if total_shards > 1:
        all_shards = [
            os.path.join(out_subdir, f"masks_{i:05d}_of_{total_shards:05d}.parquet")
            for i in range(total_shards)
        ]
        if all(os.path.exists(p) for p in all_shards):
            lock_path = os.path.join(out_subdir, "masks.merge.lock")
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                pass  # another task already claimed the merge
            else:
                try:
                    merge_shards(out_subdir, delete_shards=True)
                except Exception:
                    os.remove(lock_path)  # allow retry on failure
                    raise


def merge_shards(dataset_dir: str, output_path: str = None, delete_shards: bool = False) -> None:
    """Merge all shard parquet files in dataset_dir into a single parquet file.

    Shard files match the pattern masks_NNNNN_of_MMMMM.parquet.
    Output defaults to <dataset_dir>/masks.parquet.
    Set delete_shards=True to remove the shard files after a successful merge.
    """
    import glob
    import os
    import pyarrow.parquet as pq
    import pyarrow as pa

    pattern = os.path.join(dataset_dir, "masks_*_of_*.parquet")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        print(f"No shard files found matching {pattern}")
        return

    if output_path is None:
        output_path = os.path.join(dataset_dir, "masks.parquet")

    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists. Move or delete it before merging.")

    print(f"Merging {len(shard_files)} shards → {output_path}")

    writer = None
    total = 0
    for path in shard_files:
        table = pq.read_table(path)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        total += len(table)
        print(f"  {os.path.basename(path)}: {len(table)} rows")

    if writer is not None:
        writer.close()

    print(f"Merged {total} rows → {output_path}")

    if delete_shards:
        for path in shard_files:
            os.remove(path)
        print(f"Deleted {len(shard_files)} shard files")


