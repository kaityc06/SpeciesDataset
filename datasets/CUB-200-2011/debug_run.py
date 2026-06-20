import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import CUB200_CONFIG
import species_segmentation as ss
from species_segmentation import (
    get_results,
    classify_mask_size,
    add_text_prompt,
    add_bbox_prompts,
)

MAX_VALID_MASKS = 5
DEBUG_OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_many_masks")
os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

_original_process_image = ss.process_image
_current_sample_idx = 0


def _debug_process_image(
    processor, pil_image, cls_prompt, bboxes,
    qwen_checker=None, use_qwen_crop=False, qwen_crop_padding=0.10, use_qwen_isolate=False,
):
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
        return {"mask_records": [], "tagged": [], "title": f"{prompt_desc} — No mask returned"}

    if has_bbox:
        best_mask, best_score = max(all_masks, key=lambda x: x[1])
        tagged = [(best_mask, best_score, classify_mask_size(best_mask))]
    else:
        tagged = [(mask, score, classify_mask_size(mask)) for mask, score in all_masks]

    n_valid = sum(1 for _, _, tag in tagged if tag == "valid")
    print(f"  [debug] sample={_current_sample_idx}  total_masks={len(tagged)}  valid={n_valid}  prompt={cls_prompt!r}")

    if n_valid > MAX_VALID_MASKS:
        out_path = os.path.join(DEBUG_OUT_DIR, f"sample_{_current_sample_idx:04d}_{n_valid}masks.png")
        pil_image.save(out_path)
        print(
            f"  [debug] *** sample {_current_sample_idx} has {n_valid} valid masks "
            f"(>{MAX_VALID_MASKS}) — saved to {out_path} — skipping Qwen ***"
        )
        return {
            "mask_records": [],
            "tagged": tagged,
            "title": f"{prompt_desc} — DEBUG: {n_valid} valid masks, Qwen skipped",
        }

    # Mask count is fine — run the real pipeline.
    return _original_process_image(
        processor, pil_image, cls_prompt, bboxes,
        qwen_checker=qwen_checker,
        use_qwen_crop=use_qwen_crop,
        qwen_crop_padding=qwen_crop_padding,
        use_qwen_isolate=use_qwen_isolate,
    )


_original_process_sample = ss.process_sample


def _debug_process_sample(processor, sample, config, sample_idx=0, **kwargs):
    global _current_sample_idx
    _current_sample_idx = sample_idx
    return _original_process_sample(processor, sample, config, sample_idx=sample_idx, **kwargs)


# Patch both into the module so internal calls pick them up.
ss.process_image = _debug_process_image
ss.process_sample = _debug_process_sample

from species_segmentation import test_config

test_config(
    CUB200_CONFIG,
    qwen_crop=True,
    qwen_isolate_mask=True,
    num_samples=30,
    dedup_threshold=0.99,
    shuffle=True,
)

os._exit(0)
