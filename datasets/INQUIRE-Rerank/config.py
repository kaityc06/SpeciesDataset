import functools
from typing import Iterable

from species_segmentation import DatasetConfig


_qwen_pipe = None

@functools.lru_cache(maxsize=512)
def _extract_subject_with_llm(query: str) -> str:
    global _qwen_pipe
    if _qwen_pipe is None:
        from transformers import pipeline as _hf_pipeline
        _qwen_pipe = _hf_pipeline("text-generation", model="Qwen/Qwen3.5-2B", max_new_tokens=20)
    prompt = (
        "Extract the primary animal or biological subject from this image search query. "
        "Return only the subject noun (1 word), nothing else.\n\n"
        f'Query: "{query}"\n\nSubject:'
    )
    result = _qwen_pipe(prompt)[0]["generated_text"]
    answer = result[len(prompt):].strip().split("\n")[0]
    return answer.strip('"').strip("'").lower()


def _inquire_stream(num_samples: int) -> Iterable:
    from datasets import load_dataset

    ds = load_dataset("evendrow/INQUIRE-Rerank", split="test", streaming=True)

    yielded = 0
    for sample in ds:
        if yielded >= num_samples:
            break
        if sample.get("relevant", 0) != 1:
            continue

        pil_image = sample.get("image")
        if pil_image is None:
            continue

        query = sample.get("query", "")
        subject = _extract_subject_with_llm(query) if query else None

        yield {
            "image":        pil_image.convert("RGB"),
            "query":        query,
            "subject":      subject,
            "species_name": sample.get("inat24_species_name"),
            "category":     sample.get("category"),
            "iconic_group": sample.get("iconic_group"),
        }
        yielded += 1


INQUIRE_CONFIG = DatasetConfig(
    name="INQUIRE-Rerank",
    load_fn=_inquire_stream,
    get_image=lambda s: s["image"],
    get_prompt=lambda s: s.get("subject"),
)
