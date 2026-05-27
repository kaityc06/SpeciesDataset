import requests
from typing import Iterable
from PIL import Image
from io import BytesIO

from species_segmentation import DatasetConfig


GBIF_API = "https://api.gbif.org/v1/occurrence/search"
GBIF_PAGE_SIZE = 300


def load_image_from_url(url: str, timeout: int = 10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError) as e:
        print(f"  Skipping {url}: {e}")
        return None


def _gbif_occurrence_stream(num_samples: int, categories=None) -> Iterable:
    fetched = 0
    offset = 0
    while fetched < num_samples:
        params = {"mediaType": "StillImage", "limit": GBIF_PAGE_SIZE, "offset": offset}
        try:
            resp = requests.get(GBIF_API, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"GBIF API error: {e}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for occ in results:
            if fetched >= num_samples:
                return
            image_url = next(
                (m["identifier"] for m in occ.get("media", [])
                 if m.get("type") == "StillImage" and m.get("identifier")),
                None,
            )
            if image_url is None:
                continue
            cls = occ.get("class")
            if categories is not None and cls not in categories:
                continue
            yield {
                "image_url": image_url,
                "class":     cls,
                "species":   occ.get("species"),
                "family":    occ.get("family"),
            }
            fetched += 1

        offset += len(results)
        if data.get("endOfRecords", False):
            break


GBIF_CONFIG = DatasetConfig(
    name="GBIF (occurrence API)",
    load_fn=_gbif_occurrence_stream,
    get_image=lambda s: load_image_from_url(s["image_url"]),
    get_class=lambda s: s.get("class"),
    class_mapping={
        "Mammalia": "mammal",
        "Aves": "bird",
        "Amphibia": "amphibian",
        "Reptilia": "reptile",
        "Actinopterygii": "fish",
        "Chondrichthyes": "shark or ray",
        "Insecta": "insect",
        "Arachnida": "spider or mite",
        "Malacostraca": "shellfish",
        "Mollusca": "mollusk",
        "Gastropoda": "snail or slug",
        "Bivalvia": "clam or mussel",
        "Cephalopoda": "octopus or squid",
        "Plantae": "plant",
        "Fungi": "fungus",
    },
)
