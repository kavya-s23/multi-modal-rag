import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import pdfplumber
BATCH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_caption_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string"},
                            "ocr_text": {"type": "string"},
                            "caption": {"type": "string"},
                        },
                        "required": ["image_id", "ocr_text", "caption"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}
# Keep this import aligned with your actual module filename.
# If the NVIDIA client code is in vision_model.py, change
# vision_model_query below to vision_model.
from ...models import vision_model_query as vision


PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"
CACHE_FILE = PROCESSED / "caption_cache.json"
response_format=BATCH_RESPONSE_FORMAT,


def _load_cache() -> dict:
    if CACHE_FILE.is_file():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )


def _is_retryable(exc: Exception) -> bool:
    """Return whether a transient API error may succeed if retried."""
    message = str(exc).upper()
    return any(
        marker in message
        for marker in (
            "429",
            "RATE LIMIT",
            "500",
            "502",
            "503",
            "504",
            "TIMEOUT",
            "TEMPORARILY UNAVAILABLE",
            "INTERNAL SERVER ERROR",
        )
    )


def _batches(items: list[dict], size: int):
    """Yield consecutive groups of items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def extract_images(
    pdf_path: Path,
    document_id: str,
    cfg: dict,
) -> tuple[list[dict], list[str]]:
    """Extract PDF images, caption them in batches, and return chunks/problems."""
    ingestion_cfg = cfg["ingestion"]
    image_cfg = cfg["image"]

    model = image_cfg["vision_model"]
    max_side = int(image_cfg["max_side_pixels"])

    batch_size = max(1, int(ingestion_cfg.get("caption_batch_size", 3)))
    retries = max(0, int(ingestion_cfg.get("caption_retries", 1)))
    retry_wait = max(0.0, float(ingestion_cfg.get("caption_retry_wait", 10)))
    call_delay = max(0.0, float(ingestion_cfg.get("caption_delay_seconds", 4)))

    output_dir = PROCESSED / document_id / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    document_name = pdf_path.name
    chunks: list[dict] = []
    problems: list[str] = []
    cache = _load_cache()
    last_call = float("-inf")

    with pdfplumber.open(pdf_path) as pdf:
        found = []

        # Find image candidates and filter out small icons.
        for page_number, page in enumerate(pdf.pages, start=1):
            for image in page.images:
                x0 = max(image["x0"], 0)
                top = max(image["top"], 0)
                x1 = min(image["x1"], page.width)
                bottom = min(image["bottom"], page.height)

                if (
                    x1 - x0 < ingestion_cfg["min_image_px"]
                    or bottom - top < ingestion_cfg["min_image_px"]
                ):
                    continue

                image_key = getattr(image.get("stream"), "objid", None)
                found.append(
                    (page_number, (x0, top, x1, bottom), image_key)
                )

        # Filter repeated logos and other recurring images.
        repeats = Counter(
            key for _, _, key in found if key is not None
        )
        found = [
            item
            for item in found
            if item[2] is None
            or repeats[item[2]] <= ingestion_cfg["max_repeat_pages"]
        ]

        # Respect the per-document image limit.
        max_images = int(ingestion_cfg["max_images_per_document"])
        if len(found) > max_images:
            problems.append(
                f"{len(found)} images found; only the first "
                f"{max_images} were used."
            )
            found = found[:max_images]

        # Crop and prepare images before sending any caption requests.
        per_page: Counter = Counter()
        records: list[dict] = []

        for page_number, bbox, _ in found:
            per_page[page_number] += 1
            image_number = per_page[page_number]
            image_file = output_dir / f"p{page_number:03d}_{image_number}.png"

            try:
                page = pdf.pages[page_number - 1]
                pil_image = (
                    page.crop(bbox)
                    .to_image(resolution=ingestion_cfg["image_dpi"])
                    .original.convert("RGB")
                )
                pil_image.save(image_file)

                image_hash = hashlib.sha1(
                    image_file.read_bytes()
                ).hexdigest()

                request_image = pil_image.copy()
                request_image.thumbnail((max_side, max_side))

                records.append(
                    {
                        "page_number": page_number,
                        "bbox": bbox,
                        "image_number": image_number,
                        "image_file": image_file,
                        "request_image": request_image,
                        "cache_key": f"{image_hash}:{model}",
                    }
                )
            except Exception as exc:
                logging.warning(
                    "Could not prepare image %s: %s",
                    image_file.name,
                    str(exc)[:200],
                )
                problems.append(
                    f"{image_file.name}: could not be prepared."
                )

        captions: dict[str, tuple[str, str]] = {}
        pending: list[dict] = []

        # Reuse captions already cached for this image and model.
        for record in records:
            image_id = record["image_file"].name
            cache_key = record["cache_key"]

            if cache_key in cache:
                ocr_text, caption = cache[cache_key]
                captions[image_id] = (ocr_text, caption)
                print(f"    Using cached caption for {image_id} ...")
            else:
                pending.append(record)

        # Caption groups of images with the NVIDIA model.
        for batch_number, batch in enumerate(
            _batches(pending, batch_size),
            start=1,
        ):
            image_pairs = [
                (record["image_file"].name, record["request_image"])
                for record in batch
            ]
            image_names = ", ".join(
                record["image_file"].name for record in batch
            )

            batch_results = None
            last_error = None

            for attempt in range(retries + 1):
                wait = call_delay - (time.monotonic() - last_call)
                if wait > 0:
                    time.sleep(wait)

                last_call = time.monotonic()
                print(
                    f"    Captioning batch {batch_number} "
                    f"({len(batch)} images): {image_names} "
                    f"(attempt {attempt + 1}/{retries + 1}) ..."
                )

                try:
                    batch_results = vision.read_images(image_pairs, model)
                    break
                except Exception as exc:
                    last_error = exc

                    if not _is_retryable(exc) or attempt >= retries:
                        break

                    print(
                        f"    Temporary NVIDIA API error; "
                        f"retrying in {retry_wait:g}s ..."
                    )
                    time.sleep(retry_wait)

            if batch_results is None:
                error_message = (
                    str(last_error)[:200]
                    if last_error
                    else "The API returned no response."
                )
                for record in batch:
                    image_name = record["image_file"].name
                    logging.warning(
                        "Image %s failed: %s",
                        image_name,
                        error_message,
                    )
                    problems.append(
                        f"{image_name}: could not be captioned."
                    )
                continue

            # Save each successful result under its own image/model cache key.
            for record in batch:
                image_name = record["image_file"].name

                if image_name not in batch_results:
                    problems.append(
                        f"{image_name}: missing from the batch response."
                    )
                    continue

                ocr_text, caption = batch_results[image_name]
                captions[image_name] = (ocr_text, caption)
                cache[record["cache_key"]] = [ocr_text, caption]

            _save_cache(cache)

        # Build chunks in PDF image order.
        for record in records:
            image_file = record["image_file"]
            image_name = image_file.name

            if image_name not in captions:
                continue

            ocr_text, caption = captions[image_name]
            text = caption.strip()

            snippet = " ".join(ocr_text.split())[
                :ingestion_cfg["max_image_text_chars"]
            ]
            if snippet:
                text += f"\nText in image: {snippet}"

            if not text.strip():
                problems.append(
                    f"{image_name}: nothing could be read."
                )
                continue

            page_number = record["page_number"]
            image_number = record["image_number"]
            x0, top, x1, bottom = record["bbox"]

            chunks.append(
                {
                    "id": (
                        f"{document_id}_p{page_number}_i{image_number}"
                    ),
                    "text": text,
                    "metadata": {
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "position": 0,
                        "type": "image",
                        "image_path": str(image_file),
                        "bbox": (
                            f"{x0:.1f},{top:.1f},"
                            f"{x1:.1f},{bottom:.1f}"
                        ),
                        "image_text": ocr_text,
                    },
                }
            )

    return chunks, problems