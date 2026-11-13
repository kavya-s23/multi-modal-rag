import time

from PIL import Image, ImageOps

from ...models import vision_model_query as vision
from ..schema import InputType, QueryObject


def _open_image(q: QueryObject, cfg: dict):
    max_side = cfg["image"]["max_side_pixels"]
    with Image.open(q.image_path) as img:
        q.image_format = img.format
        img = ImageOps.exif_transpose(img)    # fix camera rotation
        img = img.convert("RGB")
    q.image_width, q.image_height = img.size
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))   # shrinks in place, keeps ratio
    return img


def _read_image(image, cfg: dict) -> tuple[str, str]:
    """Call the vision model. Return (ocr_text, caption)."""
    return vision.read_image(image, cfg["image"]["vision_model"])


def _handle_image(q: QueryObject, cfg: dict) -> QueryObject:
    # Only text + image queries, only if validation passed, never redo.
    if q.input_type != InputType.TEXT_IMAGE or q.errors or q.image_caption:
        return q

    image = _open_image(q, cfg)
    ocr_text, caption = _read_image(image, cfg)

    if not ocr_text and not caption:
        q.errors.append("We couldn't read anything from this image.")
        return q

    q.image_ocr_text = ocr_text
    q.image_caption = caption
    return q


def handle_image(q: QueryObject, cfg: dict) -> QueryObject:
    start = time.perf_counter()
    try:
        return _handle_image(q, cfg)
    finally:
        q.timings["image_handler"] = (time.perf_counter() - start) * 1000