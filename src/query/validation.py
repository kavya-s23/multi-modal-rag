from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

from .schema import InputType, QueryObject


def load_settings(path: str | Path = "configs/settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_scope(q: QueryObject, cfg: dict) -> list[str]:
    errors = []
    if len(set(q.document_ids)) != len(q.document_ids):
        errors.append("The same document is selected more than once.")
    limit = cfg["query"]["max_documents_per_query"]
    if len(q.document_ids) > limit:
        errors.append(f"{len(q.document_ids)} documents selected; "
                      f"the limit is {limit}.")
    return errors


def _check_text(q: QueryObject, cfg: dict) -> list[str]:
    text = (q.original_text or "").strip()
    lo = cfg["query"]["min_query_chars"]
    hi = cfg["query"]["max_query_chars"]
    if len(text) < lo:
        return [f"The question is too short (minimum {lo} characters)."]
    if len(text) > hi:
        return [f"The question is {len(text)} characters; the limit is {hi}."]
    return []


def _check_file(path: Path, allowed: list[str], max_mb: float,
                label: str) -> list[str]:
    if not path.is_file():
        return [f"The {label} file was not found."]
    if path.suffix.lower() not in allowed:
        return [f"The {label} type '{path.suffix}' is not supported. "
                f"Allowed: {', '.join(allowed)}."]
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return [f"The {label} is {size_mb:.1f} MB; the limit is {max_mb} MB."]
    return []


def _check_image(q: QueryObject, cfg: dict) -> list[str]:
    icfg = cfg["image"]
    errors = _check_file(q.image_path, icfg["allowed_extensions"],
                         icfg["max_file_mb"], "image")
    if errors:
        return errors            # skip opening a file that already failed
    try:
        with Image.open(q.image_path) as img:
            width, height = img.size
            img.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return ["The image file is corrupt or is not a real image."]
    if width * height > icfg["max_image_pixels"]:
        return [f"The image is {width}x{height} pixels, which is too large."]
    return []


def _check_audio(q: QueryObject, cfg: dict) -> list[str]:
    acfg = cfg["audio"]
    return _check_file(q.audio_path, acfg["allowed_extensions"],
                       acfg["max_file_mb"], "audio")


def validate_query(q: QueryObject, cfg: dict) -> QueryObject:
    """Appends user-facing problems to q.errors. Call once per query."""
    errors = _check_scope(q, cfg)

    if q.input_type == InputType.IMAGE:
        errors.append("Please type a question along with your image.")

    if q.input_type in (InputType.TEXT, InputType.TEXT_IMAGE):
        errors += _check_text(q, cfg)
    if q.input_type in (InputType.IMAGE, InputType.TEXT_IMAGE):
        errors += _check_image(q, cfg)
    if q.input_type == InputType.AUDIO:
        errors += _check_audio(q, cfg)

    q.errors.extend(errors)
    return q