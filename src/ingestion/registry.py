import json
from pathlib import Path

from .scanner import FileInfo

REGISTRY_PATH = (Path(__file__).resolve().parents[2]
                 / "data" / "processed" / "registry.json")


def _load() -> dict:
    if REGISTRY_PATH.is_file():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def status(info: FileInfo) -> str:
    """'new', 'indexed' (same content already stored) or 'changed'."""
    entry = _load().get(info.document_id)
    if entry is None:
        return "new"
    return "indexed" if entry["file_hash"] == info.file_hash else "changed"


def record(info: FileInfo) -> None:
    """Call this only after a document was ingested successfully."""
    data = _load()
    data[info.document_id] = {
        "name": info.name,
        "path": str(info.path),
        "file_hash": info.file_hash,
        "text_collection": f"{info.document_id}_text",
        "image_collection": f"{info.document_id}_images",
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")