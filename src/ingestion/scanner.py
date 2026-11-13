import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FileInfo:
    path: Path
    name: str
    extension: str
    size: int
    file_hash: str       # changes when the content changes
    document_id: str     # stays the same when the content changes


@dataclass
class ScanResult:
    files: list[FileInfo] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # unsupported types
    error: str | None = None


def _hash_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _document_id(path: Path) -> str:
    return "doc_" + hashlib.sha1(str(path).lower().encode()).hexdigest()[:12]


def _info(path: Path) -> FileInfo:
    return FileInfo(path=path, name=path.name, extension=path.suffix.lower(),
                    size=path.stat().st_size, file_hash=_hash_file(path),
                    document_id=_document_id(path))


def scan_path(raw_path: str, cfg: dict) -> ScanResult:
    """Accepts a file or a folder and returns the supported files in it."""
    icfg = cfg["ingestion"]
    allowed = set(icfg["allowed_extensions"])
    max_bytes = icfg["max_file_mb"] * 1024 * 1024

    path = Path(raw_path.strip().strip('"')).expanduser()
    if not path.exists():
        return ScanResult(error=f"Path not found: {path}")
    path = path.resolve()

    # A single file
    if path.is_file():
        if path.suffix.lower() not in allowed:
            return ScanResult(error="Only PDF, DOCX and TXT files are supported.")
        if path.stat().st_size > max_bytes:
            return ScanResult(error=f"File is larger than {icfg['max_file_mb']} MB.")
        return ScanResult(files=[_info(path)])

    # A folder
    result = ScanResult()
    candidates = path.rglob("*") if icfg["recursive"] else path.glob("*")
    for p in sorted(candidates):
        if not p.is_file():
            continue
        rel = p.relative_to(path)
        if any(part.startswith(".") for part in rel.parts):
            continue                                   # hidden files and folders
        if p.suffix.lower() not in allowed:
            result.skipped.append(rel.as_posix())
        elif p.stat().st_size > max_bytes:
            result.skipped.append(f"{rel.as_posix()} (too large)")
        else:
            result.files.append(_info(p))
    if not result.files:
        result.error = "No PDF, DOCX or TXT files were found in that folder."
    return result