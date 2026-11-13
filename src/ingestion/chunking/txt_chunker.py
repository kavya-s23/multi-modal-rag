import re
import statistics
from pathlib import Path

from .block_docx import pack, to_chunks

BULLET = re.compile(r"^\s*([•●▪◦‣\-\*–\+]|\d+[.)]|[a-z][.)])\s+")


def _read(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _is_heading(line: str, max_chars: int, typical: float) -> bool:
    s = line.strip()
    if not s or "\t" in s:                       # tab-separated rows are table data
        return False
    if s.startswith("#"):
        return True
    return (len(s) <= min(max_chars, 0.5 * typical)
            and not BULLET.match(s) and s[-1] not in ".,;:!?")


def _blocks(text: str, ccfg: dict) -> list[dict]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lens = [len(l.strip()) for l in text.split("\n") if len(l.strip()) > 40]
    typical = statistics.median(lens) if lens else 80      # normal body-line length
    max_chars = ccfg.get("heading_max_chars", 80)

    blocks = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        current = None
        for line in lines:
            if len(lines) == 1:                  # alone between blank lines: judge by shape only
                heading = (line.startswith("#") or (
                    len(line) <= max_chars and not BULLET.match(line)
                    and line[-1] not in ".,;:!?" and "\t" not in line))
            else:
                heading = _is_heading(line, max_chars, typical)

            if heading:
                if current:
                    blocks.append({"text": current, "heading": False})
                    current = None
                blocks.append({"text": line.lstrip("# ").strip(), "heading": True})
            elif current is None or BULLET.match(line):
                if current:
                    blocks.append({"text": current, "heading": False})
                current = line
            else:
                current += " " + line            # hard-wrapped continuation
        if current:
            blocks.append({"text": current, "heading": False})
    return blocks


def chunk_txt(path: Path, document_id: str, cfg: dict) -> list[dict]:
    text = _read(path)
    packed = pack(_blocks(text, cfg["chunking"]), cfg["chunking"])
    return to_chunks(packed, document_id, path.name)