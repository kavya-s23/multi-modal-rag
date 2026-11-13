import re
import statistics
from collections import Counter
from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect

BULLET = re.compile(r"^\s*([•●▪◦‣\-\*–]|\d+[.)]|[a-z][.)])\s+")
PAGE_NUM = re.compile(r"^(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?$", re.I)


def _bbox_str(x0, top, x1, bottom) -> str:
    return f"{x0:.1f},{top:.1f},{x1:.1f},{bottom:.1f}"


# ---------- tables ----------

def _table_to_markdown(rows: list[list]) -> str:
    clean = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    header, body = clean[0], clean[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _heading_above(page, top: float, ccfg: dict) -> str:
    """Text in the strip just above a table, joined into one line."""
    if top <= 1:
        return ""
    strip = page.crop((0, max(0, top - ccfg["table_heading_lookup_pts"]),
                       page.width, top))
    text = " ".join((strip.extract_text() or "").split())
    return text[-ccfg["table_heading_max_chars"]:]     # keep the end, nearest the table


def _table_chunks(page, page_number: int, base: dict, ccfg: dict):
    """Returns (table_chunks, table_boxes)."""
    chunks, boxes = [], []
    for n, table in enumerate(page.find_tables(), start=1):
        rows = [r for r in table.extract() if any((c or "").strip() for c in r)]
        if len(rows) < 2 or len(rows[0]) < 2:
            continue                                   # not a real table
        boxes.append(table.bbox)
        heading = _heading_above(page, table.bbox[1], ccfg)
        markdown = _table_to_markdown(rows)
        chunks.append({
            "id": f"{base['document_id']}_p{page_number}_t{n}",
            "text": f"Table: {heading}\n{markdown}" if heading else markdown,
            "metadata": {**base, "position": 0, "type": "table",
                         "bbox": _bbox_str(*table.bbox)},
        })
    return chunks, boxes


# ---------- page headers, footers and page numbers ----------

def _norm(text: str) -> str:
    """Lowercase, collapse spaces, digits become '#', so 'Page 3' equals 'Page 4'."""
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def _in_margin(line: dict, page_height: float, ccfg: dict) -> bool:
    band = ccfg.get("margin_fraction", 0.1) * page_height
    return line["bottom"] <= band or line["top"] >= page_height - band


def _find_running_text(pdf, ccfg: dict) -> set[str]:
    """Margin text that repeats on many pages (running titles, footers)."""
    counts = Counter()
    for page in pdf.pages:
        lines = page.extract_text_lines(return_chars=False)
        counts.update({_norm(l["text"]) for l in lines
                       if _in_margin(l, page.height, ccfg)})
    need = max(3, int(len(pdf.pages) * ccfg.get("running_text_share", 0.4)))
    return {t for t, n in counts.items() if t and n >= need}


def _is_noise(line: dict, page_height: float, running: set[str], ccfg: dict) -> bool:
    if not _in_margin(line, page_height, ccfg):
        return False
    text = line["text"].strip()
    return _norm(text) in running or bool(PAGE_NUM.match(text))


# ---------- text: lines -> blocks -> chunks ----------

def _visible(line: dict) -> list[dict]:
    return [c for c in line.get("chars", []) if c["text"].strip()]


def _is_bold(line: dict) -> bool:
    chars = _visible(line)
    return bool(chars) and all("bold" in c.get("fontname", "").lower() for c in chars)


def _is_bullet(text: str) -> bool:
    return bool(BULLET.match(text))


def _is_heading(line: dict, body_size: float, bold_ok: bool, ccfg: dict) -> bool:
    text = line["text"].strip()
    if not text or len(text) > ccfg.get("heading_max_chars", 80):
        return False
    if _is_bullet(text) or text[-1] in ".,;":          # sentences end with punctuation
        return False
    chars = _visible(line)
    if not chars:
        return False
    bigger = statistics.mean(c["size"] for c in chars) >= body_size * 1.15
    return bigger or (bold_ok and _is_bold(line))


def _blocks(lines: list[dict], ccfg: dict) -> list[dict]:
    """Group lines into blocks: a heading, a bullet item or a paragraph."""
    sizes = [c["size"] for l in lines for c in _visible(l)]
    body_size = statistics.median(sizes) if sizes else 0
    bold_share = sum(_is_bold(l) for l in lines) / len(lines) if lines else 0
    bold_ok = bold_share < 0.5                         # if most text is bold, bold means nothing
    gap_factor = ccfg.get("paragraph_gap", 0.7)

    blocks, prev, prev_heading = [], None, False
    for line in lines:
        heading = _is_heading(line, body_size, bold_ok, ccfg)
        big_gap = prev is not None and (
            line["top"] - prev["bottom"] > gap_factor * (prev["bottom"] - prev["top"]))
        if not blocks or heading or prev_heading or _is_bullet(line["text"]) or big_gap:
            blocks.append({"lines": [], "heading": heading})
        blocks[-1]["lines"].append(line)
        prev, prev_heading = line, heading
    return blocks


def _size(lines: list[dict]) -> int:
    return sum(len(l["text"]) + 1 for l in lines)


def _overlap_tail(lines: list[dict], overlap: int) -> list[dict]:
    """The last few lines (up to `overlap` characters) to repeat in the next chunk."""
    tail, total = [], 0
    for line in reversed(lines):
        total += len(line["text"]) + 1
        if total > overlap:
            break
        tail.insert(0, line)
    return tail


def _group_blocks(blocks: list[dict], ccfg: dict) -> list[list[dict]]:
    size_limit, overlap, min_chars = (
        ccfg["chunk_size"], ccfg["chunk_overlap"], ccfg["min_chars"])
    groups, current, held = [], [], []                 # held = heading lines waiting for text

    for b in blocks:
        if b["heading"]:
            if _size(current) >= min_chars:            # a new section: close the chunk
                groups.append(current)
                current = []                           # no overlap across sections
            held += b["lines"]
            continue

        piece, held = held + b["lines"], []
        oversized = _size(piece) > size_limit
        if current and _size(current) + _size(piece) > size_limit:
            groups.append(current)
            current = _overlap_tail(current, overlap)  # size cut inside a section
        for line in piece:                             # splits mid-block only if the block is huge
            if oversized and current and _size(current) + len(line["text"]) > size_limit:
                groups.append(current)
                current = _overlap_tail(current, overlap)
            current.append(line)

    current += held                                    # a heading at the very end of the page
    if current:
        groups.append(current)
    return groups


def _text_chunks(page, page_number: int, base: dict, ccfg: dict,
                 page_height: float, running: set[str]):
    lines = [l for l in page.extract_text_lines()
             if not _is_noise(l, page_height, running, ccfg)]
    groups = _group_blocks(_blocks(lines, ccfg), ccfg)

    chunks = []
    for k, group in enumerate(groups, start=1):
        text = "\n".join(l["text"] for l in group).strip()
        if len(text) < ccfg["min_chars"]:
            continue
        bbox = _bbox_str(min(l["x0"] for l in group), min(l["top"] for l in group),
                         max(l["x1"] for l in group), max(l["bottom"] for l in group))
        chunks.append({
            "id": f"{base['document_id']}_p{page_number}_c{k}",
            "text": text,
            "metadata": {**base, "position": k, "type": "text", "bbox": bbox},
        })
    return chunks, sum(len(l["text"]) for l in lines)


# ---------- main ----------

def chunk_pdf(pdf_path: Path, document_id: str,
              cfg: dict) -> tuple[list[dict], list[int]]:
    """Returns (chunks, pages_needing_ocr)."""
    ccfg = cfg["chunking"]
    min_page_chars = cfg["ingestion"]["min_page_chars"]
    try:
        pdf = pdfplumber.open(pdf_path)
    except PDFPasswordIncorrect:
        raise ValueError("The PDF is password-protected.")

    chunks, ocr_pages = [], []
    with pdf:
        running = _find_running_text(pdf, ccfg)
        for page_number, page in enumerate(pdf.pages, start=1):
            base = {"document_id": document_id, "document_name": pdf_path.name,
                    "page_number": page_number}

            table_chunks, boxes = _table_chunks(page, page_number, base, ccfg)
            chunks += table_chunks

            text_page = page
            for box in boxes:                          # cut table areas out of the text
                text_page = text_page.outside_bbox(box)

            text_chunks, n_chars = _text_chunks(
                text_page, page_number, base, ccfg, page.height, running)
            chunks += text_chunks

            if n_chars < min_page_chars and not boxes:
                ocr_pages.append(page_number)          # probably a scanned page
    return chunks, ocr_pages