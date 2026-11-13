from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .block_docx import pack, to_chunks
from .pdf_chunker import BULLET, _table_to_markdown


def _is_bold(p: Paragraph) -> bool:
    runs = [r for r in p.runs if r.text.strip()]
    if not runs:
        return False
    style_bold = bool(p.style is not None and p.style.font.bold)
    return all(r.bold or (r.bold is None and style_bold) for r in runs)


def _kind(p: Paragraph, bold_ok: bool, max_chars: int) -> str:
    style = (p.style.name or "") if p.style is not None else ""
    text = p.text.strip()
    if style.startswith("Heading") or style == "Title":
        return "heading"
    if "List" in style or BULLET.match(text):
        return "bullet"
    if (bold_ok and len(text) <= max_chars and text[-1] not in ".,;:"
            and _is_bold(p)):
        return "heading"
    return "text"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _in_fallback(el) -> bool:
    """Word stores many text boxes twice; the 'Fallback' copy is a duplicate."""
    while el is not None:
        if el.tag.endswith("}Fallback"):
            return True
        el = el.getparent()
    return False


def _textbox_texts(element, doc) -> list[str]:
    out = []
    for box in element.iter(W + "txbxContent"):
        if _in_fallback(box):
            continue
        for p in box.iterchildren(W + "p"):
            text = Paragraph(p, doc).text.strip()
            if text:
                out.append(text)
    return out

def chunk_docx(path: Path, document_id: str, cfg: dict) -> list[dict]:
    ccfg = cfg["chunking"]
    max_chars = ccfg.get("heading_max_chars", 80)
    doc = Document(str(path))

    paras = [p for p in doc.paragraphs if p.text.strip()]
    bold_share = sum(_is_bold(p) for p in paras) / len(paras) if paras else 0
    bold_ok = bold_share < 0.5            # if most text is bold, bold means nothing

    blocks, tables, titles = [], [], {}   # tables: (section, heading, markdown)
    section, last_text = 0, ""
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if child.tag.endswith("}p"):
                p = Paragraph(child, doc)
            for t in _textbox_texts(child, doc):       # text inside text boxes
                blocks.append({"text": t, "heading": False})
                last_text = t
            text = p.text.strip()
            
            if not text:
                continue
            kind = _kind(p, bold_ok, max_chars)
            if kind == "heading":
                section += 1
                titles[section] = text
            blocks.append({"text": f"• {text}" if kind == "bullet" and not BULLET.match(text) else text,
                           "heading": kind == "heading"})
            last_text = text
        elif child.tag.endswith("}tbl"):
            rows = [[c.text for c in row.cells] for row in Table(child, doc).rows]
            rows = [r for r in rows if any(c.strip() for c in r)]
            if len(rows) >= 2 and len(rows[0]) >= 2:
                tables.append((max(section, 1), last_text[-150:], _table_to_markdown(rows)))

    chunks = to_chunks(pack(blocks, ccfg), document_id, path.name, titles)
    for n, (sec, heading, md) in enumerate(tables, start=1):
        meta = {"document_id": document_id, "document_name": path.name,
                "page_number": sec, "position": 0, "type": "table"}
        if titles.get(sec):
            meta["section"] = titles[sec]
        chunks.append({
            "id": f"{document_id}_s{sec}_t{n}",
            "text": f"Table: {heading}\n{md}" if heading else md,
            "metadata": meta,
        })
    return chunks