import re


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Split one long paragraph by sentence, with a little overlap at word boundaries."""
    if len(text) <= size:
        return [text]
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        while len(s) > size:                           # a sentence with no breaks
            if cur:
                parts.append(cur)
                cur = ""
            cut = s.rfind(" ", 0, size)                # cut at a space, not mid-word
            cut = cut if cut > 0 else size
            parts.append(s[:cut])
            rest = s[max(cut - overlap, 0):]
            if " " in rest[:overlap] and cut - overlap > 0 and s[cut - overlap - 1] != " ":
                rest = rest.split(" ", 1)[1]           # drop a partial first word
            s = rest
        if cur and len(cur) + len(s) + 1 > size:
            parts.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail and len(cur) > overlap and cur[-overlap - 1] != " ":
                tail = tail.split(" ", 1)[1] if " " in tail else ""   # drop a partial first word
            cur = f"{tail} {s}".strip()
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def _len(lines: list[str]) -> int:
    return sum(len(x) + 1 for x in lines)


def pack(blocks: list[dict], ccfg: dict) -> list[tuple[str, int]]:
    """blocks: [{"text": str, "heading": bool}]. Returns [(chunk_text, section_number)]."""
    size, overlap, min_chars = ccfg["chunk_size"], ccfg["chunk_overlap"], ccfg["min_chars"]
    chunks: list[tuple[str, int]] = []
    cur: list[str] = []
    held: list[str] = []                               # headings waiting for their text
    section, cur_section = 0, 1

    def flush():
        nonlocal cur
        if cur:
            chunks.append(("\n".join(cur), cur_section))
            cur = []

    for b in blocks:
        text = b["text"].strip()
        if not text:
            continue
        if b["heading"]:
            if _len(cur) >= min_chars:                 # a new section closes the chunk
                flush()
            section += 1
            held.append(text)
            continue

        parts = _split_long(text, size, overlap)
        if held:
            parts[0] = "\n".join(held + [parts[0]])
            held = []
        for p in parts:
            if cur and _len(cur) + len(p) + 1 > size:
                flush()
            if not cur:
                cur_section = max(section, 1)
            cur.append(p)

    cur += held                                        # a heading at the very end
    flush()
    return [(t, s) for t, s in chunks if len(t) >= min_chars]


def to_chunks(packed: list[tuple[str, int]], document_id: str, name: str,
              titles: dict | None = None) -> list[dict]:
    """Same chunk shape as the PDF chunker. No bbox: these formats have no pages."""
    out = []
    for n, (text, section) in enumerate(packed, start=1):
        meta = {"document_id": document_id, "document_name": name,
                "page_number": section, "position": n, "type": "text"}
        if titles and titles.get(section):
            meta["section"] = titles[section]
        out.append({"id": f"{document_id}_s{section}_c{n}", "text": text, "metadata": meta})
    return out