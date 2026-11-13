END = ".,;:!?"


def _is_title(line: str, max_chars: int) -> bool:
    s = line.strip()
    return (3 <= len(s) <= max_chars and s[-1] not in END
            and not s.startswith(("•", "●", "-", "|", "+")))


def make_outline(chunks: list[dict], document_id: str, name: str, cfg: dict) -> list[dict]:
    """One line per section (the opening line of chunks that start with a title)."""
    ccfg = cfg["chunking"]
    limit = ccfg.get("outline_chunk_chars", 700)
    max_chars = ccfg.get("heading_max_chars", 80)

    body = [c for c in chunks if c["metadata"].get("type") == "text"]
    body.sort(key=lambda c: (c["metadata"].get("page_number", 0), c["metadata"].get("position", 0)))

    entries, last = [], None
    for c in body:
        first = c["text"].strip().split("\n")[0].strip()
        if _is_title(first, max_chars) and first != last:
            entries.append((c["metadata"].get("page_number", 1), first))
            last = first
    if len(entries) < 3:                      # too few titles to be an outline
        return []

    header = f"Outline of {name} (section titles in order):\n"
    groups, cur, size = [], [], len(header)
    for page, title in entries:
        line = f"p.{page}: {title}"
        if cur and size + len(line) + 1 > limit:
            groups.append(cur)
            cur, size = [], len(header)
        cur.append((page, line))
        size += len(line) + 1
    groups.append(cur)

    return [{
        "id": f"{document_id}_outline{n}",
        "text": header + "\n".join(line for _, line in group),
        "metadata": {"document_id": document_id, "document_name": name,
                     "page_number": group[0][0], "position": 0, "type": "outline"},
    } for n, group in enumerate(groups, start=1)]