from pathlib import Path

from src.ingestion.chunking.docx_chunker import chunk_docx
from src.ingestion.chunking.txt_chunker import chunk_txt
from src.query.validation import load_settings

cfg = load_settings()
path = Path(input("Path to a .docx or .txt file: ").strip().strip('"'))
fn = chunk_docx if path.suffix.lower() == ".docx" else chunk_txt
chunks = fn(path, "doc_test", cfg)

print(f"{len(chunks)} chunks")
for c in chunks:
    m = c["metadata"]
    print(f"\n[{m['type']}] section {m['page_number']} ({len(c['text'])} chars)")
    print(c["text"][:300])