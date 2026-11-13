from src.query.validation import load_settings
from src.retrieval.hybrid import hybrid_search
from src.storage.vector_store.chroma_store import ChromaStore

cfg = load_settings()
store = ChromaStore(cfg)
doc_id = store.document_ids()[0]

needle = "circular, metal disk"
matches = [c for c in store.get_all(doc_id) if needle in c["text"]]
print(f"Chunks containing {needle!r}: {len(matches)}")
for c in matches:
    m = c["metadata"]
    print(f"  {c['id']} page {m['page_number']} ({len(c['text'])} chars)")
    print("  ", c["text"][:300].replace("\n", " "))

hits = hybrid_search("what is a platter", [doc_id], cfg)
ids = [h["id"] for h in hits]
for c in matches:
    rank = ids.index(c["id"]) + 1 if c["id"] in ids else None
    print(f"\n{c['id']} hybrid rank: {rank} of {len(ids)}")