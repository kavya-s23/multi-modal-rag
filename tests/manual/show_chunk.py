from src.query.validation import load_settings
from src.storage.vector_store.chroma_store import ChromaStore

cfg = load_settings()
store = ChromaStore(cfg)
doc_id = store.document_ids()[0]

needle = input("Text to find in a chunk: ").strip().lower()
for c in store.get_all(doc_id):
    if needle in c["text"].lower():
        m = c["metadata"]
        print(f"\n--- {c['id']} | {m['type']} | page {m['page_number']} ---")
        print(c["text"])