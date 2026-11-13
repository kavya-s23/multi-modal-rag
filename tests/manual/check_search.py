from src.query.validation import load_settings
from src.models import text_embedder
from src.storage.vector_store.chroma_store import ChromaStore

cfg = load_settings()
store = ChromaStore(cfg)
docs = store.document_ids()
print("Indexed documents:", docs)
if not docs:
    raise SystemExit("Nothing is indexed yet. Run the ingestion check first.")

while True:
    q = input("\nQuestion (blank to quit): ").strip()
    if not q:
        break
    vec = text_embedder.embed_query(q, cfg)
    for h in store.search(docs, vec, top_k=5):
        m = h["metadata"]
        print(f"\n[{h['distance']:.3f}] {m['type']}, page {m['page_number']} ({h['id']})")
        print("  ", h["text"][:200].replace("\n", " "))