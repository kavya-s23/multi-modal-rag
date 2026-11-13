from src.query.validation import load_settings
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.storage.vector_store.chroma_store import ChromaStore

cfg = load_settings()
doc_ids = ChromaStore(cfg).document_ids()
print("Documents:", doc_ids)

while True:
    q = input("\nQuestion (blank to quit): ").strip()
    if not q:
        break
    ranked = rerank(q, hybrid_search(q, doc_ids, cfg), cfg)
    scores = [round(c["rerank_score"], 3) for c in ranked]
    print("  top scores:", scores)
    print("  best chunk:", ranked[0]["text"][:70].replace("\n", " "))