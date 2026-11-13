from src.query.validation import load_settings
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.storage.vector_store.chroma_store import ChromaStore

cfg = load_settings()
cfg["reranker"] = {**cfg["reranker"], "top_n": 100}      # keep every candidate, not just 5
store = ChromaStore(cfg)

needle = input("Text inside the chunk you expect: ").strip().lower()
question = input("Question: ").strip()

for doc_id in store.document_ids():
    chunks = [c for c in store.get_all(doc_id) if needle in c["text"].lower()]
    if not chunks:
        continue
    cand = hybrid_search(question, [doc_id], cfg)
    ranked = rerank(question, cand, cfg)
    for c in chunks:
        ids = [h["id"] for h in cand]
        rk = [h["id"] for h in ranked]
        print(f"\n{doc_id} | {c['id']} | page {c['metadata']['page_number']}")
        print("  starts:", c["text"][:80].replace("\n", " | "))
        print("  hybrid rank:", ids.index(c["id"]) + 1 if c["id"] in ids else "NOT in the 15 candidates")
        if c["id"] in rk:
            print("  rerank score:", round(ranked[rk.index(c["id"])]["rerank_score"], 3))