import re

from rank_bm25 import BM25Okapi

from src.models.text_embedder import embed_query
from src.storage.vector_store.chroma_store import ChromaStore

# Keeps IPs, versions and commands whole: 192.168.1.1, git-commit, v1.2
_TOKEN = re.compile(r"[a-z0-9]+(?:[._:/\-][a-z0-9]+)*")

_store: ChromaStore | None = None
_bm25_cache: dict[tuple[str, int], tuple[BM25Okapi, list[dict]]] = {}


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _get_store(cfg: dict) -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore(cfg)
    return _store


def _bm25_for(store: ChromaStore, doc_id: str) -> tuple[BM25Okapi, list[dict]]:
    chunks = store.get_all(doc_id)
    key = (doc_id, len(chunks))                 # a re-ingested document gets a new key
    if key not in _bm25_cache:
        _bm25_cache[key] = (BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks)
    return _bm25_cache[key]


def _bm25_hits(store: ChromaStore, doc_ids: list[str], question: str, k: int) -> list[dict]:
    tokens = tokenize(question)
    if not tokens:
        return []
    hits = []
    for doc_id in doc_ids:
        bm25, chunks = _bm25_for(store, doc_id)
        for chunk, score in zip(chunks, bm25.get_scores(tokens)):
            if score > 0:                       # no shared word, no hit
                hits.append({**chunk, "bm25": float(score)})
    hits.sort(key=lambda h: h["bm25"], reverse=True)
    return hits[:k]


def _rrf(lists: list[list[dict]], rrf_k: int) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum of 1 / (rrf_k + rank) over the lists."""
    merged: dict[tuple[str, str], dict] = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            key = (h["document_id"], h["id"])
            entry = merged.setdefault(key, {**h, "rrf": 0.0})
            entry["rrf"] += 1.0 / (rrf_k + rank)
            entry.update({k: v for k, v in h.items() if k in ("distance", "bm25")})
    return sorted(merged.values(), key=lambda h: h["rrf"], reverse=True)


def hybrid_search(question: str, doc_ids: list[str], cfg: dict) -> list[dict]:
    if not doc_ids:
        raise ValueError("No documents selected. Select at least one document.")

    store = _get_store(cfg)
    missing = [d for d in doc_ids if d not in store.document_ids()]
    if missing:
        raise ValueError(f"Not indexed in the database: {missing}. Run ingestion first.")

    rcfg = cfg["retrieval"]
    vector = store.search(doc_ids, embed_query(question, cfg), top_k=rcfg["vector_k"])
    keyword = _bm25_hits(store, doc_ids, question, rcfg["bm25_k"])
    return _rrf([vector, keyword], rcfg["rrf_k"])[:rcfg["candidates"]]