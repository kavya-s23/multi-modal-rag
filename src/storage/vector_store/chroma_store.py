from pathlib import Path

import chromadb

from .base import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUFFIX = "_text"


def _clean_meta(chunk: dict) -> dict:
    """Chroma keeps only simple values (str, int, float, bool)."""
    meta = dict(chunk["metadata"])
    if "image_text" in chunk:                  # works even if it sits outside metadata
        meta["image_text"] = chunk["image_text"]
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


class ChromaStore(VectorStore):
    def __init__(self, cfg: dict):
        path = Path(cfg["storage"]["chroma_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))

    @staticmethod
    def _name(document_id: str) -> str:
        return f"{document_id}{SUFFIX}"

    def delete_document(self, document_id: str) -> None:
        try:
            self._client.delete_collection(self._name(document_id))
        except Exception:
            pass                               # it did not exist

    def add_chunks(self, document_id, chunks, vectors) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors do not match.")
        self.delete_document(document_id)      # a rebuilt document starts clean
        col = self._client.create_collection(
            self._name(document_id), metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 500):
            part = chunks[i:i + 500]
            col.add(ids=[c["id"] for c in part],
                    documents=[c["text"] for c in part],
                    embeddings=vectors[i:i + 500],
                    metadatas=[_clean_meta(c) for c in part])

    def search(self, document_ids, query_vector, top_k=5, where=None):
        hits = []
        for doc_id in document_ids:            # only the selected documents
            try:
                col = self._client.get_collection(self._name(doc_id))
            except Exception:
                continue                       # not indexed
            n = min(top_k, col.count())
            if n == 0:
                continue
            res = col.query(query_embeddings=[query_vector], n_results=n, where=where)
            for i in range(len(res["ids"][0])):
                hits.append({"id": res["ids"][0][i], "document_id": doc_id,
                             "text": res["documents"][0][i],
                             "metadata": res["metadatas"][0][i],
                             "distance": res["distances"][0][i]})
        hits.sort(key=lambda h: h["distance"])
        return hits[:top_k]
    
    def get_all(self, document_id: str) -> list[dict]:
        """Every chunk of one document, used to build the BM25 index."""
        col = self._client.get_collection(self._name(document_id))
        res = col.get(include=["documents", "metadatas"])
        return [{"id": i, "document_id": document_id, "text": t, "metadata": m}
                for i, t, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def document_ids(self) -> list[str]:
        names = [c if isinstance(c, str) else c.name
                 for c in self._client.list_collections()]
        return sorted(n[:-len(SUFFIX)] for n in names if n.endswith(SUFFIX))