from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, document_id: str, chunks: list[dict],
                   vectors: list[list[float]]) -> None:
        """Store a document's chunks, replacing any older version."""

    @abstractmethod
    def search(self, document_ids: list[str], query_vector: list[float],
               top_k: int = 5, where: dict | None = None) -> list[dict]:
        """Closest chunks, searching only the given documents."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def document_ids(self) -> list[str]: ...