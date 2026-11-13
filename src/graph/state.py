from typing import TypedDict


class Citation(TypedDict, total=False):
    doc_id: str
    page: int
    bbox: list[float] | None   # text chunks may have none
    chunk_id: str


class RAGState(TypedDict, total=False):
    # --- input (set by main.py) ---
    query: dict                 # raw input from main.py
    doc_ids: list[str]          # user-selected documents, mandatory

    # --- conversation (kept across turns by the checkpointer) ---
    history: list[dict]         # prior {"question": ..., "answer": ...} turns

    # --- query processing ---
    qobj: dict                  # the QueryObject, dumped to a dict
    question: str               # current working question
    errors: list[str]           # any entry stops the graph and is shown to the user
    transcript_confirmed: bool

    # --- retrieval ---
    candidates: list[dict]
    ranked: list[dict]
    verdict: str                # "relevant" | "irrelevant"
    retries: int                # 0 or 1, caps the rewrite loop

    # --- output ---
    answer: str
    citations: list[Citation]
    refused: bool               # True means "Not in the document"