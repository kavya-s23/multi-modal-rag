import threading

from sentence_transformers import SentenceTransformer

# bge works best when the question (not the chunks) starts with this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_model_name = None
_lock = threading.Lock()


def _get_model(name: str) -> SentenceTransformer:
    """Load the model once, on first use, then reuse it."""
    global _model, _model_name
    with _lock:
        if _model is None or _model_name != name:
            _model = SentenceTransformer(name)
            _model_name = name
    return _model


def embed_documents(texts: list[str], cfg: dict) -> list[list[float]]:
    """For chunks: text chunks, tables and image captions."""
    ecfg = cfg["embedding"]
    vectors = _get_model(ecfg["model"]).encode(
        texts, batch_size=ecfg["batch_size"],
        normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str, cfg: dict) -> list[float]:
    """For the user's question. Uses the same model as the chunks."""
    model = _get_model(cfg["embedding"]["model"])
    return model.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()