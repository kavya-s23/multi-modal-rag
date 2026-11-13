import threading

from sentence_transformers import CrossEncoder
from torch.nn import Sigmoid

_model = None
_model_name = None
_lock = threading.Lock()


def _get_model(name: str) -> CrossEncoder:
    """Load the model once, on first use, then reuse it."""
    global _model, _model_name
    with _lock:
        if _model is None or _model_name != name:
            _model = CrossEncoder(name, max_length=512)
            _model_name = name
    return _model


def rerank(question: str, candidates: list[dict], cfg: dict) -> list[dict]:
    """Score each (question, chunk) pair, sort best first, keep top_n."""
    if not candidates:
        return []
    rcfg = cfg["reranker"]
    model = _get_model(rcfg["model"])
    scores = model.predict(
        [(question, c["text"]) for c in candidates],
        activation_fn=Sigmoid(),               # raw logits become 0..1
        show_progress_bar=False,
    )
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:rcfg["top_n"]]