from src.graph.nodes.common import get_cfg
from src.graph.state import RAGState
from src.retrieval.hybrid import hybrid_search

from src.retrieval.reranker import rerank as rerank_chunks


def rerank(state: RAGState) -> dict:
    ranked = rerank_chunks(state["question"], state.get("candidates", []), get_cfg())

    # temporary debug print: remove once the grader exists
    for h in ranked:
        print(f"   score={h['rerank_score']:.3f} p.{h['metadata'].get('page_number')} "
              f"| {h['text'][:70]!r}")
    return {"ranked": ranked}
def retrieve(state: RAGState) -> dict:
    try:
        hits = hybrid_search(state["question"], state["doc_ids"], get_cfg())
    except ValueError as e:
        return {"errors": [str(e)]}

    # temporary debug print: remove once the grader exists
    print(f"[retrieve] {len(hits)} candidates for: {state['question']!r}")
    for h in hits[:3]:
        page = h["metadata"].get("page_number")
        print(f"   p.{page} dist={h.get('distance')} bm25={h.get('bm25')} | {h['text'][:70]!r}")
    return {"candidates": hits}