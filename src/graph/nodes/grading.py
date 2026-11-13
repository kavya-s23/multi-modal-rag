from src.graph.nodes.common import get_cfg
from src.graph.state import RAGState
from src.retrieval.grader import grade as grade_chunks
from src.retrieval.grader import rewrite_for_search


def grade(state: RAGState) -> dict:
    verdict, reason = grade_chunks(state["question"], state.get("ranked", []), get_cfg())
    print(f"[grade] {verdict} ({reason})")      # temporary debug print
    return {"verdict": verdict}


def rewrite_query(state: RAGState) -> dict:
    new_q = rewrite_for_search(state["question"], get_cfg())
    print(f"[rewrite] {state['question']!r} -> {new_q!r}")   # temporary debug print
    return {"question": new_q, "retries": state.get("retries", 0) + 1}