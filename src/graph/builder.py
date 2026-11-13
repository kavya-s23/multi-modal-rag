from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from src.graph.nodes import query, retrieval, stubs
from src.graph.nodes import query, stubs
from src.graph.state import RAGState
from src.graph.nodes import grading, query, retrieval, stubs
from src.graph.nodes import generation, grading, query, retrieval, stubs
from src.graph.nodes.common import get_cfg
NODES = {
    # real
    "validate": query.validate,
    "transcribe": query.transcribe,
    "confirm_transcript": query.confirm_transcript,
    "caption_image": query.caption_image,
    "clean": query.clean,
    "rewrite_followup": query.rewrite_followup,
    # still stubs
    "retrieve": retrieval.retrieve,
    "rerank": retrieval.rerank,
    "grade": grading.grade,
    "rewrite_query": grading.rewrite_query,
    "generate": generation.generate,
    "refuse": stubs.refuse,
}


# ---------- routing functions (conditional edges) ----------

def route_after_validate(state: RAGState) -> str:
    if state.get("errors"):
        return "stop"
    t = state["qobj"]["input_type"]            # text | audio | image | text_image
    return "image" if t in ("image", "text_image") else t


def stop_if_errors(state: RAGState) -> str:
    return "stop" if state.get("errors") else "go"


def route_followup(state: RAGState) -> str:
    return "rewrite" if state.get("history") else "skip"


def route_after_grade(state: RAGState) -> str:
    if state.get("verdict") == "relevant":
        return "generate"
    ranked = state.get("ranked", [])
    top = ranked[0]["rerank_score"] if ranked else 0.0
    if top < get_cfg()["grading"].get("skip_retry_below", 0.002):
        return "refuse"                          # nothing close: a rewrite can't help
    if state.get("retries", 0) < 1:
        return "retry"
    return "refuse"


# ---------- graph ----------

def build_graph():
    g = StateGraph(RAGState)
    for name, fn in NODES.items():
        g.add_node(name, fn)

    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", route_after_validate, {
        "text": "clean",
        "audio": "transcribe",
        "image": "caption_image",
        "stop": END,
    })

    g.add_conditional_edges("transcribe", stop_if_errors,
                            {"go": "confirm_transcript", "stop": END})
    g.add_conditional_edges("confirm_transcript", stop_if_errors,
                            {"go": "clean", "stop": END})
    g.add_conditional_edges("caption_image", stop_if_errors,
                            {"go": "clean", "stop": END})

    g.add_conditional_edges("clean", route_followup, {
        "rewrite": "rewrite_followup",
        "skip": "retrieve",
    })
    g.add_edge("rewrite_followup", "retrieve")

    g.add_conditional_edges("retrieve", stop_if_errors,
                            {"go": "rerank", "stop": END})
    g.add_edge("rerank", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "retry": "rewrite_query",
        "refuse": "refuse",
    })
    g.add_edge("rewrite_query", "retrieve")

    g.add_edge("generate", END)
    g.add_edge("refuse", END)

    return g.compile(checkpointer=MemorySaver())