from src.generation.generator import generate_answer
from src.graph.nodes.common import get_cfg, load_q
from src.graph.state import RAGState


def generate(state: RAGState) -> dict:
    q = load_q(state)
    # The user's own wording (or the follow-up rewrite), not the grader's search rewrite.
    question = q.rewritten_text or q.cleaned_text or state["question"]

    note = ""
    if q.image_caption or q.image_ocr_text:
        note = ("The user also attached an image. "
                f"Description: {q.image_caption or 'none'}. "
                f"Text in the image: {q.image_ocr_text or 'none'}.")

    try:
        result = generate_answer(question, state.get("ranked", []), get_cfg(), note)
    except Exception as e:                          # API down, bad JSON twice
        return {"errors": [f"The answer could not be generated ({type(e).__name__}: {e})."]}

    if not result["citations"]:                     # the model found no supported answer
        return {"refused": True, "answer": "", "citations": []}

    history = state.get("history", []) + [
        {"question": question, "answer": result["plain"]}]
    return {"answer": result["answer"], "citations": result["citations"],
            "refused": False, "history": history}