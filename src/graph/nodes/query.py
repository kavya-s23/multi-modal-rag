from langgraph.types import interrupt
from pydantic import ValidationError

from src.graph.nodes.common import dump_q, get_cfg, load_q, working_text
from src.graph.state import RAGState
from src.query.follow_up_rewriter import handle_followup
from src.query.handlers.audio_handler import handle_audio
from src.query.handlers.image_handler import handle_image
from src.query.handlers.text_handler import clean_text
from src.query.schema import InputType, QueryObject
from src.query.validation import validate_query


def validate(state: RAGState) -> dict:
    raw = state["query"]
    modality = raw.get("modality", "text")
    text = (raw.get("text") or "").strip() or None

    input_type = {
        "audio": InputType.AUDIO,
        "image": InputType.TEXT_IMAGE,   # an image always comes with a typed question
    }.get(modality, InputType.TEXT)

    try:
        q = QueryObject(
            document_ids=state["doc_ids"],
            input_type=input_type,
            original_text=text,
            audio_path=raw.get("audio_path") or None,
            image_path=raw.get("image_path") or None,
        )
    except ValidationError as e:
        return {"errors": [err["msg"] for err in e.errors()]}

    q = validate_query(q, get_cfg())
    return {"qobj": dump_q(q), "errors": q.errors}


def transcribe(state: RAGState) -> dict:
    q = handle_audio(load_q(state), get_cfg())
    return {"qobj": dump_q(q), "errors": q.errors, "question": q.transcript or ""}


def confirm_transcript(state: RAGState) -> dict:
    # Keep this node cheap: it re-runs from the top when the graph resumes.
    q = load_q(state)
    answer = interrupt(q.transcript)          # freezes until the user confirms
    if not answer.get("confirmed"):
        return {"errors": ["Transcript not confirmed. Nothing was searched."]}
    if answer.get("text"):
        q.transcript = answer["text"]         # the user's corrected version
    q.transcript_confirmed = True
    return {"qobj": dump_q(q), "transcript_confirmed": True,
            "question": q.transcript or ""}


def caption_image(state: RAGState) -> dict:
    q = handle_image(load_q(state), get_cfg())
    print(f"[image] caption={q.image_caption!r} ocr={q.image_ocr_text!r} errors={q.errors}")  # temporary
    return {"qobj": dump_q(q), "errors": q.errors}


def clean(state: RAGState) -> dict:
    q = load_q(state)
    source = q.transcript if q.input_type == InputType.AUDIO else q.original_text
    q.cleaned_text = clean_text(source or "")
    if not q.cleaned_text:
        q.errors.append("The question is empty after cleaning.")
        return {"qobj": dump_q(q), "errors": q.errors, "question": ""}

    question = working_text(q)
    if q.input_type in (InputType.IMAGE, InputType.TEXT_IMAGE):
        seen = []
        if q.image_caption:                          # what the picture shows
            seen.append(clean_text(q.image_caption)[:250])
        if q.image_ocr_text:                         # words and numbers inside it
            seen.append(clean_text(q.image_ocr_text)[:300])
        # Image content goes first: a typed question like "what does this chart say"
        # carries no searchable content, while the image description does.
        question = " ".join(seen + [question]).strip()
    return {"qobj": dump_q(q), "errors": q.errors, "question": question}


def rewrite_followup(state: RAGState) -> dict:
    # handle_followup wants (question, answer) tuples; the state keeps dicts.
    history = [(h["question"], h["answer"]) for h in state.get("history", [])]
    q = handle_followup(load_q(state), history, get_cfg())
    return {"qobj": dump_q(q), "errors": q.errors, "question": working_text(q)}