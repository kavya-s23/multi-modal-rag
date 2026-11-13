import re
import time

from ..models import llm_for_follow_up
from .schema import QueryObject

_STARTERS = ("what about", "how about", "and ", "also ", "then ", "why ", "so ")
_REFERENCES = re.compile(
    r"\b(it|that|this|those|these|they|them|he|she|his|her|there|"
    r"the first|the second|the third|the last|same)\b", re.I)

PROMPT = """Rewrite the user's LAST question as one complete, standalone question,
using the chat history only to fill in what it refers to.
Rules:
- Keep the original meaning. Do not answer the question.
- Do not add facts that are not in the chat history.
- If the last question is already complete, return it unchanged.
- Return only the question, nothing else.

CHAT HISTORY:
{history}

LAST QUESTION: {question}
STANDALONE QUESTION:"""


def _looks_like_followup(text: str, cfg: dict) -> bool:
    t = text.lower().strip()
    return (len(t.split()) <= cfg["followup"]["max_words_to_check"]
            or t.startswith(_STARTERS)
            or bool(_REFERENCES.search(t)))


def _rewrite(q: QueryObject, history: list[tuple[str, str]], cfg: dict) -> QueryObject:
    fcfg = cfg["followup"]
    base = q.corrected_text or q.cleaned_text
    if not base or not history or not _looks_like_followup(base, cfg):
        q.final_text = base
        return q

    turns = history[-fcfg["max_history_turns"]:]
    history_text = "\n".join(f"{role}: {text}" for role, text in turns)
    prompt = PROMPT.format(history=history_text, question=base)

    try:
        new = llm_for_follow_up.generate_text(prompt, fcfg["llm_model"],
                                       fcfg["timeout_seconds"])
    except Exception:
        q.final_text = base          # a failed rewrite must never block a query
        return q

    new = new.strip().strip('"')
    too_long = len(new) > cfg["query"]["max_query_chars"]
    if not new or too_long or new.lower() == base.lower():
        q.final_text = base
        return q

    q.rewritten_text = new
    q.final_text = new
    return q


def handle_followup(q: QueryObject, history: list[tuple[str, str]],
                    cfg: dict) -> QueryObject:
    start = time.perf_counter()
    try:
        if q.errors:
            return q
        return _rewrite(q, history, cfg)
    finally:
        q.timings["followup_rewriter"] = (time.perf_counter() - start) * 1000