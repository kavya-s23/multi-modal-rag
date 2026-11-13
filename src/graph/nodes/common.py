from functools import lru_cache

from src.query.schema import QueryObject
from src.query.validation import load_settings


@lru_cache(maxsize=1)
def get_cfg() -> dict:
    """Read settings.yaml once for the whole graph."""
    return load_settings()


def load_q(state: dict) -> QueryObject:
    return QueryObject.model_validate(state["qobj"])


def dump_q(q: QueryObject) -> dict:
    return q.model_dump(mode="json", exclude={"is_valid"})


def working_text(q: QueryObject) -> str:
    """The most processed version of the question so far."""
    return (q.final_text or q.rewritten_text or q.corrected_text
            or q.cleaned_text or "")