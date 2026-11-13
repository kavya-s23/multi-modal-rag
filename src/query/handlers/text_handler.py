import time
import unicodedata

from ..schema import InputType, QueryObject

# Curly quotes and long dashes become plain ones; invisible characters are deleted.
#look up table
_REPLACE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-",
    "\u200b": None,   # zero-width space
    "\u2060": None,   # word joiner
    "\ufeff": None,   # byte-order mark
    "\u00ad": None,   # soft hyphen
})

_WHITESPACE_CONTROLS = "\t\n\r"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)   # one standard Unicode form
    text = text.translate(_REPLACE)              # quotes, dashes, invisible chars
    text = "".join(                              # drop other control characters
        ch for ch in text
        if ch in _WHITESPACE_CONTROLS or unicodedata.category(ch) != "Cc"
    )
    return " ".join(text.split())                # collapse all whitespace


def handle_text(q: QueryObject) -> QueryObject:
    start = time.perf_counter()

    if q.input_type == InputType.AUDIO:
        source = q.transcript
    elif q.input_type in (InputType.TEXT, InputType.TEXT_IMAGE):
        source = q.original_text
    else:                       # image-only: no typed text to clean
        source = None

    if source:
        q.cleaned_text = clean_text(source)

    q.timings["text_handler"] = (time.perf_counter() - start) * 1000
    return q