import os
import re
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

from src.models.llm_for_follow_up import generate_text

load_dotenv()

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"

_nvidia = None
_lock = threading.Lock()


def _nvidia_client(timeout: int) -> OpenAI:
    global _nvidia
    with _lock:
        if _nvidia is None:
            key = os.getenv("NVIDIA_API_KEY", "").strip()
            if not key:
                raise RuntimeError("NVIDIA_API_KEY is missing from .env")
            _nvidia = OpenAI(base_url=NVIDIA_URL, api_key=key, timeout=timeout)
        return _nvidia


def _ask_nvidia(prompt: str, model: str, timeout: int) -> str:
    response = _nvidia_client(timeout).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # drop printed reasoning


def _retryable(e: Exception) -> bool:
    msg = str(e)
    return any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def ask(prompt: str, cfg: dict, rounds: int = 3) -> str:
    """Gemini first. On a rate limit or overload, try NVIDIA, then wait and repeat."""
    lcfg = cfg["llm"]
    timeout = lcfg["timeout_seconds"]
    attempts = [lambda: generate_text(prompt, lcfg["model"], timeout)]
    if lcfg.get("fallback_model"):
        attempts.append(lambda: _ask_nvidia(prompt, lcfg["fallback_model"], timeout))

    last = None
    for r in range(rounds):
        for call in attempts:
            try:
                return call()
            except Exception as e:
                if not _retryable(e):
                    raise                    # bad key, wrong model name: fail at once
                last = e
        time.sleep(10 * (r + 1))             # both are busy: wait 10s, 20s
    raise last