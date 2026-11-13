import os
import re
import threading

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

_client = None
_lock = threading.Lock()


def _get_client(timeout_seconds: int) -> OpenAI:
    global _client
    with _lock:
        if _client is None:
            key = os.environ.get("NVIDIA_API_KEY")
            if not key:
                raise RuntimeError("NVIDIA_API_KEY is missing from .env")
            _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=timeout_seconds)
        return _client


def generate_text_nvidia(prompt: str, model_name: str, timeout_seconds: int = 30) -> str:
    response = _get_client(timeout_seconds).chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # drop printed reasoning