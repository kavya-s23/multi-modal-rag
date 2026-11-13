import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = None
_lock = threading.Lock()


def _get_client(timeout_seconds: int) -> genai.Client:
    global _client
    with _lock:
        if _client is None:
            _client = genai.Client(
                http_options=types.HttpOptions(timeout=timeout_seconds * 1000))
        return _client


def generate_text(prompt: str, model_name: str, timeout_seconds: int = 20) -> str:
    response = _get_client(timeout_seconds).models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return (response.text or "").strip()