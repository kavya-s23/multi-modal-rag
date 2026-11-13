import threading

import numpy as np
import whisper

SAMPLE_RATE = whisper.audio.SAMPLE_RATE   # 16000, the rate Whisper expects

_model = None
_model_name = None
_lock = threading.Lock()


def _get_model(name: str):
    """Load the model once, on first use, then reuse it."""
    global _model, _model_name
    with _lock:
        if _model is None or _model_name != name:
            _model = whisper.load_model(name)
            _model_name = name
    return _model


def load_audio(path) -> np.ndarray:
    """Decode any audio file (via ffmpeg) to a mono 16 kHz float array.
    Raises RuntimeError if the file cannot be decoded."""
    return whisper.load_audio(str(path))


def transcribe(audio: np.ndarray, model_name: str, language: str) -> tuple[str, str]:
    """Returns (text, language_code)."""
    result = _get_model(model_name).transcribe(audio, language=language)
    return result["text"].strip(), result["language"]