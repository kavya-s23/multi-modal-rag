import time

import numpy as np

from ...models import stt
from ..schema import InputType, QueryObject


def _handle_audio(q: QueryObject, cfg: dict) -> QueryObject:
    # Only audio queries, only if validation passed, and never re-transcribe.
    if q.input_type != InputType.AUDIO or q.errors or q.transcript:
        return q

    acfg = cfg["audio"]

    try:
        audio = stt.load_audio(q.audio_path)
    except RuntimeError:
        q.errors.append(
            "The audio file could not be read. It may be corrupt or not real audio.")
        return q

    duration = len(audio) / stt.SAMPLE_RATE
    if duration == 0:
        q.errors.append("The audio file is empty.")
        return q
    if duration > acfg["max_duration_seconds"]:
        q.errors.append(f"The audio is {duration:.0f} seconds; "
                        f"the limit is {acfg['max_duration_seconds']} seconds.")
        return q

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < acfg["min_rms"]:
        q.errors.append("The audio is silent or too quiet to transcribe.")
        return q

    text, language = stt.transcribe(audio, acfg["stt_model"], acfg["language"])
    if not text:
        q.errors.append("No speech was detected in the audio.")
        return q

    q.transcript = text
    q.stt_language = language
    q.transcript_confirmed = False      # the user must approve before searching
    return q


def handle_audio(q: QueryObject, cfg: dict) -> QueryObject:
    start = time.perf_counter()
    try:
        return _handle_audio(q, cfg)
    finally:
        q.timings["audio_handler"] = (time.perf_counter() - start) * 1000