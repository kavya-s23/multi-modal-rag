#importing libs
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


class InputType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT_IMAGE = "text_image"


class QueryObject(BaseModel):
    # Identity
    query_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str | None = None
    session_id: str | None = None

    # Scope
    document_ids: list[str] = Field(min_length=1)
    source_id: str | None = None

    # Original input (never modified after creation)
    input_type: InputType
    original_text: str | None = None
    audio_path: Path | None = None
    image_path: Path | None = None

    # Audio
    transcript: str | None = None
    stt_language: str | None = None
    transcript_confirmed: bool = False

    # Text processing
    cleaned_text: str | None = None
    corrected_text: str | None = None
    was_corrected: bool = False
    corrections: list[tuple[str, str]] = Field(default_factory=list)
    rewritten_text: str | None = None
    final_text: str | None = None

    # Image details
    image_format: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_caption: str | None = None
    image_ocr_text: str | None = None
    image_embedding: list[float] | None = None

    # Status
    errors: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)

    @computed_field
    @property
    def is_valid(self) -> bool:
        return not self.errors

    @model_validator(mode="after")
    def _check_input_matches_type(self) -> "QueryObject":
        has_text = bool(self.original_text and self.original_text.strip())
        t = self.input_type
        if t == InputType.TEXT and not has_text:
            raise ValueError("input_type 'text' requires original_text")
        if t == InputType.AUDIO and self.audio_path is None:
            raise ValueError("input_type 'audio' requires audio_path")
        if t == InputType.IMAGE and self.image_path is None:
            raise ValueError("input_type 'image' requires image_path")
        if t == InputType.TEXT_IMAGE and not (has_text and self.image_path):
            raise ValueError(
                "input_type 'text_image' requires original_text and image_path")
        return self