"""Normalized data contract: AudioRef and Sample."""
from pathlib import Path

from pydantic import BaseModel, model_validator

from app.config.emotions import Emotion


class AudioRef(BaseModel):
    """Reference to audio, decoded on demand. Either a parquet row or a wav file."""
    parquet_path: Path | None = None
    row_index: int | None = None
    wav_path: Path | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "AudioRef":
        has_parquet = self.parquet_path is not None
        has_wav = self.wav_path is not None
        if has_parquet == has_wav:
            raise ValueError("AudioRef needs exactly one source: parquet row OR wav file")
        if has_parquet and self.row_index is None:
            raise ValueError("parquet_path requires row_index")
        if has_wav and self.row_index is not None:
            raise ValueError("row_index is only valid with parquet_path")
        return self


class Sample(BaseModel):
    """One utterance normalized across corpora."""
    sample_id: str
    studies: list[int]
    corpus: str
    language: str
    text: str
    audio: AudioRef
    has_face: bool = False
    video_path: Path | None = None
    gold_raw: str
    gold: Emotion | None  # None = dropped (label outside canonical space)
    speaker_id: str | None = None
    gender: str | None = None
    age: str | None = None
    intensity: str | None = None
