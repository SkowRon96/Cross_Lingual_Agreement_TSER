"""Serialize a normalized Sample index to/from parquet (no audio bytes)."""
import json
from pathlib import Path

import pandas as pd

from app.config.emotions import Emotion
from app.data.schema import AudioRef, Sample


def samples_to_frame(samples: list[Sample]) -> pd.DataFrame:
    rows = []
    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "studies": json.dumps(s.studies),
            "corpus": s.corpus,
            "language": s.language,
            "text": s.text,
            "audio_parquet_path": str(s.audio.parquet_path) if s.audio.parquet_path else "",
            "audio_row_index": s.audio.row_index if s.audio.row_index is not None else -1,
            "audio_wav_path": str(s.audio.wav_path) if s.audio.wav_path else "",
            "has_face": s.has_face,
            "video_path": str(s.video_path) if s.video_path else "",
            "gold_raw": s.gold_raw,
            "gold": s.gold.value if s.gold is not None else "",
            "speaker_id": s.speaker_id or "",
            "gender": s.gender or "",
            "age": s.age or "",
            "intensity": s.intensity or "",
        })
    return pd.DataFrame(rows)


def write_index(samples: list[Sample], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples_to_frame(samples).to_parquet(path, index=False)
    return path


def _audio_ref(row: dict) -> AudioRef:
    if row["audio_wav_path"]:
        return AudioRef(wav_path=Path(row["audio_wav_path"]))
    return AudioRef(parquet_path=Path(row["audio_parquet_path"]), row_index=int(row["audio_row_index"]))


def read_index(path: Path) -> list[Sample]:
    df = pd.read_parquet(path)
    out = []
    for row in df.to_dict(orient="records"):
        out.append(Sample(
            sample_id=row["sample_id"],
            studies=json.loads(row["studies"]),
            corpus=row["corpus"],
            language=row["language"],
            text=row["text"],
            audio=_audio_ref(row),
            has_face=bool(row["has_face"]),
            video_path=Path(row["video_path"]) if row["video_path"] else None,
            gold_raw=row["gold_raw"],
            gold=Emotion(row["gold"]) if row["gold"] else None,
            speaker_id=row["speaker_id"] or None,
            gender=row["gender"] or None,
            age=row["age"] or None,
            intensity=row["intensity"] or None,
        ))
    return out
