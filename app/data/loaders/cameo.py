"""CAMEO (multilingual) loader: one split parquet -> Sample list (Study 2)."""
import re
from pathlib import Path

import pyarrow.parquet as pq

from app.data.schema import AudioRef, Sample

CAMEO_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Polish": "pl",
    "French": "fr",
    "Italian": "it",
    "Spanish": "es",
    "German": "de",
    "Russian": "ru",
    "Bengali": "bn",
}


def corpus_from_path(path: Path) -> str:
    # "ravdess-00000-of-00001.parquet" -> "ravdess"
    return re.sub(r"-\d+-of-\d+\.parquet$", "", Path(path).name)


def load_cameo_split(path: Path) -> list[Sample]:
    corpus = corpus_from_path(path)
    cols = pq.read_table(
        path,
        columns=["file_id", "emotion", "transcription", "speaker_id", "gender", "age", "language"],
    ).to_pydict()
    samples: list[Sample] = []
    for i in range(len(cols["file_id"])):
        language_full = cols["language"][i]
        iso = CAMEO_LANGUAGES[language_full]  # KeyError on unknown = fail loud
        samples.append(Sample(
            sample_id=f"{corpus}:{cols['file_id'][i]}",
            studies=[2],
            corpus=corpus,
            language=iso,
            text=cols["transcription"][i],
            audio=AudioRef(parquet_path=path, row_index=i),
            gold_raw=cols["emotion"][i],
            gold=None,
            speaker_id=cols["speaker_id"][i],
            gender=cols["gender"][i],
            age=str(cols["age"][i]),
        ))
    return samples
