from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from app.data.loaders.cameo import load_cameo_split, corpus_from_path, CAMEO_LANGUAGES


def _cameo_split(tmp_path, name, rows):
    table = pa.table({
        "file_id": [r["file_id"] for r in rows],
        "audio": pa.array([{"bytes": b"x", "path": None} for _ in rows],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "emotion": [r["emotion"] for r in rows],
        "transcription": [r["tr"] for r in rows],
        "speaker_id": [r["spk"] for r in rows],
        "gender": [r["g"] for r in rows],
        "age": [r["age"] for r in rows],
        "dataset": [r["ds"] for r in rows],
        "language": [r["lang"] for r in rows],
        "license": ["CC" for _ in rows],
    })
    p = tmp_path / name
    pq.write_table(table, p)
    return p


def test_corpus_from_path():
    assert corpus_from_path(Path("x/ravdess-00000-of-00001.parquet")) == "ravdess"
    assert corpus_from_path(Path("x/crema_d-00000-of-00001.parquet")) == "crema_d"


def test_load_cameo_split_maps_fields_and_language(tmp_path):
    p = _cameo_split(tmp_path, "ravdess-00000-of-00001.parquet", [
        {"file_id": "hash1.flac", "emotion": "calm", "tr": "Kids", "spk": "ravdess_1",
         "g": "male", "age": "29", "ds": "RAVDESS", "lang": "English"},
        {"file_id": "hash2.flac", "emotion": "anger", "tr": "Dogs", "spk": "ravdess_2",
         "g": "female", "age": "23", "ds": "RAVDESS", "lang": "English"},
    ])
    samples = load_cameo_split(p)
    assert [s.sample_id for s in samples] == ["ravdess:hash1.flac", "ravdess:hash2.flac"]
    s0 = samples[0]
    assert s0.studies == [2] and s0.corpus == "ravdess" and s0.language == "en"
    assert s0.text == "Kids" and s0.gold_raw == "calm" and s0.gold is None
    assert s0.audio.parquet_path == p and s0.audio.row_index == 0


def test_unknown_language_raises(tmp_path):
    p = _cameo_split(tmp_path, "x-00000-of-00001.parquet", [
        {"file_id": "h.flac", "emotion": "anger", "tr": "t", "spk": "s",
         "g": "male", "age": "1", "ds": "X", "lang": "Klingon"},
    ])
    with pytest.raises(KeyError):
        load_cameo_split(p)
