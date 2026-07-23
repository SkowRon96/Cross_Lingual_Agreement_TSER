from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from app.data.loaders.nemo import load_nemo


def _nemo_shard(tmp_path, name, rows):
    table = pa.table({
        "file_id": [r["file_id"] for r in rows],
        "audio": pa.array([{"bytes": b"x", "path": None} for _ in rows],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "emotion": [r["emotion"] for r in rows],
        "raw_text": [r["raw"] for r in rows],
        "normalized_text": [r["norm"] for r in rows],
        "speaker_id": [r["spk"] for r in rows],
        "gender": [r["g"] for r in rows],
        "age": [r["age"] for r in rows],
    })
    p = tmp_path / name
    pq.write_table(table, p)
    return p


def test_load_nemo_maps_fields_and_local_row_index(tmp_path):
    p0 = _nemo_shard(tmp_path, "train-00000.parquet", [
        {"file_id": "WR0_surprised_64", "emotion": "surprised", "raw": "R", "norm": "n0",
         "spk": "WR0", "g": "male", "age": 23},
    ])
    p1 = _nemo_shard(tmp_path, "train-00001.parquet", [
        {"file_id": "EB0_neutral_13", "emotion": "neutral", "raw": "R", "norm": "n1",
         "spk": "EB0", "g": "female", "age": 27},
    ])
    samples = load_nemo([p0, p1])
    assert [s.sample_id for s in samples] == ["nEMO:WR0_surprised_64", "nEMO:EB0_neutral_13"]
    s0, s1 = samples
    assert s0.studies == [1] and s0.corpus == "nEMO" and s0.language == "pl"
    assert s0.text == "n0" and s0.gold_raw == "surprised" and s0.gold is None
    assert s0.audio.parquet_path == p0 and s0.audio.row_index == 0
    assert s1.audio.parquet_path == p1 and s1.audio.row_index == 0  # local to its shard
    assert s0.speaker_id == "WR0" and s0.age == "23"
