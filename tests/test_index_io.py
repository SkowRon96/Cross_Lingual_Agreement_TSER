from pathlib import Path
from app.config.emotions import Emotion
from app.data.schema import Sample, AudioRef
from app.data.index_io import write_index, read_index


def _samples():
    return [
        Sample(sample_id="nEMO:a", studies=[1], corpus="nEMO", language="pl",
               text="tekst", audio=AudioRef(parquet_path=Path("d/x.parquet"), row_index=5),
               gold_raw="surprised", gold=Emotion.SURPRISE, speaker_id="WR0",
               gender="male", age="23"),
        Sample(sample_id="jl:b", studies=[2], corpus="jl_corpus", language="en",
               text="hello", audio=AudioRef(wav_path=Path("d/b.wav")),
               gold_raw="apology", gold=None),  # dropped label
    ]


def test_roundtrip_preserves_everything(tmp_path):
    out = write_index(_samples(), tmp_path / "nEMO.parquet")
    got = read_index(out)
    assert [s.sample_id for s in got] == ["nEMO:a", "jl:b"]
    a, b = got
    assert a.gold is Emotion.SURPRISE
    assert a.studies == [1]
    assert a.audio.parquet_path == Path("d/x.parquet") and a.audio.row_index == 5
    assert a.speaker_id == "WR0"
    assert b.gold is None            # dropped label round-trips as None
    assert b.audio.wav_path == Path("d/b.wav") and b.audio.parquet_path is None
