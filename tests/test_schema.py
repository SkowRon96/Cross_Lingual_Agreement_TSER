from pathlib import Path
import pytest
from pydantic import ValidationError
from app.config.emotions import Emotion
from app.data.schema import AudioRef, Sample


def test_audioref_from_parquet_row():
    ref = AudioRef(parquet_path=Path("datasets/nEMO/data/train-00000.parquet"), row_index=3)
    assert ref.row_index == 3
    assert ref.wav_path is None


def test_audioref_from_wav():
    ref = AudioRef(wav_path=Path("a.wav"))
    assert ref.parquet_path is None


def test_audioref_requires_exactly_one_source():
    with pytest.raises(ValidationError):
        AudioRef()  # no source
    with pytest.raises(ValidationError):
        AudioRef(wav_path=Path("a.wav"), parquet_path=Path("b.parquet"), row_index=0)  # two sources
    with pytest.raises(ValidationError):
        AudioRef(parquet_path=Path("b.parquet"))  # parquet without row_index
    with pytest.raises(ValidationError):
        AudioRef(wav_path=Path("a.wav"), row_index=0)  # row_index without parquet_path


def test_sample_minimal_valid():
    s = Sample(
        sample_id="nEMO:WR0_surprised_64",
        studies=[1],
        corpus="nEMO",
        language="pl",
        text="Kto to zrobil",
        audio=AudioRef(wav_path=Path("a.wav")),
        gold_raw="surprised",
        gold=Emotion.SURPRISE,
    )
    assert s.gold is Emotion.SURPRISE
    assert s.has_face is False


def test_sample_gold_may_be_none_when_dropped():
    s = Sample(
        sample_id="CAMEO:xyz",
        studies=[2],
        corpus="jl_corpus",
        language="en",
        text="hello",
        audio=AudioRef(wav_path=Path("a.wav")),
        gold_raw="apology",
        gold=None,
    )
    assert s.gold is None
