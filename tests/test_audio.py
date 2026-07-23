from pathlib import Path
import numpy as np
from app.data.schema import Sample, AudioRef
from app.config.emotions import Emotion
from app.data.audio import decode_audio_bytes, load_audio


def test_decode_bytes_returns_float_mono_and_sr(wav_bytes):
    arr, sr = decode_audio_bytes(wav_bytes(freq=330.0, sr=16000, seconds=0.1))
    assert sr == 16000
    assert arr.ndim == 1
    assert arr.dtype == np.float32
    assert arr.shape[0] == 1600  # 0.1s * 16000


def test_load_audio_from_parquet_row(audio_parquet, wav_bytes):
    path = audio_parquet([
        {"file_id": "a", "audio_bytes": wav_bytes(sr=24000, seconds=0.05), "emotion": "neutral"},
        {"file_id": "b", "audio_bytes": wav_bytes(sr=24000, seconds=0.05), "emotion": "anger"},
    ])
    sample = Sample(
        sample_id="mini:b", studies=[1], corpus="mini", language="pl",
        text="x", audio=AudioRef(parquet_path=path, row_index=1),
        gold_raw="anger", gold=Emotion.ANGER,
    )
    arr, sr = load_audio(sample)
    assert sr == 24000
    assert arr.shape[0] == 1200  # 0.05s * 24000


def test_load_audio_from_wav_file(tmp_path, wav_bytes):
    p = tmp_path / "x.wav"
    p.write_bytes(wav_bytes(sr=16000, seconds=0.05))
    sample = Sample(
        sample_id="w:1", studies=[1], corpus="w", language="en",
        text="x", audio=AudioRef(wav_path=p), gold_raw="neutral", gold=Emotion.NEUTRAL,
    )
    arr, sr = load_audio(sample)
    assert sr == 16000 and arr.shape[0] == 800
