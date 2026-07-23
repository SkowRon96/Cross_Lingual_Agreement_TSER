import io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
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


def test_load_audio_targets_correct_row_group(tmp_path, wav_bytes):
    """Multi-row-group parquet: each row in its own group with different durations."""
    rows = [
        wav_bytes(sr=16000, seconds=0.05),   # row 0 -> 800 samples
        wav_bytes(sr=16000, seconds=0.10),   # row 1 -> 1600 samples
        wav_bytes(sr=16000, seconds=0.15),   # row 2 -> 2400 samples
    ]
    table = pa.table({
        "file_id": ["a", "b", "c"],
        "audio": pa.array([{"bytes": b, "path": None} for b in rows],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "emotion": ["neutral", "anger", "fear"],
    })
    path = tmp_path / "multi.parquet"
    pq.write_table(table, path, row_group_size=1)
    assert pq.ParquetFile(path).num_row_groups == 3  # sanity: really multiple groups
    sample = Sample(
        sample_id="m:c", studies=[1], corpus="m", language="pl", text="x",
        audio=AudioRef(parquet_path=path, row_index=2),
        gold_raw="fear", gold=Emotion.FEAR,
    )
    arr, sr = load_audio(sample)
    assert sr == 16000 and arr.shape[0] == 2400  # got row 2 from the 3rd row group


def test_decode_downmixes_stereo_to_mono_mean(wav_bytes):
    """Stereo wav with distinct channels: decoded mono equals channel mean."""
    sr = 16000
    ch0 = np.full(400, 0.5, dtype="float32")
    ch1 = np.full(400, -0.1, dtype="float32")
    stereo = np.stack([ch0, ch1], axis=1)  # shape (400, 2)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="FLOAT")
    arr, got_sr = decode_audio_bytes(buf.getvalue())
    assert got_sr == sr
    assert arr.ndim == 1 and arr.shape[0] == 400
    assert np.allclose(arr, (ch0 + ch1) / 2, atol=1e-6)  # mean of channels = 0.2


def test_load_audio_raises_on_null_audio_bytes(tmp_path):
    import pytest
    table = pa.table({
        "file_id": ["a"],
        "audio": pa.array([{"bytes": None, "path": "external.wav"}],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "emotion": ["neutral"],
    })
    p = tmp_path / "nullbytes.parquet"
    pq.write_table(table, p)
    s = Sample(sample_id="x:a", studies=[1], corpus="x", language="pl", text="t",
               audio=AudioRef(parquet_path=p, row_index=0), gold_raw="neutral", gold=Emotion.NEUTRAL)
    with pytest.raises(ValueError):
        load_audio(s)
