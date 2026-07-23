import io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf


def _wav_bytes(freq: float = 220.0, sr: int = 16000, seconds: float = 0.1) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * freq * t).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return _wav_bytes


@pytest.fixture
def audio_parquet(tmp_path):
    """A tiny parquet mimicking the nEMO/CAMEO audio struct column layout."""
    def _make(rows: list[dict], filename: str = "mini.parquet"):
        audio_struct = [{"bytes": r["audio_bytes"], "path": None} for r in rows]
        table = pa.table({
            "file_id": [r["file_id"] for r in rows],
            "audio": pa.array(audio_struct, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            "emotion": [r["emotion"] for r in rows],
        })
        path = tmp_path / filename
        pq.write_table(table, path)
        return path
    return _make
