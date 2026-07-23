"""Audio decoding: encoded bytes / wav file -> float32 mono waveform + sample rate."""
import io

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from app.data.schema import Sample


def decode_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:  # downmix to mono
        data = data.mean(axis=1).astype("float32")
    return np.ascontiguousarray(data, dtype="float32"), int(sr)


def _read_parquet_audio_bytes(parquet_path, row_index: int) -> bytes:
    pf = pq.ParquetFile(parquet_path)
    remaining = row_index
    # Iterate through row groups, decrementing remaining until we find the group
    # containing the target row, then index into that group's audio column.
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if remaining < n:
            col = pf.read_row_group(rg, columns=["audio"]).column("audio")
            struct = col[remaining].as_py()
            raw = struct["bytes"]
            if raw is None:
                raise ValueError(f"row {row_index} in {parquet_path} has no embedded audio bytes")
            return raw
        remaining -= n
    raise IndexError(f"row_index {row_index} out of range for {parquet_path}")


def load_audio(sample: Sample) -> tuple[np.ndarray, int]:
    ref = sample.audio
    if ref.wav_path is not None:
        return decode_audio_bytes(ref.wav_path.read_bytes())
    raw = _read_parquet_audio_bytes(ref.parquet_path, ref.row_index)
    return decode_audio_bytes(raw)
