# P2 — Ingest (nEMO + CAMEO) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Znormalizować zbiory nEMO (PL) i CAMEO (8 języków) do jednolitego indeksu `Sample` z zharmonizowaną etykietą gold, z dekodowaniem audio na żądanie i raportem harmonizacji.

**Architecture:** Etap `ingest` z Podejścia A. Loadery czytają źródłowe parquet (audio osadzone jako `struct<bytes,path>`), budują listy `Sample` (P1), mapują etykiety gold przez `map_label` (P1, single source of truth, fail-loud), i zapisują lekki indeks `results/index/{corpus}.parquet` (bez kopiowania bajtów audio). Audio dekodowane na żądanie z parquet.

**Tech Stack:** Python 3.12, Pydantic v2, pandas + pyarrow, **soundfile** (nowa zależność — dekodowanie audio), pytest.

## Global Constraints

- Python **3.12**; jeden `.venv` w korzeniu; Mac M1, bez GPU.
- **Pydantic v2** dla konfiguracji i modeli; buduje na P1 (`Emotion`, `map_label`, `Sample`, `AudioRef`).
- Etykiety gold mapowane **wyłącznie** przez `app.labels.mapping.map_label` — nieznana etykieta ma **rzucać** (fail-loud, zero zmyślania). Mapowania scaleń/DROP już zdefiniowane w P1.
- Kanoniczne emocje (7): `anger, disgust, fear, happiness, sadness, surprise, neutral`.
- Kod/testy/commit messages po angielsku; **żadnego `Co-Authored-By`**.
- `datasets/`, `models/`, `old_projects/` nietykalne i niewersjonowane; indeksy lądują w `results/index/` (wersjonowane).
- Audio **nie jest kopiowane** — indeks trzyma referencję `(parquet_path, row_index)`; dekodowanie na żądanie.
- **studies:** standalone nEMO → `[1]`; wszystkie splity CAMEO → `[2]`.

## Review mode (per memory: economical-subagent-execution)

Ingest karmi wyniki → **pełna recenzja per-task dla Zadań 2–6** (audio decode, index I/O, loadery, harmonizacja gold). **Zadanie 1 (settings) — wiring neutralny dla wyników → bez recenzenta per-task, pokryte finalnym review.** Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus.

---

## File Structure

```
app/
  config/settings.py         # Pydantic Settings: dataset/results paths, seed, glob helpers
  data/
    audio.py                 # decode_audio_bytes(); load_audio(sample) -> (np.ndarray, sr)
    index_io.py              # samples_to_frame(); write_index(); read_index()
    ingest.py                # ingest_corpus(): loader -> harmonize gold -> write index + report
    loaders/
      __init__.py
      nemo.py                # load_nemo() -> list[Sample] (raw gold, gold unresolved)
      cameo.py               # load_cameo_split(); CAMEO_LANGUAGES; corpus_from_path()
tests/
  test_settings.py
  test_audio.py
  test_index_io.py
  test_loader_nemo.py
  test_loader_cameo.py
  test_ingest.py
  conftest.py                # fixtures: tiny in-memory parquet with an audio struct column
requirements.txt             # += soundfile
```

Interfejsy (kontrakt utrwalony tutaj):
- `app.config.settings.Settings` — `.datasets_root`, `.results_root`, `.index_dir`, `.nemo_parquets() -> list[Path]`, `.cameo_parquets() -> list[Path]`, `.seed`.
- `app.data.audio.decode_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]`; `load_audio(sample: Sample) -> tuple[np.ndarray, int]`.
- `app.data.index_io.samples_to_frame(samples) -> pd.DataFrame`; `write_index(samples, path)`; `read_index(path) -> list[Sample]`.
- `app.data.loaders.nemo.load_nemo(parquets: list[Path]) -> list[Sample]` (gold left as raw string in `gold_raw`; `gold` resolved by ingest).
- `app.data.loaders.cameo.load_cameo_split(path: Path) -> list[Sample]`; `CAMEO_LANGUAGES: dict[str,str]`; `corpus_from_path(path) -> str`.
- `app.data.ingest.ingest_corpus(samples, corpus, out_dir) -> IngestReport` (resolves gold via map_label, writes index, returns counts).

**Loader/ingest split of responsibility:** loaders produce `Sample` objects with `gold_raw` set to the original dataset label and `gold=None` (unresolved placeholder). `ingest_corpus` resolves `gold = map_label(gold_raw)` for every sample, counting kept/merged/dropped, and writes the index. This keeps the single map_label call site in one place (ingest) and makes the harmonization report trivial.

---

### Task 1: Settings (paths & config)  [economize: no per-task reviewer]

**Files:**
- Create: `app/config/settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (pydantic-settings) with `datasets_root: Path`, `results_root: Path`, `seed: int = 42`; property `index_dir -> results_root/"index"`; methods `nemo_parquets() -> list[Path]` (sorted glob of `datasets_root/"nEMO"/"data"/"*.parquet"`) and `cameo_parquets() -> list[Path]` (sorted glob of `datasets_root/"CAMEO"/"data"/"*.parquet"`). Defaults resolve relative to repo root; overridable via `TSER_` env vars.

- [ ] **Step 1: Write the failing test**

Create `tests/test_settings.py`:
```python
from pathlib import Path
from app.config.settings import Settings


def test_defaults_point_into_repo(tmp_path):
    s = Settings(datasets_root=tmp_path / "datasets", results_root=tmp_path / "results")
    assert s.index_dir == tmp_path / "results" / "index"
    assert s.seed == 42


def test_nemo_and_cameo_globs(tmp_path):
    (tmp_path / "datasets" / "nEMO" / "data").mkdir(parents=True)
    (tmp_path / "datasets" / "CAMEO" / "data").mkdir(parents=True)
    (tmp_path / "datasets" / "nEMO" / "data" / "train-00000.parquet").touch()
    (tmp_path / "datasets" / "nEMO" / "data" / "train-00001.parquet").touch()
    (tmp_path / "datasets" / "CAMEO" / "data" / "nemo-00000.parquet").touch()
    s = Settings(datasets_root=tmp_path / "datasets", results_root=tmp_path / "results")
    assert [p.name for p in s.nemo_parquets()] == ["train-00000.parquet", "train-00001.parquet"]
    assert [p.name for p in s.cameo_parquets()] == ["nemo-00000.parquet"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config.settings'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/config/settings.py`:
```python
"""Project settings: dataset/results paths and run config."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TSER_")

    datasets_root: Path = Field(default=_REPO_ROOT / "datasets")
    results_root: Path = Field(default=_REPO_ROOT / "results")
    seed: int = 42

    @property
    def index_dir(self) -> Path:
        return self.results_root / "index"

    def nemo_parquets(self) -> list[Path]:
        return sorted((self.datasets_root / "nEMO" / "data").glob("*.parquet"))

    def cameo_parquets(self) -> list[Path]:
        return sorted((self.datasets_root / "CAMEO" / "data").glob("*.parquet"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/config/settings.py tests/test_settings.py
git commit -m "feat: project Settings with dataset/results paths"
```

---

### Task 2: Audio decode (bytes -> waveform)

**Files:**
- Modify: `requirements.txt` (add `soundfile`)
- Create: `app/data/audio.py`, `tests/conftest.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `Sample`, `AudioRef` (P1).
- Produces:
  - `decode_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]` — decode encoded audio (wav/flac) to a 1-D float32 mono array + sample rate; downmix multi-channel by averaging.
  - `load_audio(sample: Sample) -> tuple[np.ndarray, int]` — resolve `sample.audio`: if `wav_path`, read the file; if parquet ref, read exactly the referenced row's `audio.bytes` from that parquet (row-group targeted, not whole-column) and decode.
  - Note: NO resampling here — returns native sr; model stage (P3) resamples to its target.

- [ ] **Step 1: Add soundfile to requirements and install**

Append to `requirements.txt`:
```
soundfile>=0.12,<0.14
```
Run: `.venv/bin/pip install -r requirements.txt` then `.venv/bin/python -c "import soundfile; print('ok')"`
Expected: `ok`.

- [ ] **Step 2: Write shared fixtures**

Create `tests/conftest.py`:
```python
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
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_audio.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data.audio'`.

- [ ] **Step 5: Write minimal implementation**

Create `app/data/audio.py`:
```python
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
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if remaining < n:
            col = pf.read_row_group(rg, columns=["audio"]).column("audio")
            struct = col[remaining].as_py()
            return struct["bytes"]
        remaining -= n
    raise IndexError(f"row_index {row_index} out of range for {parquet_path}")


def load_audio(sample: Sample) -> tuple[np.ndarray, int]:
    ref = sample.audio
    if ref.wav_path is not None:
        return decode_audio_bytes(ref.wav_path.read_bytes())
    raw = _read_parquet_audio_bytes(ref.parquet_path, ref.row_index)
    return decode_audio_bytes(raw)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_audio.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt app/data/audio.py tests/conftest.py tests/test_audio.py
git commit -m "feat: on-demand audio decoding from parquet/wav"
```

---

### Task 3: Index I/O (Sample <-> parquet round-trip)

**Files:**
- Create: `app/data/index_io.py`
- Test: `tests/test_index_io.py`

**Interfaces:**
- Consumes: `Sample`, `AudioRef`, `Emotion`.
- Produces:
  - `samples_to_frame(samples: list[Sample]) -> pd.DataFrame` — flattens AudioRef into `audio_parquet_path`/`audio_row_index`/`audio_wav_path` columns; `studies` as JSON string; `gold` as canonical string or empty.
  - `write_index(samples: list[Sample], path: Path) -> Path`.
  - `read_index(path: Path) -> list[Sample]` — reconstructs `Sample` (round-trip fidelity incl. `gold=None`, list `studies`, and the correct AudioRef source).

- [ ] **Step 1: Write the failing test**

Create `tests/test_index_io.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_index_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data.index_io'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/data/index_io.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_index_io.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add app/data/index_io.py tests/test_index_io.py
git commit -m "feat: Sample index parquet serialization"
```

---

### Task 4: nEMO loader

**Files:**
- Create: `app/data/loaders/__init__.py`, `app/data/loaders/nemo.py`
- Test: `tests/test_loader_nemo.py`

**Interfaces:**
- Consumes: `Sample`, `AudioRef`.
- Produces: `load_nemo(parquets: list[Path]) -> list[Sample]` — for each shard, for each row: `sample_id=f"nEMO:{file_id}"`, `studies=[1]`, `corpus="nEMO"`, `language="pl"`, `text=normalized_text`, `audio=AudioRef(parquet_path=shard, row_index=local_row)`, `gold_raw=emotion`, `gold=None` (resolved later by ingest), plus `speaker_id/gender/age` (age → str). Row index is LOCAL to its shard.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loader_nemo.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loader_nemo.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `app/data/loaders/__init__.py`:
```python
"""Dataset loaders producing normalized Sample lists."""
```

Create `app/data/loaders/nemo.py`:
```python
"""nEMO (Polish) loader: standalone HF parquet shards -> Sample list (Study 1)."""
from pathlib import Path

import pyarrow.parquet as pq

from app.data.schema import AudioRef, Sample


def load_nemo(parquets: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for shard in parquets:
        cols = pq.read_table(
            shard,
            columns=["file_id", "emotion", "normalized_text", "speaker_id", "gender", "age"],
        ).to_pydict()
        n = len(cols["file_id"])
        for i in range(n):
            samples.append(Sample(
                sample_id=f"nEMO:{cols['file_id'][i]}",
                studies=[1],
                corpus="nEMO",
                language="pl",
                text=cols["normalized_text"][i],
                audio=AudioRef(parquet_path=shard, row_index=i),
                gold_raw=cols["emotion"][i],
                gold=None,
                speaker_id=cols["speaker_id"][i],
                gender=cols["gender"][i],
                age=str(cols["age"][i]),
            ))
    return samples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loader_nemo.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add app/data/loaders/__init__.py app/data/loaders/nemo.py tests/test_loader_nemo.py
git commit -m "feat: nEMO loader"
```

---

### Task 5: CAMEO loader

**Files:**
- Create: `app/data/loaders/cameo.py`
- Test: `tests/test_loader_cameo.py`

**Interfaces:**
- Consumes: `Sample`, `AudioRef`.
- Produces:
  - `CAMEO_LANGUAGES: dict[str, str]` — full language name -> ISO code (English→en, Polish→pl, French→fr, Italian→it, Spanish→es, German→de, Russian→ru, Bengali→bn).
  - `corpus_from_path(path: Path) -> str` — split name from filename, i.e. the part before `-00000-of-...` (e.g. `ravdess-00000-of-00001.parquet` -> `"ravdess"`).
  - `load_cameo_split(path: Path) -> list[Sample]` — for each row: `sample_id=f"{corpus}:{file_id}"`, `studies=[2]`, `corpus=<split>`, `language=CAMEO_LANGUAGES[language_full]` (raise KeyError on unknown language — fail loud), `text=transcription`, `audio=AudioRef(parquet_path=path, row_index=i)`, `gold_raw=emotion`, `gold=None`, `speaker_id/gender/age` (age already str).

- [ ] **Step 1: Write the failing test**

Create `tests/test_loader_cameo.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loader_cameo.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `app/data/loaders/cameo.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loader_cameo.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/data/loaders/cameo.py tests/test_loader_cameo.py
git commit -m "feat: CAMEO split loader with language mapping"
```

---

### Task 6: Ingest orchestrator + harmonization report

**Files:**
- Create: `app/data/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Sample`, `Emotion`, `map_label` (P1), `write_index` (Task 3).
- Produces:
  - `IngestReport` (Pydantic): `corpus: str`, `total: int`, `kept: int`, `dropped: int`, `per_raw_label: dict[str, str]` (raw label -> canonical value or `"DROP"`), `dropped_labels: dict[str, int]` (raw label -> count of dropped samples).
  - `resolve_gold(samples) -> list[Sample]` — returns new Samples with `gold = map_label(s.gold_raw)` (raises on unknown label — fail loud).
  - `ingest_corpus(samples: list[Sample], corpus: str, out_dir: Path) -> IngestReport` — resolves gold, writes `out_dir/f"{corpus}.parquet"` via `write_index`, returns the report.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:
```python
from pathlib import Path
from app.config.emotions import Emotion
from app.data.schema import Sample, AudioRef
from app.data.ingest import resolve_gold, ingest_corpus
from app.data.index_io import read_index


def _s(sid, raw):
    return Sample(sample_id=sid, studies=[2], corpus="demo", language="en",
                  text="t", audio=AudioRef(wav_path=Path("a.wav")),
                  gold_raw=raw, gold=None)


def test_resolve_gold_maps_merges_and_drops():
    out = resolve_gold([_s("1", "calm"), _s("2", "excitement"), _s("3", "apology"), _s("4", "anger")])
    assert out[0].gold is Emotion.NEUTRAL       # calm -> neutral (merge)
    assert out[1].gold is Emotion.HAPPINESS     # excitement -> happiness (merge)
    assert out[2].gold is None                  # apology -> DROP
    assert out[3].gold is Emotion.ANGER


def test_ingest_corpus_writes_index_and_reports(tmp_path):
    samples = [_s("1", "calm"), _s("2", "apology"), _s("3", "anger"), _s("4", "apology")]
    report = ingest_corpus(samples, "demo", tmp_path)
    assert report.total == 4
    assert report.kept == 2                      # calm(->neutral), anger
    assert report.dropped == 2                   # two apology
    assert report.dropped_labels == {"apology": 2}
    assert report.per_raw_label["calm"] == "neutral"
    assert report.per_raw_label["apology"] == "DROP"
    # index written and round-trips, dropped rows kept with gold=None
    got = read_index(tmp_path / "demo.parquet")
    assert len(got) == 4
    assert sum(1 for s in got if s.gold is None) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `app/data/ingest.py`:
```python
"""Ingest orchestrator: resolve gold labels, write the index, report harmonization."""
from pathlib import Path

from pydantic import BaseModel

from app.data.index_io import write_index
from app.data.schema import Sample
from app.labels.mapping import map_label


class IngestReport(BaseModel):
    corpus: str
    total: int
    kept: int
    dropped: int
    per_raw_label: dict[str, str]     # raw label -> canonical value or "DROP"
    dropped_labels: dict[str, int]    # raw label -> count of dropped samples


def resolve_gold(samples: list[Sample]) -> list[Sample]:
    resolved = []
    for s in samples:
        gold = map_label(s.gold_raw)  # raises KeyError on unknown label (fail loud)
        resolved.append(s.model_copy(update={"gold": gold}))
    return resolved


def ingest_corpus(samples: list[Sample], corpus: str, out_dir: Path) -> IngestReport:
    resolved = resolve_gold(samples)
    per_raw_label: dict[str, str] = {}
    dropped_labels: dict[str, int] = {}
    kept = 0
    for s in resolved:
        canonical = s.gold.value if s.gold is not None else "DROP"
        per_raw_label[s.gold_raw] = canonical
        if s.gold is None:
            dropped_labels[s.gold_raw] = dropped_labels.get(s.gold_raw, 0) + 1
        else:
            kept += 1
    write_index(resolved, Path(out_dir) / f"{corpus}.parquet")
    return IngestReport(
        corpus=corpus,
        total=len(resolved),
        kept=kept,
        dropped=len(resolved) - kept,
        per_raw_label=per_raw_label,
        dropped_labels=dropped_labels,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full P2 suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green (P1 + P2).

- [ ] **Step 6: Commit**

```bash
git add app/data/ingest.py tests/test_ingest.py
git commit -m "feat: ingest orchestrator with harmonization report"
```

---

## Self-Review

**1. Spec coverage (P2 scope = nEMO + CAMEO):**
- Normalized Sample index per corpus, audio-on-demand (no byte copy) → Tasks 2,3,4,5,6. ✅
- Gold harmonization via single map_label source, fail-loud on unknown → Task 6 (`resolve_gold`). ✅
- Non-Ekman merges/drops with reported counts → Task 6 (`IngestReport`). ✅
- nEMO = normalized_text, 24kHz, studies=[1]; CAMEO = transcription, 16kHz, studies=[2], full 13 splits via one loader + language→ISO → Tasks 4,5. ✅
- Settings/paths → Task 1. ✅
- Out of P2 scope (later): RAVDESS from Zenodo (audio+video pairing, face frames) = plan P2-RAVDESS; resampling to model sr = P3; CLI wiring for ingest = folded into P3 CLI or a thin follow-up. Intentional.

**2. Placeholder scan:** No TBD/TODO; every code + test step is complete and runnable. ✅

**3. Type consistency:** `Sample`/`AudioRef`/`Emotion` (P1) used consistently; `load_audio`/`decode_audio_bytes`, `write_index`/`read_index`/`samples_to_frame`, `load_nemo`, `load_cameo_split`/`corpus_from_path`/`CAMEO_LANGUAGES`, `resolve_gold`/`ingest_corpus`/`IngestReport` signatures match the File Structure interface block and the tests. Loaders set `gold=None`; ingest resolves it (single map_label call site). ✅

## Otwarte punkty / do kolejnych planów
- **P2-RAVDESS:** pobranie z Zenodo (Video_Speech per-aktor), parowanie audio↔wideo↔tekst po identyfikatorze z nazwy pliku, ekstrakcja + crop klatek twarzy, `has_face=True`/`video_path`. Wymaga autoryzacji dużego pobrania.
- **P3:** resampling do sr modelu (48 kHz custom CNN / 16 kHz reszta) w wrapperach modeli; CLI `ingest`/`predict`.
