# P2-RAVDESS — Ingest (tri-modal EN) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Znormalizować oryginalny zbiór RAVDESS (EN, Zenodo) do indeksu `Sample` z parowaniem mowa↔tekst↔twarz, etykietą gold z nazwy pliku, i utilem do ekstrakcji klatek twarzy — jedyny punkt trójmodalny (Badanie 1 EN).

**Architecture:** Rozszerzenie etapu `ingest`. Loader parsuje nazwy plików RAVDESS (7 kodów), paruje audio WAV (modalność 03) z wideo AV MP4 (modalność 01) po wspólnym identyfikatorze, buduje `Sample` (P1) z `has_face=True`/`video_path`, gold przez `map_label` (P1). `video.py` wyciąga klatki twarzy z MP4 (OpenCV + Haar) — używany później przez model twarzy (P3).

**Tech Stack:** Python 3.12, Pydantic v2, pyarrow/pandas, soundfile, **opencv-python-headless** (nowa zależność — dekodowanie wideo + detekcja twarzy Haar), pytest.

## Global Constraints

- Python **3.12**; jeden `.venv`; Mac M1, bez GPU (Haar cascade na CPU — zweryfikowane, wykrywa twarz na realnym mp4 RAVDESS 720p).
- **Pydantic v2**; buduje na P1 (`Emotion`, `map_label`, `Sample`, `AudioRef`) i P2 (`ingest_corpus`, `write_index`, `Settings`).
- Gold mapowany **wyłącznie** przez `map_label` (fail-loud). RAVDESS używa słowa `fearful` → wymaga dodania `fearful→FEAR` do `MASTER_MAP` (T1).
- Etykiety emocji i tekst pochodzą **z nazwy pliku** RAVDESS (brak osobnego pliku etykiet).
- **Brak cichego pomijania:** brakujące parowanie audio↔wideo ma **rzucać** (fail-loud), nie być po cichu pomijane.
- Audio nie jest kopiowane (ref do WAV); klatki wideo wyciągane na żądanie (nie pre-ekstrahujemy do dysku w tym planie — model twarzy w P3 cache'uje predykcje).
- `datasets/` niewersjonowane; indeks → `results/index/RAVDESS.parquet`. Kod/testy/commity po angielsku; **żadnego `Co-Authored-By`**.
- studies=[1], corpus="RAVDESS", language="en".

## Review mode (per memory: economical-subagent-execution)

Cały RAVDESS karmi wyniki → **pełna recenzja per-task dla T2–T4** (loader/pairing, ekstrakcja twarzy, integracja na realnych danych). **T1 (dodanie jednego synonimu do MASTER_MAP) — mechaniczne, pokryte finalnym review** (mechanizm `map_label` już zrecenzowany w P1). Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus.

## Dataset facts (verified on disk 2026-07-23)

- Wideo AV: `datasets/RAVDESS/Actor_01..24/01-01-EE-II-SS-RR-AA.mp4` (modalność 01 = audio+wideo, 1440 plików). Modalność `02-*` (wideo-only, 1440) — **ignorujemy**.
- Audio: `datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01..24/03-01-EE-II-SS-RR-AA.wav` (modalność 03, 1440 plików).
- Nazwa = 7 kodów `modality-vocal-emotion-intensity-statement-repetition-actor`.
- Emocja: `01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised`.
- Zdanie: `01`="Kids are talking by the door", `02`="Dogs are sitting by the door".
- Intensywność: `01 normal, 02 strong`. Aktor: nieparzysty=male, parzysty=female.

---

## File Structure

```
app/
  labels/mapping.py        # += "fearful": Emotion.FEAR  (T1)
  config/settings.py       # += ravdess_audio_dir / ravdess_video_dir properties (T4)
  data/
    ravdess.py             # RAVDESS_EMOTIONS, RAVDESS_STATEMENTS, parse_ravdess_name(), load_ravdess()
    video.py               # sample_frame_indices(), detect_and_crop_face(), extract_face_frames()
tests/
  test_mapping.py          # += fearful case (T1)
  test_loader_ravdess.py   # parsing, pairing, fail-loud, field mapping
  test_video.py            # frame-index sampling, blank-frame -> None, real-mp4 integration (skip if absent)
  test_ingest_ravdess.py   # T4: end-to-end ingest of a small real subset (skip if datasets absent)
requirements.txt           # += opencv-python-headless
```

Interfejsy:
- `app.data.ravdess.parse_ravdess_name(stem: str) -> dict` — 7 codes -> {modality,vocal,emotion,intensity,statement,repetition,actor}.
- `app.data.ravdess.load_ravdess(audio_dir: Path, video_dir: Path) -> list[Sample]`.
- `app.data.video.sample_frame_indices(total: int, n: int) -> list[int]`.
- `app.data.video.detect_and_crop_face(frame_bgr: np.ndarray) -> np.ndarray | None` (returns RGB crop).
- `app.data.video.extract_face_frames(video_path: Path, num_frames: int = 8) -> list[np.ndarray]` (RGB face crops; frames with no detected face skipped).

---

### Task 1: Add "fearful" -> FEAR to the label map  [economize: no per-task reviewer]

**Files:**
- Modify: `app/labels/mapping.py`
- Test: `tests/test_mapping.py`

**Interfaces:**
- Consumes/Produces: extends existing `MASTER_MAP`; no signature change.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mapping.py`:
```python
def test_ravdess_fearful_maps_to_fear():
    from app.config.emotions import Emotion
    assert map_label("fearful") is Emotion.FEAR
    assert map_label("FEARFUL") is Emotion.FEAR  # normalization
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mapping.py::test_ravdess_fearful_maps_to_fear -v`
Expected: FAIL with `KeyError: "Unknown emotion label: 'fearful'"`.

- [ ] **Step 3: Add the mapping**

In `app/labels/mapping.py`, add to `MASTER_MAP` next to the `"fear": Emotion.FEAR` entry:
```python
    "fear": Emotion.FEAR,
    "fearful": Emotion.FEAR,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mapping.py -v`
Expected: PASS (all mapping tests incl. the new one).

- [ ] **Step 5: Commit**

```bash
git add app/labels/mapping.py tests/test_mapping.py
git commit -m "feat: map RAVDESS 'fearful' label to FEAR"
```

---

### Task 2: RAVDESS loader (parse filenames, pair audio+video)

**Files:**
- Create: `app/data/ravdess.py`
- Test: `tests/test_loader_ravdess.py`

**Interfaces:**
- Consumes: `Sample`, `AudioRef` (P1).
- Produces: `RAVDESS_EMOTIONS`, `RAVDESS_STATEMENTS`, `parse_ravdess_name(stem)`, `load_ravdess(audio_dir, video_dir) -> list[Sample]`.
  - Iterates audio WAVs (`03-*.wav` under `audio_dir/Actor_*/`); for each, parses codes, locates the paired AV video `video_dir/Actor_{AA}/01-01-EE-II-SS-RR-AA.mp4`; **raises FileNotFoundError if the paired video is missing** (fail-loud).
  - `sample_id = f"RAVDESS:{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}"`; `gold_raw = RAVDESS_EMOTIONS[emotion]`; `gold=None` (resolved by ingest); `text = RAVDESS_STATEMENTS[statement]`; `audio = AudioRef(wav_path=wav)`; `has_face=True`, `video_path=mp4`; `speaker_id=f"ravdess_actor_{actor}"`; `gender="male" if int(actor)%2==1 else "female"`; `intensity="normal"/"strong"`; `age=None`; studies=[1], corpus="RAVDESS", language="en".

- [ ] **Step 1: Write the failing test**

Create `tests/test_loader_ravdess.py`:
```python
from pathlib import Path
import pytest
from app.data.ravdess import parse_ravdess_name, load_ravdess


def test_parse_ravdess_name():
    d = parse_ravdess_name("03-01-06-02-02-01-16")
    assert d["modality"] == "03" and d["emotion"] == "06"
    assert d["statement"] == "02" and d["actor"] == "16"


def _make_pair(tmp_path, emotion="06", statement="02", intensity="02", rep="01", actor="16"):
    key = f"01-{emotion}-{intensity}-{statement}-{rep}-{actor}"
    adir = tmp_path / "Audio_Speech_Actors_01-24" / f"Actor_{actor}"
    vdir = tmp_path / f"Actor_{actor}"
    adir.mkdir(parents=True); vdir.mkdir(parents=True)
    wav = adir / f"03-{key}.wav"; wav.write_bytes(b"RIFF")  # content irrelevant; loader only uses the path
    mp4 = vdir / f"01-{key}.mp4"; mp4.write_bytes(b"\x00")
    return tmp_path, wav, mp4


def test_load_ravdess_pairs_and_maps(tmp_path):
    root, wav, mp4 = _make_pair(tmp_path, emotion="06", statement="02", actor="16")
    samples = load_ravdess(root / "Audio_Speech_Actors_01-24", root)
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "RAVDESS:01-06-02-02-01-16"
    assert s.corpus == "RAVDESS" and s.language == "en" and s.studies == [1]
    assert s.gold_raw == "fearful" and s.gold is None
    assert s.text == "Dogs are sitting by the door"
    assert s.audio.wav_path == wav and s.has_face is True and s.video_path == mp4
    assert s.speaker_id == "ravdess_actor_16" and s.gender == "female"  # 16 even
    assert s.intensity == "strong"


def test_load_ravdess_missing_video_raises(tmp_path):
    root, wav, mp4 = _make_pair(tmp_path, actor="16")
    mp4.unlink()  # remove the paired video
    with pytest.raises(FileNotFoundError):
        load_ravdess(root / "Audio_Speech_Actors_01-24", root)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loader_ravdess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data.ravdess'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/data/ravdess.py`:
```python
"""RAVDESS (English) loader: original Zenodo files -> tri-modal Sample list (Study 1).

Emotion and text are decoded from the 7-code filename; audio (modality 03 WAV)
is paired with the audio-visual video (modality 01 MP4) by the shared identifier.
"""
from pathlib import Path

from app.data.schema import AudioRef, Sample

RAVDESS_EMOTIONS: dict[str, str] = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_STATEMENTS: dict[str, str] = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
_FIELDS = ["modality", "vocal", "emotion", "intensity", "statement", "repetition", "actor"]


def parse_ravdess_name(stem: str) -> dict:
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Not a RAVDESS filename stem: {stem!r}")
    return dict(zip(_FIELDS, parts))


def load_ravdess(audio_dir: Path, video_dir: Path) -> list[Sample]:
    audio_dir = Path(audio_dir)
    video_dir = Path(video_dir)
    samples: list[Sample] = []
    for wav in sorted(audio_dir.glob("Actor_*/03-*.wav")):
        c = parse_ravdess_name(wav.stem)
        key = f"{c['vocal']}-{c['emotion']}-{c['intensity']}-{c['statement']}-{c['repetition']}-{c['actor']}"
        mp4 = video_dir / f"Actor_{c['actor']}" / f"01-{key}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(f"No paired RAVDESS video for {wav.name}: expected {mp4}")
        actor_num = int(c["actor"])
        samples.append(Sample(
            sample_id=f"RAVDESS:{key}",
            studies=[1],
            corpus="RAVDESS",
            language="en",
            text=RAVDESS_STATEMENTS[c["statement"]],
            audio=AudioRef(wav_path=wav),
            has_face=True,
            video_path=mp4,
            gold_raw=RAVDESS_EMOTIONS[c["emotion"]],
            gold=None,
            speaker_id=f"ravdess_actor_{c['actor']}",
            gender="male" if actor_num % 2 == 1 else "female",
            age=None,
            intensity="normal" if c["intensity"] == "01" else "strong",
        ))
    return samples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loader_ravdess.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/data/ravdess.py tests/test_loader_ravdess.py
git commit -m "feat: RAVDESS loader with audio/video/text pairing"
```

---

### Task 3: Face frame extraction (OpenCV + Haar)

**Files:**
- Modify: `requirements.txt` (add `opencv-python-headless`)
- Create: `app/data/video.py`
- Test: `tests/test_video.py`

**Interfaces:**
- Produces:
  - `sample_frame_indices(total: int, n: int) -> list[int]` — up to `n` frame indices spread uniformly over `[0, total)`; if `total <= n`, returns all indices `0..total-1`; returns `[]` for `total <= 0`.
  - `detect_and_crop_face(frame_bgr: np.ndarray) -> np.ndarray | None` — Haar-detect the largest face, crop it (with a small margin), convert BGR→RGB, return the crop; `None` if no face.
  - `extract_face_frames(video_path: Path, num_frames: int = 8) -> list[np.ndarray]` — sample frames, detect+crop, return RGB crops (skipping frames with no face). Aggregation over crops is the face model's job (P3).

- [ ] **Step 1: Add opencv to requirements and install**

Append to `requirements.txt`:
```
opencv-python-headless>=4.8,<5
```
Run: `.venv/bin/pip install -r requirements.txt` then `.venv/bin/python -c "import cv2; print(cv2.__version__)"`
Expected: prints a 4.x version.

- [ ] **Step 2: Write the failing test**

Create `tests/test_video.py`:
```python
import glob
import numpy as np
import pytest
from pathlib import Path
from app.data.video import sample_frame_indices, detect_and_crop_face, extract_face_frames


def test_sample_frame_indices_uniform():
    assert sample_frame_indices(100, 5) == [0, 25, 50, 74, 99]  # endpoints included, spread uniformly


def test_sample_frame_indices_fewer_than_n():
    assert sample_frame_indices(3, 8) == [0, 1, 2]


def test_sample_frame_indices_empty():
    assert sample_frame_indices(0, 8) == []


def test_detect_face_returns_none_on_blank_frame():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)  # no face
    assert detect_and_crop_face(blank) is None


@pytest.mark.skipif(not glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"),
                    reason="RAVDESS dataset not present")
def test_extract_face_frames_on_real_ravdess_video():
    path = Path(sorted(glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"))[0])
    crops = extract_face_frames(path, num_frames=6)
    assert len(crops) >= 1                         # at least one face found
    for c in crops:
        assert c.ndim == 3 and c.shape[2] == 3     # RGB
        assert c.dtype == np.uint8
        assert c.shape[0] > 0 and c.shape[1] > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data.video'`.

- [ ] **Step 4: Write minimal implementation**

Create `app/data/video.py`:
```python
"""Face-frame extraction from RAVDESS videos (OpenCV + Haar cascade, CPU).

Verified on real RAVDESS 720p footage: the built-in frontal-face Haar cascade
detects the (single, frontal, well-lit) actor face without any extra model.
Returns RGB face crops; running the FER classifier over them is the P3 model's job.
"""
from pathlib import Path

import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def sample_frame_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def detect_and_crop_face(frame_bgr: np.ndarray, margin: float = 0.15) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # largest face
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame_bgr.shape[1], x + w + mx), min(frame_bgr.shape[0], y + h + my)
    crop_bgr = frame_bgr[y0:y1, x0:x1]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def extract_face_frames(video_path: Path, num_frames: int = 8) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crops: list[np.ndarray] = []
        for idx in sample_frame_indices(total, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            face = detect_and_crop_face(frame)
            if face is not None:
                crops.append(face)
        return crops
    finally:
        cap.release()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_video.py -v`
Expected: PASS (integration test runs since the dataset is present).

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/data/video.py tests/test_video.py
git commit -m "feat: OpenCV Haar face-frame extraction from video"
```

---

### Task 4: RAVDESS ingest driver + real-data integration

**Files:**
- Modify: `app/config/settings.py` (add RAVDESS dir properties)
- Create: `tests/test_ingest_ravdess.py`
- Test: end-to-end on a small real subset

**Interfaces:**
- `Settings.ravdess_audio_dir -> datasets_root/"RAVDESS"/"Audio_Speech_Actors_01-24"`; `Settings.ravdess_video_dir -> datasets_root/"RAVDESS"`.
- Reuses P2 `ingest_corpus(samples, "RAVDESS", index_dir)` — no new ingest code; verifies gold resolution (incl. `calm→neutral`, `fearful→fear`) and index round-trip on real RAVDESS samples.

- [ ] **Step 1: Add Settings properties**

In `app/config/settings.py`, add:
```python
    @property
    def ravdess_audio_dir(self) -> Path:
        return self.datasets_root / "RAVDESS" / "Audio_Speech_Actors_01-24"

    @property
    def ravdess_video_dir(self) -> Path:
        return self.datasets_root / "RAVDESS"
```

- [ ] **Step 2: Write the integration test**

Create `tests/test_ingest_ravdess.py`:
```python
import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.data.ravdess import load_ravdess
from app.data.ingest import ingest_corpus
from app.data.index_io import read_index

_HAS_DATA = bool(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))


@pytest.mark.skipif(not _HAS_DATA, reason="RAVDESS dataset not present")
def test_ingest_ravdess_real_subset(tmp_path):
    # Load the full corpus, then ingest just Actor_01's samples for speed.
    samples = load_ravdess(Path("datasets/RAVDESS/Audio_Speech_Actors_01-24"),
                           Path("datasets/RAVDESS"))
    subset = [s for s in samples if s.speaker_id == "ravdess_actor_01"]
    assert len(subset) == 60  # RAVDESS: 60 speech utterances per actor
    report = ingest_corpus(subset, "RAVDESS_actor01", tmp_path)
    assert report.total == 60
    assert report.dropped == 0  # every RAVDESS label maps (calm->neutral, fearful->fear, ...)
    assert report.per_raw_label["calm"] == "neutral"
    assert report.per_raw_label["fearful"] == "fear"
    got = read_index(tmp_path / "RAVDESS_actor01.parquet")
    assert all(s.has_face and s.video_path is not None for s in got)
    assert all(s.gold is not None for s in got)  # nothing dropped
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ingest_ravdess.py -v`
Expected: FAIL (settings properties missing → AttributeError, or the test drives code not yet present). If it errors on the Settings import path, ensure Step 1 was applied.

- [ ] **Step 4: Make it pass**

Confirm Step 1 (Settings properties) applied. The remaining logic reuses existing `load_ravdess` (T2) + `ingest_corpus` (P2). Run again:
Run: `.venv/bin/python -m pytest tests/test_ingest_ravdess.py -v`
Expected: PASS (60 Actor_01 utterances ingested, 0 dropped).

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add app/config/settings.py tests/test_ingest_ravdess.py
git commit -m "feat: RAVDESS ingest wiring + real-subset integration test"
```

---

## Self-Review

**1. Spec coverage (P2-RAVDESS = tri-modal EN ingest):**
- RAVDESS from original Zenodo (audio+video), paired by filename identifier → T2. ✅
- gold/text from filename; `fearful` handled in the single map source → T1, T2. ✅
- Face-frame extraction utility (feeds P3 face model) → T3. ✅
- Fail-loud on missing pairing; audio not copied; frames on demand → T2, T3. ✅
- Index written via reused P2 ingest; verified on real data → T4. ✅
- Out of scope (later): running the ViT FER classifier over the crops + aggregation = P3 (models/face); CLI to ingest all corpora = P3.

**2. Placeholder scan:** No TBD/TODO; every code + test step is complete and runnable.

**3. Type consistency:** `Sample`/`AudioRef`/`Emotion` (P1), `ingest_corpus`/`read_index` (P2), `map_label` (P1+T1) used consistently. `parse_ravdess_name`/`load_ravdess`, `sample_frame_indices`/`detect_and_crop_face`/`extract_face_frames`, Settings properties match the interface block and tests. Loaders set `gold=None`; ingest resolves it.

## Otwarte punkty / do P3
- Model twarzy (trpakov ViT) uruchamiany na cropach z `extract_face_frames`, agregacja softmax po klatkach, wybór `num_frames`.
- Jeśli Haar okaże się gubić twarze na części nagrań (kontrola jakości w P3) — wymiana detektora za interfejsem `detect_and_crop_face` (np. mediapipe), bez zmiany reszty.
- CLI `ingest --corpus ravdess` + uruchomienie pełnej ingestii 1440 próbek.
