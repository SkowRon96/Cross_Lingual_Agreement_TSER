# P1 — Fundament — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zbudować rdzeń aplikacji badawczej — kanoniczną przestrzeń emocji, harmonizację etykiet, schematy danych, protokół modeli i cache'owany magazyn predykcji — z pełnym pokryciem testami TDD.

**Architecture:** Podejście A (potok etapów). P1 dostarcza współdzielone, bezstanowe komponenty, od których zależą wszystkie kolejne etapy (`ingest`/`predict`/`analyze`/`report`). Nic tu nie uruchamia modeli ani nie czyta datasetów — to czysta logika + I/O magazynu, w pełni testowalna jednostkowo.

**Tech Stack:** Python 3.12, Pydantic v2, pandas + pyarrow (parquet), pytest. Bez torch/transformers (wchodzą w P3).

## Global Constraints

- Python **3.12** (`python3.12`; zweryfikowane 3.12.1). Jeden `.venv` w korzeniu repo.
- **Mac M1, 16 GB unified, bez GPU.** Kod device-agnostic; w P1 brak zależności GPU.
- **Pydantic wszędzie** (config, `Sample`, `Prediction`, obiekty wyników).
- Kod, testy, nazwy, commit messages **po angielsku**; instrukcje `.md` po polsku.
- **Żadnego zmyślania wyników** — komponenty muszą zawodzić głośno (raise), nie zgadywać po cichu.
- Kanoniczna przestrzeń emocji (7): `anger, disgust, fear, happiness, sadness, surprise, neutral`.
- **No Co-Authored-By** w commitach.
- `old_projects/`, `datasets/`, `models/` nietykalne/niewersjonowane.

---

## File Structure

```
app/
  __init__.py
  config/
    __init__.py
    emotions.py        # Emotion enum, CANONICAL_ORDER, EMOTION_COLORS
    settings.py        # Pydantic Settings (paths, device, seed, bootstrap_n)
  labels/
    __init__.py
    mapping.py         # MASTER_MAP, DROP_LABELS, map_label()
    mapper.py          # LabelMapper (native scores -> canonical, aggregate/drop/renorm)
  data/
    __init__.py
    schema.py          # AudioRef, Sample
  models/
    __init__.py
    base.py            # Prediction, EmotionModel protocol, to_prediction()
  predict/
    __init__.py
    store.py           # PredictionStore (parquet + manifest + cache signature)
tests/
  __init__.py
  test_emotions.py
  test_mapping.py
  test_mapper.py
  test_schema.py
  test_base.py
  test_store.py
.vscode/
  launch.json         # debug configs (rozszerzane w kolejnych planach)
requirements.txt      # pinned core deps (rozszerzane w P3)
pyproject.toml        # pytest config + package metadata
```

Interfejsy między plikami (kontrakt utrwalony tutaj):
- `app.config.emotions.Emotion` (str-Enum) — używany wszędzie jako typ etykiety kanonicznej.
- `app.labels.mapping.map_label(raw, overrides=None) -> Emotion | None` — pojedyncza etykieta.
- `app.labels.mapper.LabelMapper(native_labels, overrides=None)` — `.supported`, `.map_scores(scores) -> dict[Emotion,float]`, `.map_label(native) -> Emotion|None`.
- `app.data.schema.Sample`, `AudioRef`.
- `app.models.base.Prediction`, `EmotionModel`, `to_prediction(sample_id, model_id, modality, native_labels, scores, mapper) -> Prediction`.
- `app.predict.store.PredictionStore(root: Path)` — `.write()`, `.read()`, `.existing_sample_ids()`, `.append()`, `.is_cached()`, `.signature()`.

---

### Task 1: Environment & project skeleton

**Files:**
- Create: `requirements.txt` (overwrite provisional), `pyproject.toml`, `.vscode/launch.json`
- Create: `app/__init__.py`, `app/config/__init__.py`, `app/labels/__init__.py`, `app/data/__init__.py`, `app/models/__init__.py`, `app/predict/__init__.py`
- Create: `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `app` package; `pytest` runs green; `.venv` with core deps.

- [ ] **Step 1: Create the venv with Python 3.12**

Run:
```bash
cd /Users/kamil/Documents/PROGRAMMING/Cross_Lingual_Agreement_TSER
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
Expected: `python --version` → `Python 3.12.x`.

- [ ] **Step 2: Pin core requirements**

Overwrite `requirements.txt`:
```
# Core deps for P1 (Foundation) + P2 (Ingest). Model/plotting deps added in P3/P4.
# Python 3.12 | Apple M1 (CPU/MPS, no CUDA)
pydantic>=2.7,<3
pydantic-settings>=2.3,<3
numpy>=1.26,<3
pandas>=2.2,<3
pyarrow>=16,<20
pytest>=8,<9
```

- [ ] **Step 3: Install and freeze-check**

Run:
```bash
pip install -r requirements.txt
python -c "import pydantic, pydantic_settings, numpy, pandas, pyarrow, pytest; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Create pyproject.toml (pytest config + package)**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cross-lingual-agreement-tser"
version = "0.1.0"
requires-python = ">=3.12"

[tool.setuptools]
packages = ["app"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 5: Create empty package __init__.py files**

Create these files, each containing a single line:
```python
"""Cross-Lingual Agreement (TSER) research application."""
```
Paths: `app/__init__.py`, `app/config/__init__.py`, `app/labels/__init__.py`, `app/data/__init__.py`, `app/models/__init__.py`, `app/predict/__init__.py`, `tests/__init__.py`.

- [ ] **Step 6: Write a smoke test**

Create `tests/test_smoke.py`:
```python
def test_app_imports():
    import app
    assert app.__doc__ is not None
```

- [ ] **Step 7: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 8: Create initial launch.json**

Create `.vscode/launch.json` (debug configs extended in later plans):
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Pytest: all",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["-v"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

- [ ] **Step 9: Commit**

```bash
git add requirements.txt pyproject.toml .vscode/launch.json app tests
git commit -m "chore: project skeleton, venv deps, pytest setup"
```

---

### Task 2: Canonical emotions

**Files:**
- Create: `app/config/emotions.py`
- Test: `tests/test_emotions.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Emotion` (str-Enum), `CANONICAL_ORDER: list[Emotion]` (fixed 7-order), `EMOTION_COLORS: dict[Emotion, str]` (colorblind-safe hex).

- [ ] **Step 1: Write the failing test**

Create `tests/test_emotions.py`:
```python
from app.config.emotions import Emotion, CANONICAL_ORDER, EMOTION_COLORS


def test_seven_canonical_emotions():
    assert [e.value for e in CANONICAL_ORDER] == [
        "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
    ]


def test_emotion_is_str_enum():
    assert Emotion.HAPPINESS == "happiness"
    assert Emotion("surprise") is Emotion.SURPRISE


def test_every_emotion_has_a_distinct_color():
    assert set(EMOTION_COLORS) == set(CANONICAL_ORDER)
    assert len(set(EMOTION_COLORS.values())) == len(CANONICAL_ORDER)
    for hex_code in EMOTION_COLORS.values():
        assert hex_code.startswith("#") and len(hex_code) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emotions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config.emotions'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/config/emotions.py`:
```python
"""Canonical Ekman-6 + neutral emotion space (single source of truth)."""
from enum import Enum


class Emotion(str, Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


# Fixed order used in every table, confusion matrix and figure.
CANONICAL_ORDER: list[Emotion] = [
    Emotion.ANGER,
    Emotion.DISGUST,
    Emotion.FEAR,
    Emotion.HAPPINESS,
    Emotion.SADNESS,
    Emotion.SURPRISE,
    Emotion.NEUTRAL,
]

# Okabe-Ito colorblind-safe palette; neutral rendered as grey.
EMOTION_COLORS: dict[Emotion, str] = {
    Emotion.ANGER: "#D55E00",      # vermillion
    Emotion.DISGUST: "#009E73",    # bluish green
    Emotion.FEAR: "#CC79A7",       # reddish purple
    Emotion.HAPPINESS: "#E69F00",  # orange
    Emotion.SADNESS: "#0072B2",    # blue
    Emotion.SURPRISE: "#F0E442",   # yellow
    Emotion.NEUTRAL: "#999999",    # grey
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emotions.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/config/emotions.py tests/test_emotions.py
git commit -m "feat: canonical emotion space (Ekman-6 + neutral)"
```

---

### Task 3: Label mapping (raw -> canonical, single source of truth)

**Files:**
- Create: `app/labels/mapping.py`
- Test: `tests/test_mapping.py`

**Interfaces:**
- Consumes: `Emotion` from `app.config.emotions`.
- Produces: `MASTER_MAP: dict[str, Emotion]`, `DROP_LABELS: set[str]`, `map_label(raw: str, overrides: dict[str, Emotion | None] | None = None) -> Emotion | None`. Returns an `Emotion`, or `None` when the label is explicitly dropped; raises `KeyError` on an unknown label (fail loud, never guess).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mapping.py`:
```python
import pytest
from app.config.emotions import Emotion
from app.labels.mapping import map_label, MASTER_MAP, DROP_LABELS


def test_direct_ekman_labels():
    assert map_label("anger") is Emotion.ANGER
    assert map_label("disgust") is Emotion.DISGUST
    assert map_label("fear") is Emotion.FEAR
    assert map_label("sadness") is Emotion.SADNESS
    assert map_label("neutral") is Emotion.NEUTRAL


def test_synonyms_normalize_to_canonical():
    assert map_label("joy") is Emotion.HAPPINESS
    assert map_label("happy") is Emotion.HAPPINESS
    assert map_label("happiness") is Emotion.HAPPINESS
    assert map_label("surprised") is Emotion.SURPRISE
    assert map_label("angry") is Emotion.ANGER
    assert map_label("sad") is Emotion.SADNESS


def test_established_merges():
    assert map_label("calm") is Emotion.NEUTRAL
    assert map_label("excitement") is Emotion.HAPPINESS
    assert map_label("enthusiasm") is Emotion.HAPPINESS


def test_case_and_whitespace_insensitive():
    assert map_label("  Anger ") is Emotion.ANGER
    assert map_label("HAPPY") is Emotion.HAPPINESS


def test_dropped_labels_return_none():
    for lbl in ["hope", "pride", "contempt", "frustration", "gratitude",
                "love", "sarcasm", "anxiety", "apology", "assertiveness",
                "concern", "encouragement", "poker"]:
        assert map_label(lbl) is None


def test_none_of_them_dropped_by_default_but_overridable():
    assert map_label("none of them") is None
    assert map_label("none of them", overrides={"none of them": Emotion.NEUTRAL}) is Emotion.NEUTRAL


def test_override_can_drop_a_normally_mapped_label():
    assert map_label("none of them", overrides={"none of them": None}) is None


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        map_label("banana")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mapping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.labels.mapping'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/labels/mapping.py`:
```python
"""Master raw-label -> canonical-emotion mapping (single source of truth).

Applied identically to gold labels and model predictions. Unknown labels
raise KeyError so a new/unexpected label can never be silently mishandled.
"""
from app.config.emotions import Emotion

# Raw label string (lowercased) -> canonical Emotion.
MASTER_MAP: dict[str, Emotion] = {
    "anger": Emotion.ANGER,
    "angry": Emotion.ANGER,
    "disgust": Emotion.DISGUST,
    "fear": Emotion.FEAR,
    "happiness": Emotion.HAPPINESS,
    "happy": Emotion.HAPPINESS,
    "joy": Emotion.HAPPINESS,
    "sadness": Emotion.SADNESS,
    "sad": Emotion.SADNESS,
    "surprise": Emotion.SURPRISE,
    "surprised": Emotion.SURPRISE,
    "neutral": Emotion.NEUTRAL,
    # Established merges (see spec section 4).
    "calm": Emotion.NEUTRAL,
    "excitement": Emotion.HAPPINESS,
    "enthusiasm": Emotion.HAPPINESS,
}

# Labels intentionally dropped (non-Ekman, no established equivalence).
# "none of them" is dropped by default; models lacking an explicit neutral
# override it to Emotion.NEUTRAL at LabelMapper construction.
DROP_LABELS: set[str] = {
    "hope", "pride", "contempt", "frustration", "gratitude", "love",
    "sarcasm", "anxiety", "apology", "assertiveness", "concern",
    "encouragement", "poker", "none of them",
}


def map_label(
    raw: str,
    overrides: dict[str, Emotion | None] | None = None,
) -> Emotion | None:
    """Map a raw label to a canonical Emotion, or None if dropped.

    Resolution order: per-model overrides -> master map -> drop set ->
    KeyError. `overrides` keys are matched after the same normalization.
    """
    key = raw.strip().lower()
    if overrides:
        norm_overrides = {k.strip().lower(): v for k, v in overrides.items()}
        if key in norm_overrides:
            return norm_overrides[key]
    if key in MASTER_MAP:
        return MASTER_MAP[key]
    if key in DROP_LABELS:
        return None
    raise KeyError(f"Unknown emotion label: {raw!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mapping.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add app/labels/mapping.py tests/test_mapping.py
git commit -m "feat: master raw->canonical label mapping"
```

---

### Task 4: LabelMapper (native scores -> canonical, aggregate/drop/renormalize)

**Files:**
- Create: `app/labels/mapper.py`
- Test: `tests/test_mapper.py`

**Interfaces:**
- Consumes: `Emotion`, `map_label`.
- Produces: `LabelMapper(native_labels: list[str], overrides: dict[str, Emotion | None] | None = None)` with:
  - `.supported -> set[Emotion]` — canonical emotions this model can emit.
  - `.map_label(native: str) -> Emotion | None`.
  - `.map_scores(scores: Sequence[float]) -> dict[Emotion, float]` — sums probabilities of natives mapping to the same emotion, drops the mass of DROP-labels, renormalizes over supported emotions. Returns `{}` when all mass is dropped (caller treats as abstain).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mapper.py`:
```python
import math
import pytest
from app.config.emotions import Emotion
from app.labels.mapper import LabelMapper


def approx_sum_one(d):
    return math.isclose(sum(d.values()), 1.0, rel_tol=1e-9)


def test_supported_reflects_mapped_labels_only():
    m = LabelMapper(["angry", "happy", "sad", "neutral"])  # IEMOCAP-style
    assert m.supported == {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}


def test_map_label_passthrough():
    m = LabelMapper(["calm", "neutral"])
    assert m.map_label("calm") is Emotion.NEUTRAL


def test_scores_aggregate_when_two_natives_map_to_one_emotion():
    # RAVDESS-style: calm and neutral both -> NEUTRAL, summed.
    labels = ["angry", "calm", "neutral"]
    m = LabelMapper(labels)
    out = m.map_scores([0.2, 0.3, 0.5])
    assert approx_sum_one(out)
    assert math.isclose(out[Emotion.NEUTRAL], 0.8, rel_tol=1e-9)
    assert math.isclose(out[Emotion.ANGER], 0.2, rel_tol=1e-9)


def test_dropped_mass_is_removed_and_renormalized():
    # contempt is dropped; remaining mass renormalizes to sum 1.
    labels = ["anger", "contempt", "neutral"]
    m = LabelMapper(labels)
    out = m.map_scores([0.5, 0.3, 0.2])
    assert Emotion.DISGUST not in out
    assert approx_sum_one(out)
    # 0.5 and 0.2 renormalized over 0.7 total.
    assert math.isclose(out[Emotion.ANGER], 0.5 / 0.7, rel_tol=1e-9)
    assert math.isclose(out[Emotion.NEUTRAL], 0.2 / 0.7, rel_tol=1e-9)


def test_all_mass_dropped_returns_empty():
    m = LabelMapper(["contempt", "love"])
    assert m.map_scores([0.6, 0.4]) == {}


def test_none_of_them_override_to_neutral():
    # visegradmedia-style: no explicit neutral; none-of-them -> NEUTRAL.
    labels = ["anger", "fear", "disgust", "sadness", "happiness", "none of them"]
    m = LabelMapper(labels, overrides={"none of them": Emotion.NEUTRAL})
    assert Emotion.NEUTRAL in m.supported
    out = m.map_scores([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    assert math.isclose(out[Emotion.NEUTRAL], 0.5, rel_tol=1e-9)


def test_length_mismatch_raises():
    m = LabelMapper(["anger", "neutral"])
    with pytest.raises(ValueError):
        m.map_scores([0.5, 0.3, 0.2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.labels.mapper'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/labels/mapper.py`:
```python
"""LabelMapper: map a model's native score vector into the canonical space.

Aggregates probabilities of native labels that collapse to the same canonical
emotion, discards the mass of dropped labels, and renormalizes over the
emotions the model actually supports.
"""
from collections.abc import Sequence

from app.config.emotions import Emotion
from app.labels.mapping import map_label


class LabelMapper:
    def __init__(
        self,
        native_labels: list[str],
        overrides: dict[str, Emotion | None] | None = None,
    ) -> None:
        self.native_labels = list(native_labels)
        self.overrides = overrides
        # Precompute native index -> Emotion | None (None means dropped).
        self._targets: list[Emotion | None] = [
            map_label(lbl, overrides=overrides) for lbl in self.native_labels
        ]

    @property
    def supported(self) -> set[Emotion]:
        return {t for t in self._targets if t is not None}

    def map_label(self, native: str) -> Emotion | None:
        return map_label(native, overrides=self.overrides)

    def map_scores(self, scores: Sequence[float]) -> dict[Emotion, float]:
        if len(scores) != len(self.native_labels):
            raise ValueError(
                f"Expected {len(self.native_labels)} scores, got {len(scores)}"
            )
        aggregated: dict[Emotion, float] = {}
        kept_mass = 0.0
        for target, score in zip(self._targets, scores):
            if target is None:
                continue
            aggregated[target] = aggregated.get(target, 0.0) + float(score)
            kept_mass += float(score)
        if kept_mass <= 0.0:
            return {}
        return {emotion: value / kept_mass for emotion, value in aggregated.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mapper.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add app/labels/mapper.py tests/test_mapper.py
git commit -m "feat: LabelMapper native->canonical score aggregation"
```

---

### Task 5: Data schema (AudioRef, Sample)

**Files:**
- Create: `app/data/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `Emotion`.
- Produces:
  - `AudioRef` (Pydantic) — either `(parquet_path: Path, row_index: int)` OR `wav_path: Path`; exactly one source, validated.
  - `Sample` (Pydantic) — `sample_id: str`, `studies: list[int]`, `corpus: str`, `language: str`, `text: str`, `audio: AudioRef`, `has_face: bool = False`, `video_path: Path | None = None`, `gold_raw: str`, `gold: Emotion | None`, `speaker_id/gender/age/intensity: str | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data.schema'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/data/schema.py`:
```python
"""Normalized data contract: AudioRef and Sample."""
from pathlib import Path

from pydantic import BaseModel, model_validator

from app.config.emotions import Emotion


class AudioRef(BaseModel):
    """Reference to audio, decoded on demand. Either a parquet row or a wav file."""
    parquet_path: Path | None = None
    row_index: int | None = None
    wav_path: Path | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "AudioRef":
        has_parquet = self.parquet_path is not None
        has_wav = self.wav_path is not None
        if has_parquet == has_wav:
            raise ValueError("AudioRef needs exactly one source: parquet row OR wav file")
        if has_parquet and self.row_index is None:
            raise ValueError("parquet_path requires row_index")
        if has_wav and self.row_index is not None:
            raise ValueError("row_index is only valid with parquet_path")
        return self


class Sample(BaseModel):
    """One utterance normalized across corpora."""
    sample_id: str
    studies: list[int]
    corpus: str
    language: str
    text: str
    audio: AudioRef
    has_face: bool = False
    video_path: Path | None = None
    gold_raw: str
    gold: Emotion | None  # None = dropped (label outside canonical space)
    speaker_id: str | None = None
    gender: str | None = None
    age: str | None = None
    intensity: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app/data/schema.py tests/test_schema.py
git commit -m "feat: Sample and AudioRef data contract"
```

---

### Task 6: Prediction, EmotionModel protocol, to_prediction()

**Files:**
- Create: `app/models/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `Emotion`, `LabelMapper`.
- Produces:
  - `Prediction` (Pydantic) — `sample_id, model_id, modality, pred_raw: str | None, pred: Emotion | None, confidence: float | None, scores: dict[Emotion, float], abstained: bool, abstain_reason: str | None`.
  - `EmotionModel` (Protocol) — attrs `model_id: str`, `modality: str`; method `predict_batch(samples: list[Sample]) -> list[Prediction]`.
  - `to_prediction(sample_id, model_id, modality, native_labels, scores, mapper) -> Prediction` — DRY builder every model uses; runs `mapper.map_scores`, picks argmax as `pred`, max value as `confidence`; if scores empty -> abstained prediction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_base.py`:
```python
import math
from app.config.emotions import Emotion
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction


def test_to_prediction_picks_argmax_and_confidence():
    mapper = LabelMapper(["angry", "calm", "neutral"])
    p = to_prediction(
        sample_id="s1", model_id="dummy", modality="speech",
        native_labels=["angry", "calm", "neutral"],
        scores=[0.2, 0.3, 0.5], mapper=mapper,
    )
    assert p.pred is Emotion.NEUTRAL           # 0.3 (calm) + 0.5 (neutral) = 0.8
    assert math.isclose(p.confidence, 0.8, rel_tol=1e-9)
    assert p.abstained is False
    assert p.pred_raw == "neutral"             # native argmax before mapping


def test_to_prediction_abstains_when_all_mass_dropped():
    mapper = LabelMapper(["contempt", "love"])
    p = to_prediction(
        sample_id="s2", model_id="dummy", modality="text",
        native_labels=["contempt", "love"],
        scores=[0.6, 0.4], mapper=mapper,
    )
    assert p.abstained is True
    assert p.pred is None
    assert p.confidence is None
    assert p.scores == {}
    assert p.abstain_reason is not None


def test_prediction_serializes_scores_with_emotion_keys():
    p = Prediction(
        sample_id="s3", model_id="m", modality="text",
        pred_raw="joy", pred=Emotion.HAPPINESS, confidence=0.9,
        scores={Emotion.HAPPINESS: 0.9, Emotion.NEUTRAL: 0.1},
    )
    dumped = p.model_dump(mode="json")
    assert dumped["pred"] == "happiness"
    assert dumped["scores"]["happiness"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.base'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/models/base.py`:
```python
"""Model contract: Prediction, EmotionModel protocol, and to_prediction() builder."""
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.config.emotions import Emotion
from app.data.schema import Sample
from app.labels.mapper import LabelMapper


class Prediction(BaseModel):
    sample_id: str
    model_id: str
    modality: str
    pred_raw: str | None
    pred: Emotion | None
    confidence: float | None
    scores: dict[Emotion, float]
    abstained: bool = False
    abstain_reason: str | None = None


@runtime_checkable
class EmotionModel(Protocol):
    model_id: str
    modality: str

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]: ...


def to_prediction(
    sample_id: str,
    model_id: str,
    modality: str,
    native_labels: Sequence[str],
    scores: Sequence[float],
    mapper: LabelMapper,
) -> Prediction:
    """Build a Prediction from a model's native scores via the mapper.

    Records the native argmax as pred_raw, maps+aggregates into canonical
    scores, then takes the canonical argmax as pred and its value as
    confidence. Abstains (pred=None) when all mass maps to dropped labels.
    """
    native_argmax = native_labels[max(range(len(scores)), key=lambda i: scores[i])]
    canonical = mapper.map_scores(scores)
    if not canonical:
        return Prediction(
            sample_id=sample_id, model_id=model_id, modality=modality,
            pred_raw=native_argmax, pred=None, confidence=None, scores={},
            abstained=True,
            abstain_reason="all probability mass mapped to dropped labels",
        )
    pred = max(canonical, key=canonical.get)
    return Prediction(
        sample_id=sample_id, model_id=model_id, modality=modality,
        pred_raw=native_argmax, pred=pred, confidence=canonical[pred],
        scores=canonical, abstained=False, abstain_reason=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_base.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/models/base.py tests/test_base.py
git commit -m "feat: Prediction, EmotionModel protocol, to_prediction builder"
```

---

### Task 7: PredictionStore (parquet + manifest + cache signature)

**Files:**
- Create: `app/predict/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Prediction`, `Emotion`.
- Produces: `PredictionStore(root: Path)` with:
  - `.signature(model_id: str, config_hash: str, sample_ids: Iterable[str]) -> str` — stable hash of model + config + sample-set.
  - `.path(study: int, corpus: str, model_id: str) -> Path`.
  - `.write(study, corpus, model_id, predictions: list[Prediction], signature: str) -> Path` — writes parquet (scores as JSON column) + records signature in `manifest.json`.
  - `.read(study, corpus, model_id) -> list[Prediction]`.
  - `.existing_sample_ids(study, corpus, model_id) -> set[str]`.
  - `.append(study, corpus, model_id, predictions) -> None` — for resumable long runs.
  - `.is_cached(study, corpus, model_id, signature) -> bool` — True iff parquet exists AND manifest signature matches.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:
```python
from pathlib import Path
from app.config.emotions import Emotion
from app.models.base import Prediction
from app.predict.store import PredictionStore


def _pred(sid, pred=Emotion.HAPPINESS):
    return Prediction(
        sample_id=sid, model_id="m1", modality="text",
        pred_raw="joy", pred=pred, confidence=0.9,
        scores={Emotion.HAPPINESS: 0.9, Emotion.NEUTRAL: 0.1},
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    store = PredictionStore(tmp_path)
    preds = [_pred("a"), _pred("b", Emotion.NEUTRAL)]
    sig = store.signature("m1", "cfg1", ["a", "b"])
    store.write(1, "nEMO", "m1", preds, signature=sig)
    got = store.read(1, "nEMO", "m1")
    assert {p.sample_id for p in got} == {"a", "b"}
    got_b = next(p for p in got if p.sample_id == "b")
    assert got_b.pred is Emotion.NEUTRAL
    assert got_b.scores[Emotion.HAPPINESS] == 0.9


def test_signature_is_order_independent_over_sample_ids(tmp_path: Path):
    store = PredictionStore(tmp_path)
    assert store.signature("m1", "cfg1", ["a", "b"]) == store.signature("m1", "cfg1", ["b", "a"])


def test_signature_changes_with_config(tmp_path: Path):
    store = PredictionStore(tmp_path)
    assert store.signature("m1", "cfg1", ["a"]) != store.signature("m1", "cfg2", ["a"])


def test_is_cached_matches_signature(tmp_path: Path):
    store = PredictionStore(tmp_path)
    sig = store.signature("m1", "cfg1", ["a"])
    assert store.is_cached(1, "nEMO", "m1", sig) is False
    store.write(1, "nEMO", "m1", [_pred("a")], signature=sig)
    assert store.is_cached(1, "nEMO", "m1", sig) is True
    assert store.is_cached(1, "nEMO", "m1", "different-signature") is False


def test_existing_sample_ids_and_append(tmp_path: Path):
    store = PredictionStore(tmp_path)
    sig = store.signature("m1", "cfg1", ["a", "b"])
    store.write(1, "nEMO", "m1", [_pred("a")], signature=sig)
    assert store.existing_sample_ids(1, "nEMO", "m1") == {"a"}
    store.append(1, "nEMO", "m1", [_pred("b")])
    assert store.existing_sample_ids(1, "nEMO", "m1") == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.predict.store'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/predict/store.py`:
```python
"""PredictionStore: cached, versioned parquet store for model predictions."""
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from app.config.emotions import Emotion
from app.models.base import Prediction


class PredictionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    # ----- paths & signatures -----
    def path(self, study: int, corpus: str, model_id: str) -> Path:
        return self.root / f"study{study}" / corpus / f"{model_id}.parquet"

    def signature(self, model_id: str, config_hash: str, sample_ids: Iterable[str]) -> str:
        ids_digest = hashlib.sha256(
            "\n".join(sorted(sample_ids)).encode("utf-8")
        ).hexdigest()
        payload = f"{model_id}|{config_hash}|{ids_digest}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ----- serialization helpers -----
    @staticmethod
    def _to_row(p: Prediction) -> dict:
        d = p.model_dump(mode="json")
        d["scores"] = json.dumps(d["scores"])  # dict -> JSON string column
        return d

    @staticmethod
    def _from_row(row: dict) -> Prediction:
        scores_raw = json.loads(row["scores"]) if row["scores"] else {}
        scores = {Emotion(k): float(v) for k, v in scores_raw.items()}
        return Prediction(
            sample_id=row["sample_id"], model_id=row["model_id"], modality=row["modality"],
            pred_raw=row["pred_raw"], pred=(Emotion(row["pred"]) if row["pred"] else None),
            confidence=(None if pd.isna(row["confidence"]) else float(row["confidence"])),
            scores=scores, abstained=bool(row["abstained"]),
            abstain_reason=(row["abstain_reason"] or None),
        )

    # ----- read / write -----
    def write(self, study: int, corpus: str, model_id: str,
              predictions: list[Prediction], signature: str) -> Path:
        out = self.path(study, corpus, model_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([self._to_row(p) for p in predictions])
        df.to_parquet(out, index=False)
        self._record_manifest(study, corpus, model_id, signature, len(predictions))
        return out

    def read(self, study: int, corpus: str, model_id: str) -> list[Prediction]:
        df = pd.read_parquet(self.path(study, corpus, model_id))
        return [self._from_row(row) for row in df.to_dict(orient="records")]

    def existing_sample_ids(self, study: int, corpus: str, model_id: str) -> set[str]:
        p = self.path(study, corpus, model_id)
        if not p.exists():
            return set()
        return set(pd.read_parquet(p, columns=["sample_id"])["sample_id"].tolist())

    def append(self, study: int, corpus: str, model_id: str,
               predictions: list[Prediction]) -> None:
        p = self.path(study, corpus, model_id)
        new = pd.DataFrame([self._to_row(x) for x in predictions])
        if p.exists():
            existing = pd.read_parquet(p)
            new = pd.concat([existing, new], ignore_index=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(p, index=False)

    # ----- manifest / cache -----
    def _record_manifest(self, study: int, corpus: str, model_id: str,
                          signature: str, n: int) -> None:
        manifest = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        key = f"study{study}/{corpus}/{model_id}"
        manifest[key] = {"signature": signature, "n_samples": n}
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def is_cached(self, study: int, corpus: str, model_id: str, signature: str) -> bool:
        if not self.path(study, corpus, model_id).exists():
            return False
        if not self.manifest_path.exists():
            return False
        manifest = json.loads(self.manifest_path.read_text())
        key = f"study{study}/{corpus}/{model_id}"
        return manifest.get(key, {}).get("signature") == signature
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full P1 suite**

Run: `pytest -v`
Expected: PASS (all tasks green: smoke + emotions + mapping + mapper + schema + base + store).

- [ ] **Step 6: Commit**

```bash
git add app/predict/store.py tests/test_store.py
git commit -m "feat: cached parquet PredictionStore with manifest"
```

---

## Self-Review

**1. Spec coverage (P1 scope only):**
- Canonical Ekman-6+neutral space → Task 2. ✅
- Label harmonization (master map + DROP + none-of-them override) → Task 3. ✅
- Native→canonical score aggregation + drop + renormalize + supported set → Task 4. ✅
- `Sample`/`AudioRef` data contract (audio-on-demand, gold may be None) → Task 5. ✅
- `Prediction` + `EmotionModel` protocol + DRY `to_prediction` + abstention → Task 6. ✅
- Prediction store: parquet, cache by signature (model+config+sample-set), incremental append/resume, manifest → Task 7. ✅
- Pydantic throughout ✅; Python 3.12 venv, launch.json, no-GPU ✅ (Task 1).
- Out of P1 scope (later plans): dataset loaders/audio decode (P2); concrete models + registry + zero-shot backend + benchmark (P3); metrics/kappa/bootstrap/tests/conflicts, common-support exclusion logic, figures/tables (P4). Intentional.

**2. Placeholder scan:** No TBD/TODO; every code + test step contains complete, runnable content. ✅

**3. Type consistency:** `Emotion`, `map_label(raw, overrides)`, `LabelMapper(native_labels, overrides)` / `.supported` / `.map_scores` / `.map_label`, `Sample`/`AudioRef`, `Prediction` / `to_prediction(...)`, `PredictionStore` method signatures are consistent across Tasks 2–7 and match the File Structure interface block. ✅

## Otwarte punkty przeniesione do kolejnych planów

- Wybór backendu Qwen2-Audio (MLX-4bit vs API) + mikro-benchmark → **P3**.
- Weryfikacja tokenizera autorskiego LSTM (`finalwordindex.pkl` vs `test.h5`) → **P3**.
- Reguła wspólnego wsparcia klas i wykluczenia w liczeniu κ → **P4**.
- StandardScaler autorskiego modelu mowy → **P3**.
- Zakres pobrania RAVDESS z Zenodo + pipeline klatek twarzy → **P2/P3**.
