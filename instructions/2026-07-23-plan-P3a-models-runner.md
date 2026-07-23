# P3a — Model interface + prediction runner + EN text — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zbudować maszynerię predykcji — konkretny model HF do klasyfikacji tekstu (jako `EmotionModel`), rejestr modeli i cache'owany, wznawialny runner — i dowieźć pierwszą modalność (EN-tekst, j-hartmann) end-to-end na realnych danych do magazynu predykcji z P1.

**Architecture:** Etap `predict` z Podejścia A. `HFTextModel` implementuje protokół `EmotionModel` (P1): buduje `LabelMapper` z `id2label` modelu, tokenizuje `Sample.text`, liczy softmax, mapuje na przestrzeń kanoniczną przez `to_prediction`. `run_prediction` czyta indeks (P2), pobiera model z rejestru, i zapisuje predykcje do `PredictionStore` (P1) z cache po sygnaturze i wznawianiem.

**Tech Stack:** Python 3.12, PyTorch (CPU/MPS), Transformers, Pydantic v2, pytest. Bez GPU.

## Global Constraints

- Python **3.12**; jeden `.venv`; Mac M1 — **device-agnostic** (`mps` jeśli dostępne, inaczej `cpu`); bez CUDA.
- Buduje na P1 (`EmotionModel`, `Prediction`, `to_prediction`, `LabelMapper`, `PredictionStore`, `Emotion`, `map_label`) i P2 (`Settings`, `read_index`).
- Etykiety mapowane **wyłącznie** przez `LabelMapper`/`map_label` (fail-loud na nieznanej). `native_labels` brane z `id2label` modelu (kolejność wg indeksu) — **przypinane z konfiguracji przy ładowaniu**, nie zgadywane.
- Predykcje **cache'owane** (P1 store): sygnatura = `signature(model_id, config_hash, sample_ids)`; `is_cached` przed liczeniem; **zapis przyrostowy + wznawianie** (pomijanie policzonych `sample_id`) — potrzebne dla drogich modeli w P3b/P3d.
- **Determinizm:** `torch.no_grad`, brak samplingu (klasyfikatory). Model w `eval()`.
- Pobrania HF lądują w `models/hf_cache/` (folder w `.gitignore`), nie w `~/.cache`.
- Kod/testy/commity po angielsku; **żadnego `Co-Authored-By`**. `datasets/`,`models/`,`old_projects/` nietykalne/niewersjonowane.

## Review mode (per memory: economical-subagent-execution)

Predykcje karmią wyniki → **pełna recenzja per-task dla T2–T4** (wrapper modelu, runner/cache, integracja na realnym modelu). **T1 (zależności + właściwość cache-dir) — wiring neutralny → bez recenzenta per-task, pokryty finalnym review.** Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus.

---

## File Structure

```
app/
  config/settings.py        # += hf_home property (models/hf_cache)  (T1)
  models/
    text/__init__.py
    text/hf_text.py          # HFTextModel(EmotionModel): id2label -> LabelMapper -> to_prediction  (T2)
    registry.py              # build_model(model_id) -> EmotionModel; MODEL_IDS  (T2)
  predict/
    runner.py                # run_prediction(...) with cache + resume; chunked batching  (T3)
  cli.py                     # `predict` subcommand (thin wrapper over runner)  (T4)
tests/
  test_hf_text.py            # native-labels-from-config + real j-hartmann inference (downloads model)
  test_registry.py
  test_runner.py             # cache/resume/force with a stub EmotionModel (no download)
  test_cli_predict.py        # end-to-end: real model over a small real ingested subset
requirements.txt             # += torch, transformers, huggingface-hub, safetensors
```

Interfejsy:
- `app.config.settings.Settings.hf_home -> Path` (`models/hf_cache`).
- `app.models.text.hf_text.native_labels_from_id2label(id2label: dict) -> list[str]` (pure).
- `app.models.text.hf_text.HFTextModel(model_id: str, hf_repo: str, cache_dir: Path|None=None)` — attrs `model_id`, `modality="text"`, `native_labels`, `mapper`; methods `config_signature() -> str`, `predict_batch(samples) -> list[Prediction]`.
- `app.models.registry.build_model(model_id: str) -> EmotionModel`; `MODEL_IDS: list[str]`.
- `app.predict.runner.run_prediction(samples, model, store, study, corpus, *, force=False, batch_size=32) -> str` (returns `"cached"` | `"done"`); `chunked(seq, n)`.

---

### Task 1: ML dependencies + HF cache location  [economize: no per-task reviewer]

**Files:**
- Modify: `requirements.txt`, `app/config/settings.py`
- Test: `tests/test_settings.py` (append)

**Interfaces:**
- Produces: `Settings.hf_home -> datasets_root.parent/"models"/"hf_cache"` — actually `repo_root/"models"/"hf_cache"`; created on access.

- [ ] **Step 1: Add deps and install**

Append to `requirements.txt`:
```
torch>=2.2,<3
transformers>=4.44,<5
huggingface-hub>=0.24,<1
safetensors>=0.4,<1
```
Run: `.venv/bin/pip install -r requirements.txt` then
`.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.backends.mps.is_available())"`
Expected: prints versions and `True` (MPS available on M1) or `False` (falls back to CPU — still fine).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_settings.py`:
```python
def test_hf_home_under_models(tmp_path):
    from app.config.settings import Settings
    s = Settings(datasets_root=tmp_path / "datasets", results_root=tmp_path / "results")
    assert s.hf_home.name == "hf_cache"
    assert s.hf_home.parent.name == "models"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_settings.py::test_hf_home_under_models -v`
Expected: FAIL (AttributeError: hf_home).

- [ ] **Step 4: Add the property**

In `app/config/settings.py`, add a repo-root-relative property (reuse the existing `_REPO_ROOT`):
```python
    @property
    def hf_home(self) -> Path:
        return _REPO_ROOT / "models" / "hf_cache"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/config/settings.py tests/test_settings.py
git commit -m "chore: add torch/transformers deps and HF cache location"
```

---

### Task 2: HFTextModel wrapper + registry

**Files:**
- Create: `app/models/text/__init__.py`, `app/models/text/hf_text.py`, `app/models/registry.py`
- Test: `tests/test_hf_text.py`, `tests/test_registry.py`

**Interfaces:**
- `native_labels_from_id2label(id2label)` — return label strings ordered by integer key (config `id2label` may have string keys `"0".."6"`).
- `HFTextModel` — loads tokenizer + `AutoModelForSequenceClassification` (in `eval()`, on mps/cpu), `native_labels` from config, `mapper=LabelMapper(native_labels)`. `config_signature()` returns `hf_repo` (identifies the weights). `predict_batch` tokenizes texts (padding+truncation), forward under `no_grad`, softmax, and builds one `Prediction` per sample via `to_prediction`.
- `registry.build_model(model_id)` — `"en_text_hartmann"` → `HFTextModel("en_text_hartmann", "j-hartmann/emotion-english-distilroberta-base", cache_dir=Settings().hf_home)`; unknown id raises `KeyError`. `MODEL_IDS` lists registered ids.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_text.py`:
```python
import numpy as np
import pytest
from app.config.emotions import Emotion
from app.models.text.hf_text import native_labels_from_id2label, HFTextModel


def test_native_labels_ordered_by_int_key():
    id2label = {"2": "fear", "0": "anger", "1": "disgust"}
    assert native_labels_from_id2label(id2label) == ["anger", "disgust", "fear"]


@pytest.mark.slow
def test_hartmann_predicts_happiness_on_happy_text(tmp_path):
    # Downloads the real model (~330MB) into a temp HF cache; cached across runs.
    m = HFTextModel("en_text_hartmann",
                    "j-hartmann/emotion-english-distilroberta-base",
                    cache_dir=tmp_path)
    from app.data.schema import Sample, AudioRef
    from pathlib import Path
    s = Sample(sample_id="t:1", studies=[1], corpus="t", language="en",
               text="I am so happy and delighted today!",
               audio=AudioRef(wav_path=Path("a.wav")), gold_raw="happy", gold=Emotion.HAPPINESS)
    [pred] = m.predict_batch([s])
    assert pred.modality == "text"
    assert pred.pred is Emotion.HAPPINESS        # joy -> happiness, clearly happy text
    assert 0.0 < pred.confidence <= 1.0
    assert set(pred.scores).issubset(set(Emotion))
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
```

Create `tests/test_registry.py`:
```python
import pytest
from app.models.registry import build_model, MODEL_IDS


def test_registry_lists_and_unknown_raises():
    assert "en_text_hartmann" in MODEL_IDS
    with pytest.raises(KeyError):
        build_model("nonexistent_model")


@pytest.mark.slow
def test_build_hartmann():
    m = build_model("en_text_hartmann")
    assert m.model_id == "en_text_hartmann" and m.modality == "text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hf_text.py tests/test_registry.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Register the `slow` marker**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add:
```toml
markers = ["slow: tests that download models or run heavy inference"]
```

- [ ] **Step 4: Write the implementation**

Create `app/models/text/__init__.py`:
```python
"""Text emotion models."""
```

Create `app/models/text/hf_text.py`:
```python
"""Generic Hugging Face sequence-classification text emotion model."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction


def native_labels_from_id2label(id2label: dict) -> list[str]:
    return [id2label[k] for k in sorted(id2label, key=lambda x: int(x))]


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class HFTextModel:
    modality = "text"

    def __init__(self, model_id: str, hf_repo: str, cache_dir: Path | None = None):
        self.model_id = model_id
        self.hf_repo = hf_repo
        cd = str(cache_dir) if cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, cache_dir=cd)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_repo, cache_dir=cd)
        self.device = _device()
        self.model.to(self.device).eval()
        self.native_labels = native_labels_from_id2label(self.model.config.id2label)
        self.mapper = LabelMapper(self.native_labels)

    def config_signature(self) -> str:
        return self.hf_repo

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        texts = [s.text for s in samples]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i].tolist(), self.mapper)
            for i, s in enumerate(samples)
        ]
```

Create `app/models/registry.py`:
```python
"""Model registry: model_id -> concrete EmotionModel factory (fixed roster)."""
from collections.abc import Callable

from app.config.settings import Settings
from app.models.base import EmotionModel
from app.models.text.hf_text import HFTextModel

_settings = Settings()

_FACTORIES: dict[str, Callable[[], EmotionModel]] = {
    "en_text_hartmann": lambda: HFTextModel(
        "en_text_hartmann",
        "j-hartmann/emotion-english-distilroberta-base",
        cache_dir=_settings.hf_home,
    ),
}

MODEL_IDS: list[str] = list(_FACTORIES)


def build_model(model_id: str) -> EmotionModel:
    if model_id not in _FACTORIES:
        raise KeyError(f"Unknown model_id: {model_id!r}. Known: {MODEL_IDS}")
    return _FACTORIES[model_id]()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hf_text.py tests/test_registry.py -v`
Expected: PASS (the `slow` tests download j-hartmann once, then pass; the fast tests pass immediately).

- [ ] **Step 6: Commit**

```bash
git add app/models/text app/models/registry.py tests/test_hf_text.py tests/test_registry.py pyproject.toml
git commit -m "feat: HFTextModel wrapper and model registry"
```

---

### Task 3: Prediction runner (cache + resume)

**Files:**
- Create: `app/predict/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- `chunked(seq, n)` — yield successive `n`-sized lists.
- `run_prediction(samples, model, store, study, corpus, *, force=False, batch_size=32) -> str`:
  - `cfg = model.config_signature()`; `target_ids = [s.sample_id for s in samples]`.
  - If not `force` and `store.is_cached(study, corpus, model.model_id, cfg, target_ids)` → return `"cached"`.
  - `done = set()` if `force` else `store.existing_sample_ids(...)`; `todo = [s for s in samples if s.sample_id not in done]`.
  - Predict `todo` in `batch_size` chunks; for each chunk: `store.write(...)` if the store path does not exist yet (first write — with `cfg`), else `store.append(...)`. (`append` re-stamps the manifest over all on-disk ids, so after completion `is_cached(target_ids)` is True.)
  - Return `"done"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:
```python
from pathlib import Path
from app.config.emotions import Emotion
from app.data.schema import Sample, AudioRef
from app.models.base import Prediction
from app.predict.store import PredictionStore
from app.predict.runner import run_prediction, chunked


class StubModel:
    model_id = "stub"
    modality = "text"
    def __init__(self):
        self.calls = 0
    def config_signature(self):
        return "cfg-v1"
    def predict_batch(self, samples):
        self.calls += len(samples)
        return [Prediction(sample_id=s.sample_id, model_id=self.model_id, modality="text",
                           pred_raw="joy", pred=Emotion.HAPPINESS, confidence=0.9,
                           scores={Emotion.HAPPINESS: 0.9, Emotion.NEUTRAL: 0.1})
                for s in samples]


def _samples(n):
    return [Sample(sample_id=f"c:{i}", studies=[1], corpus="c", language="en", text="t",
                   audio=AudioRef(wav_path=Path("a.wav")), gold_raw="happy", gold=Emotion.HAPPINESS)
            for i in range(n)]


def test_chunked():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_run_then_cached(tmp_path):
    store = PredictionStore(tmp_path)
    m = StubModel()
    assert run_prediction(_samples(5), m, store, 1, "c", batch_size=2) == "done"
    assert m.calls == 5
    assert len(store.read(1, "c", "stub")) == 5
    # second run: fully cached, model not called again
    m2 = StubModel()
    assert run_prediction(_samples(5), m2, store, 1, "c") == "cached"
    assert m2.calls == 0


def test_resume_only_predicts_missing(tmp_path):
    store = PredictionStore(tmp_path)
    m = StubModel()
    run_prediction(_samples(3), m, store, 1, "c", batch_size=10)  # predicts 3
    m2 = StubModel()
    # now ask for 5 (superset): only the 2 new ones are predicted
    assert run_prediction(_samples(5), m2, store, 1, "c", batch_size=10) == "done"
    assert m2.calls == 2
    assert len(store.read(1, "c", "stub")) == 5


def test_force_recomputes(tmp_path):
    store = PredictionStore(tmp_path)
    run_prediction(_samples(3), StubModel(), store, 1, "c")
    m = StubModel()
    assert run_prediction(_samples(3), m, store, 1, "c", force=True) == "done"
    assert m.calls == 3  # recomputed despite cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `app/predict/runner.py`:
```python
"""Prediction runner: cache-aware, resumable batched inference into the store."""
from collections.abc import Iterable, Iterator

from app.data.schema import Sample
from app.models.base import EmotionModel
from app.predict.store import PredictionStore


def chunked(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run_prediction(samples: list[Sample], model: EmotionModel, store: PredictionStore,
                   study: int, corpus: str, *, force: bool = False,
                   batch_size: int = 32) -> str:
    cfg = model.config_signature()
    target_ids = [s.sample_id for s in samples]
    if not force and store.is_cached(study, corpus, model.model_id, cfg, target_ids):
        return "cached"
    done = set() if force else store.existing_sample_ids(study, corpus, model.model_id)
    todo = [s for s in samples if s.sample_id not in done]
    first = not done  # force or empty store -> the first write (re)creates the file
    for batch in chunked(todo, batch_size):
        preds = model.predict_batch(batch)
        if first:
            store.write(study, corpus, model.model_id, preds, cfg)
            first = False
        else:
            store.append(study, corpus, model.model_id, preds)
    return "done"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add app/predict/runner.py tests/test_runner.py
git commit -m "feat: cache-aware resumable prediction runner"
```

---

### Task 4: CLI `predict` + end-to-end real integration

**Files:**
- Create: `app/cli.py`
- Test: `tests/test_cli_predict.py`

**Interfaces:**
- `app.cli.main(argv=None)` — argparse with a `predict` subcommand: `--study`, `--corpus`, `--model`, optional `--index` (path, defaults to `Settings().index_dir/f"{corpus}.parquet"`), `--force`, `--batch-size`. Reads the index (P2 `read_index`), builds the model (registry), runs `run_prediction` into a `PredictionStore(Settings().results_root/"predictions")`, prints a one-line summary.
- Runnable as `python -m app.cli predict --study 1 --corpus RAVDESS --model en_text_hartmann`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_predict.py`:
```python
import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.data.ravdess import load_ravdess
from app.data.index_io import write_index
from app.predict.store import PredictionStore
from app.cli import main

_HAS_DATA = bool(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA, reason="RAVDESS dataset not present")
def test_cli_predict_en_text_on_ravdess_subset(tmp_path, capsys):
    # Build a tiny real index (3 Actor_01 utterances) and predict EN text end-to-end.
    samples = load_ravdess(Path("datasets/RAVDESS/Audio_Speech_Actors_01-24"),
                           Path("datasets/RAVDESS"))
    subset = [s for s in samples if s.speaker_id == "ravdess_actor_01"][:3]
    for s in subset:                      # resolve gold so the index is realistic
        pass
    index = tmp_path / "RAVDESS.parquet"
    write_index(subset, index)
    results = tmp_path / "results"
    rc = main(["predict", "--study", "1", "--corpus", "RAVDESS",
               "--model", "en_text_hartmann", "--index", str(index),
               "--results-root", str(results)])
    assert rc == 0
    store = PredictionStore(results / "predictions")
    preds = store.read(1, "RAVDESS", "en_text_hartmann")
    assert len(preds) == 3
    for p in preds:
        assert p.pred in set(Emotion) and p.modality == "text"
    # RAVDESS text is the fixed neutral sentence -> model should lean neutral-ish,
    # but we only assert validity here, not a specific label (text carries little emotion).
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_predict.py -v`
Expected: FAIL (ModuleNotFoundError: app.cli).

- [ ] **Step 3: Write the implementation**

Create `app/cli.py`:
```python
"""Command-line entry point for the research pipeline."""
import argparse
from pathlib import Path

from app.config.settings import Settings
from app.data.index_io import read_index
from app.models.registry import build_model
from app.predict.runner import run_prediction
from app.predict.store import PredictionStore


def _predict(args) -> int:
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    index = Path(args.index) if args.index else settings.index_dir / f"{args.corpus}.parquet"
    samples = read_index(index)
    model = build_model(args.model)
    store = PredictionStore(results_root / "predictions")
    status = run_prediction(samples, model, store, args.study, args.corpus,
                            force=args.force, batch_size=args.batch_size)
    print(f"predict {args.model} on {args.corpus} (study {args.study}): "
          f"{status}, {len(samples)} samples -> {store.path(args.study, args.corpus, args.model)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Run a model over a corpus index")
    p.add_argument("--study", type=int, required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--index", default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=_predict)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_predict.py -v`
Expected: PASS (downloads j-hartmann if not cached, predicts 3 real RAVDESS-text samples).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green (P1+P2+P2-RAVDESS+P3a; `slow` tests run and pass).

- [ ] **Step 6: Commit**

```bash
git add app/cli.py tests/test_cli_predict.py
git commit -m "feat: CLI predict command with end-to-end EN-text integration"
```

---

## Self-Review

**1. Spec coverage (P3a scope):**
- `EmotionModel` realized by a concrete model (`HFTextModel`), `native_labels` pinned from config `id2label`, scores mapped via `LabelMapper`/`to_prediction` → T2. ✅
- Model registry (id → model) → T2. ✅
- Cache-aware, resumable, batched prediction runner writing to P1 store → T3. ✅
- CLI `predict` + end-to-end on real data (EN text on real RAVDESS subset) → T4. ✅
- device-agnostic (mps/cpu), deterministic (no_grad, eval), HF cache in models/ → T1/T2. ✅
- Out of scope (later sub-plans): Qwen zero-shot backend + benchmark (P3b); PL text models + face ViT (P3c); custom speech CNN + speechbrain (P3d); custom LSTM (P3e, optional).

**2. Placeholder scan:** No TBD/TODO; every code + test step is complete and runnable. (The `slow`-marked tests download the real j-hartmann model — this is intentional real-behavior verification, cached after first run.)

**3. Type consistency:** `EmotionModel`/`Prediction`/`to_prediction`/`LabelMapper`/`PredictionStore` (P1), `Settings`/`read_index` (P2) used per their signatures. `HFTextModel.config_signature()`/`predict_batch`, `build_model`/`MODEL_IDS`, `run_prediction`/`chunked` match the interface block and tests. Store calls use the P1 contract `write(study,corpus,model_id,preds,config_hash)` / `append(...)` / `is_cached(study,corpus,model_id,config_hash,expected_ids)`.

## Otwarte punkty / do kolejnych pod-planów
- **P3b:** `SpeechZeroShotBackend` (Qwen2-Audio) + mikro-benchmark (MLX-4bit vs API) — rozstrzyga feasibility M1.
- **P3c:** PL text (roberta6, tabularisai, poltextlab — po dostępie HF; przypiąć `id2label`/kolejność LABEL_0..5 dla visegradmedia) + twarz (trpakov ViT na cropach z `extract_face_frames`, agregacja softmax po klatkach, `num_frames`).
- **P3d:** autorski CNN-Transformer (przepisać `ParallelModel`/`HybridModel` z notebooków; log-mel `sr=48000` 128×563; **resampling audio z natywnego sr do 48 kHz**; StandardScaler — strategia normalizacji) + speechbrain IEMOCAP (4 klasy).
- **P3e (opcjonalny):** autorski LSTM — rekonstrukcja word_index z word2vec (finalwordindex.pkl ma 5364 słów vs Embedding 294182 — niezgodny); TF/Keras na M1.
- Uruchomienie pełnej ingestii (`results/index/*.parquet`) dla wszystkich korpusów przed masowym `predict` (obecnie indeksy powstają ad hoc w testach/CLI).
