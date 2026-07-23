# P3e — Custom LSTM text model (isolated Keras inference) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Udostępnić autorski LSTM tekstowy (`model.h5` + `finalwordindex.pkl`, wytrenowany w `.venv-tf`) jako `EmotionModel` w pipeline — bez wciągania TensorFlow do głównego `.venv` (izolacja przez podproces).

**Architecture:** Standalone skrypt inferencyjny (`infer.py`) uruchamiany interpreterem `.venv-tf` (ma TF/Keras): czyta JSON `{"texts": [...]}` ze stdin, ładuje `model.h5` + tokenizer (Keras `Tokenizer` z zapisanym `word_index`), tokenizuje (maxlen 30), predykuje, zwraca JSON `{"labels": [...], "probs": [[...]]}` na stdout. Wrapper `LSTMTextModel` w głównym `.venv` woła ten skrypt przez `subprocess`, parsuje prawdopodobieństwa i mapuje przez `LabelMapper`/`to_prediction`.

**Tech Stack:** Python 3.12 (główny `.venv`, wrapper — tylko stdlib + P1/P2); Python 3.11 `.venv-tf` (TF 2.15/Keras 2 — tylko infer.py). Pydantic v2, pytest.

## Global Constraints

- Główny `.venv` (torch) **NIE dostaje TF** — izolacja przez podproces do `.venv-tf`. `infer.py` jest **standalone** (importuje tylko `tensorflow`, `numpy`, `pickle`, `json`, `sys`, `pathlib` — NIC z pakietu `app`, bo `app` ciągnie torch).
- Etykiety w kolejności wyjścia modelu (z `main_newest.py`): `["neutral","happiness","sadness","fear","anger","disgust","surprise"]`. Mapowane przez `LabelMapper`/`map_label` (happiness→happiness, reszta wprost) → wszystkie 7 wspierane.
- Tokenizacja MUSI odtwarzać trening: `Tokenizer()` z domyślnymi ustawieniami (lower=True, filtry interpunkcji), `word_index` ustawiony z `finalwordindex.pkl`, `texts` zlowercase'owane, `pad_sequences(maxlen=30)`.
- Wagi/tokenizer czytane read-only z `old_projects/.../results/nkjp+wiki-forms-all-300-cbow-ns_v2/`.
- `TF_CPP_MIN_LOG_LEVEL=3` w podprocesie (ciche logi TF); stdout tylko czysty JSON.
- Kod/testy/commity po angielsku; **żadnego `Co-Authored-By`**.
- **Model DRUGORZĘDNY** (jeden z 4 modeli PL-tekst; nagłówek = tabularisai). Ulepszenia → spec §9 (na koniec, tylko podmiana `model.h5`).

## Review mode (per memory: economical-subagent-execution)

LSTM karmi wyniki (PL tekst) + poprawność tokenizacji jest krytyczna → **pełna recenzja T1 i T2**. Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus.

---

## File Structure

```
app/
  config/settings.py           # += lstm_model_dir, tf_python, lstm_infer_script properties
  models/text/lstm/__init__.py
  models/text/lstm/infer.py     # STANDALONE Keras inference (run by .venv-tf)
  models/text/lstm/wrapper.py   # LSTMTextModel (main venv, subprocess -> .venv-tf)
  models/registry.py            # += pl_text_lstm
tests/
  test_lstm_infer.py            # runs infer.py via .venv-tf on real model (slow/skip)
  test_lstm_wrapper.py          # LSTMTextModel via registry end-to-end (slow/skip)
```

Interfejsy:
- `Settings.lstm_model_dir -> old_projects/.../results/nkjp+wiki-forms-all-300-cbow-ns_v2`; `Settings.tf_python -> repo_root/.venv-tf/bin/python`; `Settings.lstm_infer_script -> repo_root/app/models/text/lstm/infer.py`.
- `infer.py`: CLI `python infer.py <model_dir>`, stdin `{"texts":[...]}`, stdout `{"labels":[...], "probs":[[...]]}`.
- `LSTMTextModel(model_id, model_dir, tf_python, infer_script)` — `modality="text"`, `config_signature()`, `predict_batch(samples)->list[Prediction]` (subprocess).

---

### Task 1: Standalone Keras inference script + Settings paths

**Files:**
- Modify: `app/config/settings.py`
- Create: `app/models/text/lstm/__init__.py`, `app/models/text/lstm/infer.py`
- Test: `tests/test_lstm_infer.py`

**Interfaces:** see Global. `infer.py` reconstructs the Keras `Tokenizer` from the pickled `word_index` and reproduces training tokenization exactly.

- [ ] **Step 1: Add Settings paths**

In `app/config/settings.py` add (reuse `_REPO_ROOT`):
```python
    @property
    def lstm_model_dir(self) -> Path:
        return (_REPO_ROOT / "old_projects" / "chatgpt-pl-emotion-recognition"
                / "results" / "nkjp+wiki-forms-all-300-cbow-ns_v2")

    @property
    def tf_python(self) -> Path:
        return _REPO_ROOT / ".venv-tf" / "bin" / "python"

    @property
    def lstm_infer_script(self) -> Path:
        return _REPO_ROOT / "app" / "models" / "text" / "lstm" / "infer.py"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_lstm_infer.py`:
```python
import json
import subprocess
import pytest
from pathlib import Path
from app.config.settings import Settings

_S = Settings()
_HAS = (_S.lstm_model_dir / "model.h5").exists() and _S.tf_python.exists()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS, reason="trained LSTM model or .venv-tf not present")
def test_infer_script_runs_via_tf_venv():
    payload = {"texts": ["jestem bardzo szczęśliwy i zadowolony z dzisiejszego dnia",
                          "jestem smutny i przygnębiony, wszystko stracone"]}
    proc = subprocess.run(
        [str(_S.tf_python), str(_S.lstm_infer_script), str(_S.lstm_model_dir)],
        input=json.dumps(payload), text=True, capture_output=True,
        env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout)                       # stdout must be pure JSON
    assert out["labels"] == ["neutral", "happiness", "sadness", "fear", "anger", "disgust", "surprise"]
    assert len(out["probs"]) == 2 and len(out["probs"][0]) == 7
    # clearly-happy sentence -> happiness is the argmax (model has ~97% happiness recall)
    assert out["labels"][max(range(7), key=lambda i: out["probs"][0][i])] == "happiness"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lstm_infer.py -v`
Expected: FAIL (infer.py missing → subprocess returncode != 0, or skip if model absent — if skipped, create the file then it runs).

- [ ] **Step 4: Write the implementation**

Create `app/models/text/lstm/__init__.py`:
```python
"""Custom LSTM text model (isolated Keras inference via .venv-tf subprocess)."""
```

Create `app/models/text/lstm/infer.py`:
```python
"""Standalone Keras LSTM inference. Run with the TF venv (.venv-tf), NOT the main venv.

Reads {"texts": [...]} JSON on stdin, writes {"labels": [...], "probs": [[...]]} JSON on
stdout. Reconstructs the Keras Tokenizer from the saved word_index and reproduces the
training-time tokenization (default Tokenizer preprocessing, lowercased text, maxlen 30).

Usage: python infer.py <model_dir>     # model_dir has model.h5 + finalwordindex.pkl
Imports ONLY tensorflow/numpy/stdlib — never the `app` package (which pulls in torch).
"""
import json
import os
import pickle
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MAXLEN = 30
LABELS = ["neutral", "happiness", "sadness", "fear", "anger", "disgust", "surprise"]


def main() -> None:
    model_dir = sys.argv[1]
    payload = json.load(sys.stdin)
    texts = [str(t).lower() for t in payload["texts"]]

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import pad_sequences

    model = load_model(f"{model_dir}/model.h5")
    with open(f"{model_dir}/finalwordindex.pkl", "rb") as f:
        word_index = pickle.load(f)
    tokenizer = Tokenizer()          # default preprocessing (lower + punctuation filters), matches training
    tokenizer.word_index = word_index
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=MAXLEN)

    probs = model.predict(x, verbose=0)
    json.dump({"labels": LABELS, "probs": probs.tolist()}, sys.stdout)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_lstm_infer.py -v`
Expected: PASS (the slow test spawns `.venv-tf` python, loads the real model, predicts; happy sentence → happiness). If the happy sentence does NOT map to happiness, STOP and report the actual output (do not weaken the assertion silently).

- [ ] **Step 6: Commit**

```bash
git add app/config/settings.py app/models/text/lstm/__init__.py app/models/text/lstm/infer.py tests/test_lstm_infer.py
git commit -m "feat: standalone Keras LSTM inference script (isolated in .venv-tf)"
```

---

### Task 2: LSTMTextModel wrapper + registry

**Files:**
- Create: `app/models/text/lstm/wrapper.py`
- Modify: `app/models/registry.py`
- Test: `tests/test_lstm_wrapper.py`

**Interfaces:** `LSTMTextModel` (see Global). `predict_batch` shells out to `infer.py` via `.venv-tf` python, parses probabilities, and builds one `Prediction` per sample via `to_prediction` (LabelMapper renormalizes the 7 sigmoid outputs over the supported canonical set). Registry: `pl_text_lstm`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lstm_wrapper.py`:
```python
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.config.settings import Settings

_S = Settings()
_HAS = (_S.lstm_model_dir / "model.h5").exists() and _S.tf_python.exists()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS, reason="trained LSTM model or .venv-tf not present")
def test_lstm_wrapper_predicts_via_registry():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("pl_text_lstm")
    assert m.modality == "text"
    assert m.mapper.supported == set(Emotion)   # all 7 (happiness merge only)
    s = Sample(sample_id="pl:1", studies=[1], corpus="nEMO", language="pl",
               text="jestem bardzo szczęśliwy i zadowolony z dzisiejszego dnia",
               audio=AudioRef(wav_path=Path("a.wav")), gold_raw="happiness", gold=Emotion.HAPPINESS)
    [pred] = m.predict_batch([s])
    assert pred.pred is Emotion.HAPPINESS
    assert pred.modality == "text"
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5     # renormalized over supported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lstm_wrapper.py -v`
Expected: FAIL (ImportError / KeyError unknown model_id).

- [ ] **Step 3: Write the implementation**

Create `app/models/text/lstm/wrapper.py`:
```python
"""LSTMTextModel: EmotionModel that runs the isolated Keras inference subprocess."""
import json
import subprocess
from pathlib import Path

from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction

NATIVE_LABELS = ["neutral", "happiness", "sadness", "fear", "anger", "disgust", "surprise"]


class LSTMTextModel:
    modality = "text"

    def __init__(self, model_id: str, model_dir: Path, tf_python: Path, infer_script: Path):
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.tf_python = Path(tf_python)
        self.infer_script = Path(infer_script)
        self.native_labels = NATIVE_LABELS
        self.mapper = LabelMapper(NATIVE_LABELS)

    def config_signature(self) -> str:
        return f"lstm:{self.model_dir.name}"

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        texts = [s.text for s in samples]
        proc = subprocess.run(
            [str(self.tf_python), str(self.infer_script), str(self.model_dir)],
            input=json.dumps({"texts": texts}), text=True, capture_output=True,
            env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin"},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LSTM inference subprocess failed:\n{proc.stderr[-800:]}")
        out = json.loads(proc.stdout)
        if out["labels"] != self.native_labels:
            raise ValueError(f"Unexpected label order from infer.py: {out['labels']}")
        probs = out["probs"]
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i], self.mapper)
            for i, s in enumerate(samples)
        ]
```

- [ ] **Step 4: Register the model**

In `app/models/registry.py`:
```python
from app.models.text.lstm.wrapper import LSTMTextModel
```
```python
    "pl_text_lstm": lambda: LSTMTextModel(
        "pl_text_lstm", _settings.lstm_model_dir, _settings.tf_python, _settings.lstm_infer_script),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_lstm_wrapper.py -v`
Expected: PASS (builds the model, runs the subprocess on a happy Polish sentence → HAPPINESS, scores renormalized). If HAPPINESS assertion fails, STOP and report actual output.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green (P1..P3e). slow LSTM tests run only if the model + .venv-tf are present.

- [ ] **Step 7: Commit**

```bash
git add app/models/text/lstm/wrapper.py app/models/registry.py tests/test_lstm_wrapper.py
git commit -m "feat: LSTMTextModel wrapper (subprocess-isolated) + register pl_text_lstm"
```

---

## Self-Review

**1. Spec coverage (P3e scope):**
- Custom LSTM available as `EmotionModel` (`pl_text_lstm`) without TF in the main venv (subprocess isolation) → T1/T2. ✅
- Faithful tokenization (reconstructed Keras Tokenizer + word_index, maxlen 30, lowercased) → T1. ✅
- Native label order from training; single-source LabelMapper mapping (7 sigmoid outputs → renormalized canonical) → T2. ✅
- Weights read read-only from old_projects results dir → Settings. ✅
- End-stage improvements (softmax/categorical, early stopping, class weights, XED validation) recorded in spec §9 as swap-in-only. ✅
- Out of scope: Qwen zero-shot + benchmark (P3b); analysis/report (P4).

**2. Placeholder scan:** No TBD/TODO; complete code + tests. `slow`+`skipif` guards keep it runnable without the model/.venv-tf. Two STOP-and-report guards (subprocess failure; happiness-argmax mismatch) protect correctness.

**3. Type consistency:** `EmotionModel` contract satisfied by `LSTMTextModel`; `to_prediction`/`LabelMapper`/`Prediction`/`Sample` (P1), `Settings` used per signatures. `infer.py` stays standalone (no `app` imports). Native label order identical in `infer.py` (LABELS) and `wrapper.py` (NATIVE_LABELS) — both `[neutral,happiness,sadness,fear,anger,disgust,surprise]`; the wrapper asserts they match at runtime.

## Otwarte punkty
- Wydajność: podproces ładuje `model.h5` na każde wywołanie `predict_batch` (~1 s). Dla pełnego korpusu uruchamiać z dużym `--batch-size` (jedno wywołanie podprocesu). Nie wpływa na poprawność.
- Ulepszenia modelu (softmax+categorical, early stopping, walidacja XED) — spec §9, na sam koniec, tylko podmiana `model.h5`.
