# P3d — Speech models (custom CNN-Transformer + speechbrain IEMOCAP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dodać modele mowy: (a) autorski `ParallelModel` (CNN-Transformer, PyTorch) z wagami z `old_projects`, wiernie odtworzony (log-mel 48 kHz + StandardScaler per-korpus), i (b) speechbrain wav2vec2 IEMOCAP (4 klasy, „czysty" model EN).

**Architecture:** Oba implementują protokół `EmotionModel` (P1). Autorski model: `load_audio` (P2) → resample 48 kHz + offset0.5/dur3/pad → log-mel (128×563) → StandardScaler dopasowany na korpusie → `ParallelModel` → softmax → `to_prediction`. speechbrain: `load_audio` → 16 kHz → klasyfikator → prawdopodobieństwa 4 klas → `to_prediction`.

**Tech Stack:** Python 3.12, PyTorch (CPU/MPS), librosa (log-mel + resampling), scikit-learn (StandardScaler), speechbrain, Pydantic v2, pytest.

## Global Constraints

- Python 3.12; główny `.venv`; device-agnostic (mps/cpu). Buduje na P1 (`EmotionModel`,`to_prediction`,`LabelMapper`), P2 (`load_audio`,`Settings`), P3a (rejestr, `native_labels_from_id2label` nieużywane tu).
- Etykiety mapowane **wyłącznie** przez `LabelMapper`/`map_label` (fail-loud).
- **StandardScaler: refit per-korpus** — oryginalny scaler treningowy nieodtwarzalny (nie zapisany; split bez ziarna). Jawne przybliżenie, udokumentowane. `config_signature` zawiera `scaler=per-corpus`. **Konsekwencja:** autorski model mowy musi dostać cały korpus w jednym `predict_batch` (scaler dopasowywany na przekazanej partii) — uruchamiać z `--batch-size` ≥ rozmiar korpusu.
- **Leakage (świadomy):** autorski model trenowany na RAVDESS/nEMO → na tych zbiorach to wyciek; model **drugorzędny** (nie nagłówek). speechbrain (IEMOCAP) to „czysty" EN. Oznaczone w spec §4/§7.
- Wagi `.pt` czytane z `old_projects/.../notebooks/models/` (read-only, niewersjonowane) — tylko odczyt.
- Kod/testy/commity po angielsku; **żadnego `Co-Authored-By`**.

## Review mode (per memory: economical-subagent-execution)

Modele mowy karmią wyniki → **pełna recenzja per-task (T1–T3)**. Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus.

## Recovered architecture & pipeline (from old_projects notebooks, verified 2026-07-23)

- `librosa.load(offset=0.5, duration=3, sr=48000)` → zero-pad do 144000 próbek.
- `getMELspectrogram`: `librosa.feature.melspectrogram(n_fft=1024, win_length=512, window='hamming', hop_length=256, n_mels=128, fmax=sr/2)` → `power_to_db(ref=np.max)` → **(128, 563)**.
- `StandardScaler`: fit na spłaszczonych `(n, 1*128*563)` cechach train, transform val/test (tu: fit per-korpus).
- `ParallelModel` (dokładny kod w T2). `forward` → `(logits, softmax)`. Wejście `(b,1,128,563)`.
- Etykiety wg indeksu wyjścia:
  - RAVDESS-8: `["surprise","neutral","calm","happy","sad","angry","fear","disgust"]` (calm→neutral). Wagi: `cnn_transf_parallel_model_v1.pt`.
  - nEMO-6: `["anger","fear","happiness","sadness","surprised","neutral"]`. Wagi: `cnn_transf_parallel_model_nemo_v1.pt`.

---

## File Structure

```
app/
  config/settings.py         # += speech_weights_dir property  (T1)
  models/speech/__init__.py
  models/speech/features.py   # to_model_signal(), mel_spectrogram()  (T1)
  models/speech/architectures.py  # ParallelModel (copied verbatim)  (T2)
  models/speech/cnn_transformer.py # CNNTransformerSpeechModel + per-corpus scaler  (T2)
  models/speech/speechbrain_iemocap.py # SpeechBrainIEMOCAP  (T3)
  models/registry.py          # += speech_cnn_tf_ravdess, speech_cnn_tf_nemo, en_speech_iemocap
tests/
  test_speech_features.py
  test_cnn_transformer.py
  test_speechbrain_iemocap.py
requirements.txt              # += librosa, scikit-learn, speechbrain
```

Interfejsy:
- `features.to_model_signal(audio: np.ndarray, native_sr: int) -> np.ndarray` (float32, len 144000; resample→48000, offset0.5/dur3, pad).
- `features.mel_spectrogram(signal: np.ndarray, sr=48000) -> np.ndarray` (float32, (128,563)).
- `CNNTransformerSpeechModel(model_id, weights_path, native_labels)` — `modality="speech"`, per-corpus scaler in `predict_batch`.
- `SpeechBrainIEMOCAP(model_id, cache_dir=None)` — `modality="speech"`, 4-class.
- Registry ids: `speech_cnn_tf_ravdess`, `speech_cnn_tf_nemo`, `en_speech_iemocap`.

---

### Task 1: Audio feature pipeline (resample + log-mel)

**Files:**
- Modify: `requirements.txt` (add `librosa`, `scikit-learn`), `app/config/settings.py`
- Create: `app/models/speech/__init__.py`, `app/models/speech/features.py`
- Test: `tests/test_speech_features.py`

**Interfaces:** `to_model_signal`, `mel_spectrogram` (see above); `Settings.speech_weights_dir -> repo_root/"old_projects"/"polish-speech-emotion-recognition"/"notebooks"/"models"`.

- [ ] **Step 1: Add deps + settings property**

Append to `requirements.txt`:
```
librosa>=0.10,<1
scikit-learn>=1.4,<2
```
Run: `.venv/bin/pip install -r requirements.txt` then `.venv/bin/python -c "import librosa, sklearn; print(librosa.__version__, sklearn.__version__)"`.
In `app/config/settings.py` add:
```python
    @property
    def speech_weights_dir(self) -> Path:
        return _REPO_ROOT / "old_projects" / "polish-speech-emotion-recognition" / "notebooks" / "models"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_speech_features.py`:
```python
import numpy as np
from app.models.speech.features import to_model_signal, mel_spectrogram, SAMPLE_RATE


def test_to_model_signal_length_and_resample():
    # 5s of 24kHz audio -> resampled to 48kHz, offset 0.5s + 3s window, padded to 144000
    native_sr = 24000
    audio = np.sin(2 * np.pi * 220 * np.arange(5 * native_sr) / native_sr).astype("float32")
    sig = to_model_signal(audio, native_sr)
    assert sig.dtype == np.float32
    assert sig.shape == (SAMPLE_RATE * 3,)   # 144000


def test_to_model_signal_pads_short_audio():
    native_sr = 48000
    audio = np.ones(int(0.7 * native_sr), dtype="float32")  # only 0.7s; after 0.5s offset -> 0.2s left
    sig = to_model_signal(audio, native_sr)
    assert sig.shape == (144000,)
    assert np.count_nonzero(sig) <= int(0.2 * native_sr) + 1  # mostly zero-padded


def test_mel_spectrogram_shape():
    sig = np.random.randn(144000).astype("float32")
    mel = mel_spectrogram(sig)
    assert mel.shape == (128, 563)
    assert mel.dtype == np.float32
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_speech_features.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 4: Write the implementation**

Create `app/models/speech/__init__.py`:
```python
"""Speech emotion models."""
```

Create `app/models/speech/features.py`:
```python
"""Audio feature pipeline for the custom CNN-Transformer speech model.

Faithfully reproduces the training pipeline: resample to 48 kHz, take a 3 s
window starting at 0.5 s (zero-padded), then a log-mel spectrogram (128 x 563).
"""
import librosa
import numpy as np

SAMPLE_RATE = 48000
OFFSET_S = 0.5
DURATION_S = 3
TARGET_LEN = SAMPLE_RATE * DURATION_S  # 144000


def to_model_signal(audio: np.ndarray, native_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")
    if native_sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SAMPLE_RATE)
    start = int(OFFSET_S * SAMPLE_RATE)
    window = audio[start:start + TARGET_LEN]
    signal = np.zeros(TARGET_LEN, dtype="float32")
    signal[:len(window)] = window
    return signal


def mel_spectrogram(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=1024, win_length=512, window="hamming",
        hop_length=256, n_mels=128, fmax=sr / 2,
    )
    return librosa.power_to_db(mel, ref=np.max).astype("float32")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_speech_features.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/config/settings.py app/models/speech/__init__.py app/models/speech/features.py tests/test_speech_features.py
git commit -m "feat: speech feature pipeline (48kHz resample + log-mel)"
```

---

### Task 2: Custom CNN-Transformer speech model

**Files:**
- Create: `app/models/speech/architectures.py`, `app/models/speech/cnn_transformer.py`
- Modify: `app/models/registry.py`
- Test: `tests/test_cnn_transformer.py`

**Interfaces:** `ParallelModel(num_emotions)` (verbatim from the notebook); `CNNTransformerSpeechModel(model_id, weights_path, native_labels)` with per-corpus StandardScaler in `predict_batch`. Registry: `speech_cnn_tf_ravdess` (8-class), `speech_cnn_tf_nemo` (6-class).

- [ ] **Step 1: Write the failing test**

Create `tests/test_cnn_transformer.py`:
```python
import glob
import numpy as np
import pytest
import torch
from pathlib import Path
from app.config.emotions import Emotion
from app.models.speech.architectures import ParallelModel


def test_parallel_model_output_shapes():
    m = ParallelModel(num_emotions=8).eval()
    x = torch.randn(2, 1, 128, 563)
    with torch.no_grad():
        logits, soft = m(x)
    assert logits.shape == (2, 8)
    assert soft.shape == (2, 8)
    assert torch.allclose(soft.sum(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(not glob.glob("old_projects/polish-speech-emotion-recognition/notebooks/models/cnn_transf_parallel_model_v1.pt"),
                    reason="custom speech weights not present")
def test_ravdess_weights_load_and_predict_on_real_audio():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("speech_cnn_tf_ravdess")
    assert m.modality == "speech"
    assert Emotion.NEUTRAL in m.mapper.supported  # calm+neutral both -> neutral
    # a real RAVDESS wav
    wav = sorted(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))[0]
    s = Sample(sample_id="RAVDESS:probe", studies=[1], corpus="RAVDESS", language="en",
               text="Kids are talking by the door", audio=AudioRef(wav_path=Path(wav)),
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])   # single-sample batch: scaler fit on this one (probe only)
    assert pred.pred in set(Emotion) and pred.modality == "speech"
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cnn_transformer.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write architectures.py (verbatim ParallelModel)**

Create `app/models/speech/architectures.py`:
```python
"""Speech model architectures, copied verbatim from the original notebooks
(old_projects/polish-speech-emotion-recognition) so the saved .pt state_dicts load."""
import torch
import torch.nn as nn


class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=4), nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=4), nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=4), nn.Dropout(p=0.3),
        )
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512,
                                                  dropout=0.4, activation="relu")
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        self.out_linear = nn.Linear(320, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax
```

- [ ] **Step 4: Write cnn_transformer.py (wrapper)**

Create `app/models/speech/cnn_transformer.py`:
```python
"""Custom CNN-Transformer speech emotion model (own model; leaky on RAVDESS/nEMO).

StandardScaler is refit per-corpus (the original training scaler was not persisted
and the train split was unseeded, so it is unrecoverable) — an explicit approximation.
Therefore predict_batch must be called with the whole corpus in one batch.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from app.data.audio import load_audio
from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction
from app.models.speech.architectures import ParallelModel
from app.models.speech.features import mel_spectrogram, to_model_signal


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class CNNTransformerSpeechModel:
    modality = "speech"

    def __init__(self, model_id: str, weights_path: Path, native_labels: list[str]):
        self.model_id = model_id
        self.weights_path = Path(weights_path)
        self.native_labels = native_labels
        self.device = _device()
        self.model = ParallelModel(len(native_labels))
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.mapper = LabelMapper(native_labels)

    def config_signature(self) -> str:
        return f"cnn_tf:{self.weights_path.name}@labels={','.join(self.native_labels)}@scaler=per-corpus"

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        mels = []
        for s in samples:
            audio, sr = load_audio(s)
            mels.append(mel_spectrogram(to_model_signal(audio, sr)))
        X = np.stack(mels)[:, None, :, :]                      # (n,1,128,563)
        b, c, h, w = X.shape
        Xf = StandardScaler().fit_transform(X.reshape(b, -1))  # per-corpus refit
        X = Xf.reshape(b, c, h, w).astype("float32")
        with torch.no_grad():
            _, soft = self.model(torch.from_numpy(X).to(self.device))
        probs = soft.cpu().numpy()
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i].tolist(), self.mapper)
            for i, s in enumerate(samples)
        ]
```

- [ ] **Step 5: Register the models**

In `app/models/registry.py`, add import + constants + factory entries:
```python
from app.models.speech.cnn_transformer import CNNTransformerSpeechModel

RAVDESS8_LABELS = ["surprise", "neutral", "calm", "happy", "sad", "angry", "fear", "disgust"]
NEMO6_LABELS = ["anger", "fear", "happiness", "sadness", "surprised", "neutral"]
```
```python
    "speech_cnn_tf_ravdess": lambda: CNNTransformerSpeechModel(
        "speech_cnn_tf_ravdess",
        _settings.speech_weights_dir / "cnn_transf_parallel_model_v1.pt", RAVDESS8_LABELS),
    "speech_cnn_tf_nemo": lambda: CNNTransformerSpeechModel(
        "speech_cnn_tf_nemo",
        _settings.speech_weights_dir / "cnn_transf_parallel_model_nemo_v1.pt", NEMO6_LABELS),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cnn_transformer.py -v`
Expected: PASS (the shape test immediately; the slow test loads real weights + a real wav).
If the `.pt` fails to load with a class/shape mismatch, STOP and report — the architecture must match the saved state_dict exactly.

- [ ] **Step 7: Commit**

```bash
git add app/models/speech/architectures.py app/models/speech/cnn_transformer.py app/models/registry.py tests/test_cnn_transformer.py
git commit -m "feat: custom CNN-Transformer speech model with per-corpus scaler"
```

---

### Task 3: speechbrain wav2vec2 IEMOCAP model

**Files:**
- Modify: `requirements.txt` (add `speechbrain`)
- Create: `app/models/speech/speechbrain_iemocap.py`
- Modify: `app/models/registry.py` (register `en_speech_iemocap`)
- Test: `tests/test_speechbrain_iemocap.py`

**Interfaces:** `SpeechBrainIEMOCAP(model_id, cache_dir=None)` — loads `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`; `predict_batch` runs each sample's audio (resampled to 16 kHz) through the classifier, reads the per-class probabilities and the label order from the model's label encoder, maps the IEMOCAP labels (ang/hap/sad/neu) to canonical via a `{"ang":"anger","hap":"happiness","sad":"sadness","neu":"neutral"}` id2label + `LabelMapper`, and builds a `Prediction` per sample.

- [ ] **Step 1: Add speechbrain and install**

Append to `requirements.txt`:
```
speechbrain>=1.0,<2
```
Run: `.venv/bin/pip install -r requirements.txt` then `.venv/bin/python -c "import speechbrain; print(speechbrain.__version__)"`.
If installing speechbrain forces an incompatible torch/torchaudio downgrade, STOP and report the version conflict before proceeding (do not silently downgrade torch).

- [ ] **Step 2: Write the failing test**

Create `tests/test_speechbrain_iemocap.py`:
```python
import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.models.speech.speechbrain_iemocap import IEMOCAP_TO_CANONICAL


def test_iemocap_label_map_covers_four_classes():
    assert set(IEMOCAP_TO_CANONICAL.values()) == {"anger", "happiness", "sadness", "neutral"}


@pytest.mark.slow
@pytest.mark.skipif(not glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"),
                    reason="RAVDESS dataset not present")
def test_speechbrain_predicts_on_real_audio():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("en_speech_iemocap")
    assert m.modality == "speech"
    # IEMOCAP has only 4 classes -> no fear/disgust/surprise
    assert m.mapper.supported == {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}
    wav = sorted(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))[0]
    s = Sample(sample_id="RAVDESS:probe", studies=[1], corpus="RAVDESS", language="en",
               text="Kids are talking by the door", audio=AudioRef(wav_path=Path(wav)),
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])
    assert pred.pred in {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_speechbrain_iemocap.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 4: Write the implementation**

Create `app/models/speech/speechbrain_iemocap.py`:
```python
"""speechbrain wav2vec2 IEMOCAP speech-emotion model (clean EN reference; 4 classes)."""
from pathlib import Path

import numpy as np
import torch

from app.data.audio import load_audio
from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction

# IEMOCAP label abbreviations -> canonical raw words (mapped further by map_label).
IEMOCAP_TO_CANONICAL = {"ang": "anger", "hap": "happiness", "sad": "sadness", "neu": "neutral"}
_TARGET_SR = 16000


class SpeechBrainIEMOCAP:
    modality = "speech"

    def __init__(self, model_id: str, cache_dir: Path | None = None):
        from speechbrain.inference.interfaces import foreign_class
        self.model_id = model_id
        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=str(cache_dir) if cache_dir else None,
        )
        # Label order from the model's encoder, mapped to canonical raw words.
        enc = self.classifier.hparams.label_encoder
        self.native_labels = [
            IEMOCAP_TO_CANONICAL[enc.decode_ndim(i)] for i in range(len(IEMOCAP_TO_CANONICAL))
        ]
        self.mapper = LabelMapper(self.native_labels)

    def config_signature(self) -> str:
        return "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        out = []
        for s in samples:
            audio, sr = load_audio(s)
            wav = torch.from_numpy(np.asarray(audio, dtype="float32"))
            if sr != _TARGET_SR:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, _TARGET_SR)
            out_prob, _, _, _ = self.classifier.classify_batch(wav.unsqueeze(0))
            probs = out_prob.squeeze(0).exp().cpu().numpy()  # log-probs -> probs
            probs = (probs / probs.sum()).astype("float64")
            out.append(to_prediction(s.sample_id, self.model_id, self.modality,
                                     self.native_labels, probs.tolist(), self.mapper))
        return out
```

- [ ] **Step 5: Register the model**

In `app/models/registry.py`:
```python
from app.models.speech.speechbrain_iemocap import SpeechBrainIEMOCAP
```
```python
    "en_speech_iemocap": lambda: SpeechBrainIEMOCAP("en_speech_iemocap", cache_dir=_settings.hf_home),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speechbrain_iemocap.py -v`
Expected: PASS (fast label-map test; the slow test downloads the speechbrain model, resamples a real wav to 16 kHz, and predicts).
If the label order from `label_encoder` does not decode to the four expected abbreviations, STOP and report the actual labels — the mapping must be verified against the real encoder, not assumed.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt app/models/speech/speechbrain_iemocap.py app/models/registry.py tests/test_speechbrain_iemocap.py
git commit -m "feat: speechbrain wav2vec2 IEMOCAP speech model"
```

---

## Self-Review

**1. Spec coverage (P3d scope):**
- Custom `ParallelModel` reimplemented verbatim + wrapper with faithful feature pipeline (48 kHz, log-mel 128×563) → T1/T2. ✅
- StandardScaler refit per-corpus (documented approximation; whole-corpus batch) → T2 + Global Constraints. ✅
- RAVDESS-8 / nEMO-6 label sets (calm→neutral) via LabelMapper → T2. ✅
- speechbrain IEMOCAP clean EN model, 4 classes, label order from the real encoder → T3. ✅
- Leakage documented (custom model secondary/leaky; speechbrain clean) → Global Constraints + spec §4/§7. ✅
- Out of scope (later): Qwen zero-shot backend + benchmark (P3b); LSTM wrapper (P3e, after training); analysis/report (P4).

**2. Placeholder scan:** No TBD/TODO; every code + test step is complete. The `slow` tests load real weights / download speechbrain / read real audio — intentional verification. Two explicit STOP-and-report guards (state_dict mismatch; speechbrain torch conflict; encoder label mismatch) protect correctness.

**3. Type consistency:** `EmotionModel` contract (`model_id`/`modality`/`config_signature`/`predict_batch`) satisfied by both wrappers; `load_audio` (P2), `to_prediction`/`LabelMapper` (P1), `Settings` used per signatures. `ParallelModel(num_emotions)` matches the saved state_dict's construction.

## Otwarte punkty / uwagi
- **Per-corpus scaler ⇒ full-corpus batch:** przy realnym uruchomieniu `predict speech_cnn_tf_*` używać `--batch-size` ≥ rozmiar korpusu (inaczej scaler dopasowany na fragmencie). Testy używają 1-próbkowej partii tylko do sprawdzenia ścieżki (nie jakości).
- **speechbrain vs torch 2.13** — możliwy konflikt torchaudio; T3 ma STOP-and-report zamiast cichego downgrade.
- **Realna dokładność** autorskiego modelu będzie raportowana przez nas (oryginalne 94–96% z README to walidacja/wyciek; realny held-out ~61%).
- **P3b** (Qwen), **P3e** (LSTM wrapper) — osobno.
