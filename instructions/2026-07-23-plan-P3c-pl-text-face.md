# P3c — PL/multilingual text models + face ViT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dodać pozostałe modele tekstowe (poltextlab, tabularisai, roberta6) do rejestru — reużywając `HFTextModel` z P3a — oraz zbudować wrapper twarzy (trpakov ViT) działający na cropach z `extract_face_frames` (P2-RAVDESS).

**Architecture:** Modele tekstowe to (poza roberta6) zwykłe wpisy rejestru na generycznym `HFTextModel`; roberta6 wymaga jawnego `id2label_override` (config w repo jest generyczny). Model twarzy to nowy `ViTFaceModel` (pipeline obrazowy: klatki → ViT image-classification → uśrednienie softmax po klatkach → `to_prediction`), implementujący protokół `EmotionModel`.

**Tech Stack:** Python 3.12, PyTorch (CPU/MPS), Transformers, Pillow (przetwarzanie obrazu), Pydantic v2, pytest.

## Global Constraints

- Python **3.12**; jeden `.venv`; device-agnostic (mps/cpu, bez CUDA).
- Buduje na P1 (`LabelMapper`, `to_prediction`, `Emotion`, `map_label`), P2/P2-RAVDESS (`extract_face_frames`, `Settings.hf_home`), P3a (`HFTextModel`, `native_labels_from_id2label`, rejestr, runner).
- Etykiety mapowane **wyłącznie** przez `LabelMapper`/`map_label` (fail-loud). `native_labels` z `id2label` modelu — **z wyjątkiem roberta6**, gdzie config jest generyczny (`LABEL_0..5`) i podajemy jawny `id2label_override`.
- **roberta6 — ZAŁOŻENIE (niezweryfikowane):** `L0..5 = anger, fear, disgust, sadness, joy, none-of-them`. Repo nie publikuje `label2id`; próby odzysku (README-fingerprint, sonda nEMO) nie dały pewnego mapowania. roberta6 jest **drugorzędny** (nigdy w nagłówku κ), a założenie **jawnie oflagowane** w kodzie (docstring/komentarz) i w spec §4/§7. `config_signature` zawiera założone etykiety, więc zmiana założenia unieważnia cache.
- `none of them` → **neutral** dla roberta6 (brak osobnego neutral) — via `label_overrides`.
- Pobrania HF w `models/hf_cache/` (gitignore). Kod/testy/commity po angielsku; **żadnego `Co-Authored-By`**.

## Review mode (per memory: economical-subagent-execution)

Modele karmią wyniki → **pełna recenzja per-task dla T1 i T2**. Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus.

## Verified label sets (grounded 2026-07-23 via huggingface_hub)

- **poltextlab** (`xlmr_pooled10`) `id2label`: `{0:Neutral,1:Anger,2:Fear,3:Disgust,4:Sadness,5:Joy,6:Hope,7:Pride,8:Enthusiasm,9:None of them}` → `map_label`: Joy/Enthusiasm→happiness; Hope/Pride/None-of-them→DROP (ma osobny Neutral). supported = {anger,fear,disgust,sadness,happiness,neutral} (**brak surprise**).
- **tabularisai** (`mling_tabularis`) `id2label`: `{0:anger,1:contempt,2:disgust,3:fear,4:frustration,5:gratitude,6:joy,7:love,8:neutral,9:sadness,10:surprise}` → contempt/frustration/gratitude/love→DROP; reszta→kanoniczne. supported = wszystkie 7.
- **roberta6** (`pl_text_roberta6`) config generyczny → **override** `{0:anger,1:fear,2:disgust,3:sadness,4:joy,5:"none of them"}` + `label_overrides={"none of them": NEUTRAL}`. supported = {anger,fear,disgust,sadness,happiness,neutral} (**brak surprise**).
- **trpakov** (`en_face_vit`) `id2label`: `{0:angry,1:disgust,2:fear,3:happy,4:neutral,5:sad,6:surprise}` → angry→anger, happy→happiness, sad→sadness, reszta direct. supported = wszystkie 7.

---

## File Structure

```
app/
  models/
    text/hf_text.py          # += id2label_override + label_overrides params  (T1)
    registry.py              # += xlmr_pooled10, mling_tabularis, pl_text_roberta6 (+ROBERTA6_ID2LABEL), en_face_vit  (T1,T2)
    face/__init__.py
    face/vit_face.py         # ViTFaceModel + aggregate_face_scores  (T2)
tests/
  test_hf_text_overrides.py  # id2label_override + label_overrides + per-model supported sets
  test_registry_p3c.py       # new ids registered; roberta6 override applied
  test_vit_face.py           # aggregate helper + real RAVDESS-video integration
requirements.txt             # += pillow
```

Interfejsy:
- `HFTextModel(model_id, hf_repo, cache_dir=None, id2label_override: dict|None=None, label_overrides: dict|None=None)` — jeśli `id2label_override` podany, `native_labels` z niego (z walidacją długości vs `model.config.num_labels`); `label_overrides` przekazywane do `LabelMapper`. `config_signature` zawiera założone etykiety gdy override użyty.
- `app.models.face.vit_face.aggregate_face_scores(per_crop_probs: list[list[float]]) -> list[float]` (średnia po klatkach).
- `app.models.face.vit_face.ViTFaceModel(model_id, hf_repo, cache_dir=None, num_frames=8)` — `modality="face"`, `config_signature()` zawiera `num_frames`, `predict_batch` per sample: `extract_face_frames(video_path,num_frames)` → ViT → średnia softmax → `to_prediction`; brak twarzy → abstain.

---

### Task 1: Text models — id2label_override + register poltextlab/tabularisai/roberta6

**Files:**
- Modify: `app/models/text/hf_text.py`, `app/models/registry.py`
- Test: `tests/test_hf_text_overrides.py`, `tests/test_registry_p3c.py`

**Interfaces:** see Global.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_text_overrides.py`:
```python
from app.config.emotions import Emotion
from app.labels.mapper import LabelMapper
from app.models.text.hf_text import native_labels_from_id2label

# Verified id2label sets (grounded from HF configs).
POLTEXTLAB = {"0":"Neutral","1":"Anger","2":"Fear","3":"Disgust","4":"Sadness",
              "5":"Joy","6":"Hope","7":"Pride","8":"Enthusiasm","9":"None of them"}
TABULARIS = {"0":"anger","1":"contempt","2":"disgust","3":"fear","4":"frustration",
             "5":"gratitude","6":"joy","7":"love","8":"neutral","9":"sadness","10":"surprise"}
ROBERTA6 = {0:"anger",1:"fear",2:"disgust",3:"sadness",4:"joy",5:"none of them"}


def test_poltextlab_supported_set():
    m = LabelMapper(native_labels_from_id2label(POLTEXTLAB))
    assert m.supported == {Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST,
                           Emotion.SADNESS, Emotion.HAPPINESS, Emotion.NEUTRAL}  # no surprise


def test_tabularisai_supported_set_is_full_seven():
    m = LabelMapper(native_labels_from_id2label(TABULARIS))
    assert m.supported == set(Emotion)  # all 7


def test_roberta6_override_with_none_of_them_neutral():
    m = LabelMapper(native_labels_from_id2label(ROBERTA6),
                    overrides={"none of them": Emotion.NEUTRAL})
    assert m.supported == {Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST,
                           Emotion.SADNESS, Emotion.HAPPINESS, Emotion.NEUTRAL}  # no surprise
    assert m.map_label("none of them") is Emotion.NEUTRAL
```

Create `tests/test_registry_p3c.py`:
```python
import pytest
from app.models.registry import MODEL_IDS, build_model, ROBERTA6_ID2LABEL


def test_new_text_ids_registered():
    for mid in ["xlmr_pooled10", "mling_tabularis", "pl_text_roberta6"]:
        assert mid in MODEL_IDS


def test_roberta6_assumed_order_constant():
    # The documented ASSUMPTION (unverified) — guards against accidental reordering.
    assert ROBERTA6_ID2LABEL == {0:"anger",1:"fear",2:"disgust",3:"sadness",4:"joy",5:"none of them"}


@pytest.mark.slow
def test_build_roberta6_applies_override():
    m = build_model("pl_text_roberta6")
    assert m.native_labels == ["anger","fear","disgust","sadness","joy","none of them"]
    from app.config.emotions import Emotion
    assert Emotion.NEUTRAL in m.mapper.supported and Emotion.SURPRISE not in m.mapper.supported
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hf_text_overrides.py tests/test_registry_p3c.py -v`
Expected: FAIL (ImportError: ROBERTA6_ID2LABEL / native_labels helper mapping mismatch, or KeyError on unknown labels if map_label lacks something — it should not).

- [ ] **Step 3: Extend HFTextModel**

In `app/models/text/hf_text.py`, change `HFTextModel.__init__` signature and label derivation:
```python
    def __init__(self, model_id: str, hf_repo: str, cache_dir: Path | None = None,
                 id2label_override: dict | None = None,
                 label_overrides: dict | None = None):
        self.model_id = model_id
        self.hf_repo = hf_repo
        cd = str(cache_dir) if cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, cache_dir=cd)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_repo, cache_dir=cd)
        self.device = _device()
        self.model.to(self.device).eval()
        self._overridden = id2label_override is not None
        if id2label_override is not None:
            n = self.model.config.num_labels
            if len(id2label_override) != n:
                raise ValueError(
                    f"id2label_override has {len(id2label_override)} entries but model has {n} labels")
            self.native_labels = native_labels_from_id2label(id2label_override)
        else:
            self.native_labels = native_labels_from_id2label(self.model.config.id2label)
        self.label_overrides = label_overrides
        self.mapper = LabelMapper(self.native_labels, overrides=label_overrides)

    def config_signature(self) -> str:
        if self._overridden:
            return f"{self.hf_repo}@labels={','.join(self.native_labels)}"
        return self.hf_repo
```
(Leave `predict_batch` unchanged.)

- [ ] **Step 4: Register the models**

In `app/models/registry.py`, add the import and constant + factory entries:
```python
from app.config.emotions import Emotion
from app.models.text.hf_text import HFTextModel

# roberta6: repo config is generic (LABEL_0..5); this label order is an ASSUMPTION
# (unverified — no label2id published; README fingerprint / nEMO probe inconclusive).
# roberta6 is a SECONDARY model, never used in a headline result. See spec §4/§7.
ROBERTA6_ID2LABEL = {0: "anger", 1: "fear", 2: "disgust", 3: "sadness", 4: "joy", 5: "none of them"}
```
Add to `_FACTORIES`:
```python
    "xlmr_pooled10": lambda: HFTextModel(
        "xlmr_pooled10", "poltextlab/xlm-roberta-large-pooled-emotions10-v3",
        cache_dir=_settings.hf_home),
    "mling_tabularis": lambda: HFTextModel(
        "mling_tabularis", "tabularisai/multilingual-emotion-classification",
        cache_dir=_settings.hf_home),
    "pl_text_roberta6": lambda: HFTextModel(
        "pl_text_roberta6", "visegradmedia-emotion/Emotion_RoBERTa_polish6",
        cache_dir=_settings.hf_home,
        id2label_override=ROBERTA6_ID2LABEL,
        label_overrides={"none of them": Emotion.NEUTRAL}),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hf_text_overrides.py tests/test_registry_p3c.py -v`
Expected: PASS (fast tests immediately; the `slow` roberta6 build test downloads the model and asserts the override).

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add app/models/text/hf_text.py app/models/registry.py tests/test_hf_text_overrides.py tests/test_registry_p3c.py
git commit -m "feat: register poltextlab/tabularisai/roberta6 text models (roberta6 assumed label order)"
```

---

### Task 2: Face model — ViT wrapper on extracted frames

**Files:**
- Modify: `requirements.txt` (add `pillow`)
- Create: `app/models/face/__init__.py`, `app/models/face/vit_face.py`
- Modify: `app/models/registry.py` (register `en_face_vit`)
- Test: `tests/test_vit_face.py`

**Interfaces:** see Global. `ViTFaceModel.predict_batch` processes one sample at a time: extracts face crops from `sample.video_path`, runs the ViT image classifier on the crops (batched), averages the per-crop softmax, and builds a `Prediction` via `to_prediction`. A sample with no `video_path` or zero detected faces → an abstained `Prediction`.

- [ ] **Step 1: Add pillow and install**

Append to `requirements.txt`:
```
pillow>=10,<12
```
Run: `.venv/bin/pip install -r requirements.txt` then `.venv/bin/python -c "import PIL; print(PIL.__version__)"`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_vit_face.py`:
```python
import glob
import numpy as np
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.models.face.vit_face import aggregate_face_scores


def test_aggregate_face_scores_is_mean_over_crops():
    per_crop = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
    out = aggregate_face_scores(per_crop)
    assert out == pytest.approx([0.5, 0.5, 0.0])


@pytest.mark.slow
@pytest.mark.skipif(not glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"),
                    reason="RAVDESS dataset not present")
def test_vit_face_predicts_on_real_ravdess_video():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("en_face_vit")
    assert m.modality == "face"
    video = sorted(glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"))[0]
    s = Sample(sample_id="RAVDESS:probe", studies=[1], corpus="RAVDESS", language="en",
               text="Kids are talking by the door",
               audio=AudioRef(wav_path=Path("a.wav")), has_face=True, video_path=Path(video),
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])
    assert pred.modality == "face"
    assert pred.pred in set(Emotion)                      # a face was found -> a real prediction
    assert pred.abstained is False
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5


@pytest.mark.slow
def test_vit_face_abstains_without_video(tmp_path):
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("en_face_vit")
    s = Sample(sample_id="t:noface", studies=[1], corpus="t", language="en", text="x",
               audio=AudioRef(wav_path=tmp_path / "a.wav"), has_face=False, video_path=None,
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])
    assert pred.abstained is True and pred.pred is None and pred.scores == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vit_face.py -v`
Expected: FAIL (ModuleNotFoundError: app.models.face.vit_face).

- [ ] **Step 4: Write the implementation**

Create `app/models/face/__init__.py`:
```python
"""Facial-expression emotion models."""
```

Create `app/models/face/vit_face.py`:
```python
"""ViT facial-expression emotion model over extracted video face crops."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.data.schema import Sample
from app.data.video import extract_face_frames
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction
from app.models.text.hf_text import native_labels_from_id2label


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def aggregate_face_scores(per_crop_probs: list[list[float]]) -> list[float]:
    return np.asarray(per_crop_probs, dtype=np.float64).mean(axis=0).tolist()


class ViTFaceModel:
    modality = "face"

    def __init__(self, model_id: str, hf_repo: str, cache_dir: Path | None = None,
                 num_frames: int = 8):
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.num_frames = num_frames
        cd = str(cache_dir) if cache_dir else None
        self.processor = AutoImageProcessor.from_pretrained(hf_repo, cache_dir=cd)
        self.model = AutoModelForImageClassification.from_pretrained(hf_repo, cache_dir=cd)
        self.device = _device()
        self.model.to(self.device).eval()
        self.native_labels = native_labels_from_id2label(self.model.config.id2label)
        self.mapper = LabelMapper(self.native_labels)

    def config_signature(self) -> str:
        return f"{self.hf_repo}@nframes={self.num_frames}"

    def _abstain(self, sample_id: str) -> Prediction:
        return Prediction(sample_id=sample_id, model_id=self.model_id, modality=self.modality,
                          pred_raw=None, pred=None, confidence=None, scores={},
                          abstained=True, abstain_reason="no face detected in any sampled frame")

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        out: list[Prediction] = []
        for s in samples:
            crops = extract_face_frames(s.video_path, num_frames=self.num_frames) if s.video_path else []
            if not crops:
                out.append(self._abstain(s.sample_id))
                continue
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (n_crops, n_labels)
            mean_scores = aggregate_face_scores(probs.tolist())
            out.append(to_prediction(s.sample_id, self.model_id, self.modality,
                                     self.native_labels, mean_scores, self.mapper))
        return out
```

- [ ] **Step 5: Register the face model**

In `app/models/registry.py`, add the import and factory entry:
```python
from app.models.face.vit_face import ViTFaceModel
```
```python
    "en_face_vit": lambda: ViTFaceModel(
        "en_face_vit", "trpakov/vit-face-expression", cache_dir=_settings.hf_home),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vit_face.py -v`
Expected: PASS (fast aggregate test; slow tests download the ViT, extract faces from a real RAVDESS video, and predict / abstain).

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt app/models/face app/models/registry.py tests/test_vit_face.py
git commit -m "feat: ViT face-expression model over extracted video frames"
```

---

## Self-Review

**1. Spec coverage (P3c scope):**
- PL/multilingual text models registered (poltextlab, tabularisai) on the generic `HFTextModel`; label sets verified against grounded configs → T1. ✅
- roberta6 added with an explicit, documented, unverified `id2label_override` + none-of-them→neutral; secondary/never-headline; assumption folded into `config_signature` → T1. ✅
- Face model (trpakov ViT) over `extract_face_frames`, softmax averaged across crops, abstain on no-face → T2. ✅
- device-agnostic, deterministic, single-source label mapping, HF cache in models/ → T1/T2. ✅
- Out of scope (later): Qwen zero-shot backend + benchmark (P3b); custom speech CNN + speechbrain (P3d); custom LSTM (P3e).

**2. Placeholder scan:** No TBD/TODO; every code + test step is complete. The `slow`-marked tests download real models / read the real RAVDESS video — intentional real verification.

**3. Type consistency:** `HFTextModel` extended signature is backward-compatible (new params default None, so P3a's `en_text_hartmann` registration still works). `EmotionModel` contract (`model_id`, `modality`, `config_signature`, `predict_batch`) satisfied by `ViTFaceModel`. `native_labels_from_id2label`, `LabelMapper`, `to_prediction`, `extract_face_frames`, `Settings.hf_home` used per their signatures. Face model reuses `native_labels_from_id2label` from `hf_text` (public helper).

## Otwarte punkty / do kolejnych pod-planów
- **roberta6 label order** pozostaje ZAŁOŻENIEM — jeśli autor opublikuje `label2id`, podmienić `ROBERTA6_ID2LABEL` (config_signature unieważni cache automatycznie).
- **P3b:** Qwen2-Audio zero-shot + benchmark. **P3d:** autorski CNN-Transformer (resampling 48 kHz, StandardScaler) + speechbrain IEMOCAP. **P3e:** autorski LSTM.
- Agregacja twarzy: obecnie średnia softmax po `num_frames=8` klatkach; ewentualne strojenie liczby klatek/strategii — po pierwszych wynikach.
