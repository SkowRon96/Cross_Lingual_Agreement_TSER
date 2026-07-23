# P3b — Qwen2-Audio zero-shot (offline on Windows+GPU) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Uzyskać predykcje zero-shot mowy z Qwen2-Audio-7B dla wszystkich korpusów (Badanie 2: 8 języków; Badanie 1: PL/EN mowa) — uruchamiane **offline na maszynie Windows+RTX 3090 (fp16)** — i zaimportować je do `PredictionStore` jako model `speech_zeroshot`, spójnie z resztą pipeline'u.

**Architecture:** Ponieważ model nie zmieści się na M1, inferencja jest **offline** na Windowsie: samodzielny skrypt (`run_qwen.py`, tylko torch/transformers/soundfile/pyarrow) czyta audio z przeniesionych parquetów CAMEO + wav RAVDESS, promptuje Qwen2-Audio o **7 klas kanonicznych**, parsuje odpowiedź, zapisuje przenośny parquet predykcji (klucz = `sample_id` identyczny z loaderami P2). Na Macu: prompt+parser (zweryfikowane, TDD) są źródłem prawdy (skrypt Windows kopiuje tę samą logikę), a CLI `import-predictions` wciąga plik z Windows do `PredictionStore`.

**Tech Stack:** Mac (główny `.venv`): Pydantic, pyarrow, pytest. Windows (osobne środowisko): Python 3.11, torch cu124, transformers, accelerate, soundfile, pyarrow, pandas.

## Global Constraints

- **Model:** `Qwen/Qwen2-Audio-7B-Instruct`, **fp16 na CUDA** (RTX 3090 24 GB — bez kwantyzacji). Determinizm: `do_sample=False` (greedy).
- **7 klas kanonicznych** w prompcie: `anger, disgust, fear, happiness, sadness, surprise, neutral`. Parser mapuje wolny tekst → jedna kanoniczna etykieta lub **abstain** (fail-loud na pustce, nie zgadywanie).
- **`sample_id` musi być identyczny** z loaderami P2 (CAMEO: `{split}:{file_id}`; RAVDESS: `RAVDESS:{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}`), inaczej import nie połączy się z gold.
- **Wznawialność (Windows):** predykcje zapisywane przyrostowo; przy restarcie pomijać już policzone `sample_id`.
- Confidence zero-shot jest niewiarygodny (spec §2) → importujemy **twardą etykietę** (`scores={pred:1.0}`, `confidence=None`); korelacje confidence w Badaniu 2 pomijają ten model.
- Skrypt Windows jest **samodzielny** (NIE importuje `app`); prompt+parser są skopiowane z zweryfikowanej wersji `app` (identyczne — test na Macu pilnuje wersji `app`).
- Kod/testy/commity po angielsku; **żadnego `Co-Authored-By`**. `speech_zeroshot` to model **główny/czysty** dla mowy (nagłówek).

## Review mode (per memory: economical-subagent-execution)

Prompt/parser + import karmią wyniki → **pełna recenzja T1, T2, T3**. Implementerzy: haiku. Recenzenci: sonnet. Finalny whole-branch review: opus. (T3 = skrypt Windows: nie uruchamialny na Macu — recenzja sprawdza logikę, zgodność promptu/parsera z T1, konstrukcję sample_id i schemat wyjścia.)

## Coordination / data transfer (dla użytkownika)

Przenieść na Windows: `datasets/CAMEO/data/*.parquet` (~5,3 GB, audio osadzone) + `datasets/RAVDESS/Audio_Speech_Actors_01-24/` (wav). Model (~16 GB) pobierze się na Windowsie z HF. Wynik (`qwen_predictions.parquet`) przenieść z powrotem do Maca → `import-predictions`.

---

## File Structure

```
app/
  models/speech/zeroshot/__init__.py
  models/speech/zeroshot/prompt.py     # build_prompt(); parse_response()  (T1, Mac, source of truth)
  data/import_preds.py                 # import_predictions() -> PredictionStore  (T2, Mac)
  cli.py                               # += `import-predictions` subcommand  (T2)
scripts/qwen_windows/
  run_qwen.py                          # STANDALONE Windows inference (T3, delivered, not run on Mac)
  README.md                            # exact Windows commands (T3)
tests/
  test_zeroshot_prompt.py
  test_import_preds.py
```

Interfejsy:
- `app.models.speech.zeroshot.prompt.build_prompt() -> str`; `parse_response(raw: str) -> str | None` (canonical emotion word or None).
- `app.data.import_preds.import_predictions(preds_path, store, study, corpus, model_id="speech_zeroshot") -> int` (returns count).
- Windows `run_qwen.py`: reads CAMEO parquets / RAVDESS wavs → writes `qwen_predictions.parquet` with columns `[sample_id, corpus, raw_response, pred]` (pred = canonical word or "" for abstain).

---

### Task 1: Prompt + response parser (Mac, source of truth)

**Files:**
- Create: `app/models/speech/zeroshot/__init__.py`, `app/models/speech/zeroshot/prompt.py`
- Test: `tests/test_zeroshot_prompt.py`

**Interfaces:**
- `CANONICAL_WORDS = ["anger","disgust","fear","happiness","sadness","surprise","neutral"]`.
- `build_prompt() -> str` — a one-line instruction telling the model to classify the speaker's emotion into exactly one of the 7 words and answer with only that word.
- `parse_response(raw: str) -> str | None` — normalize (lowercase), then find which canonical emotion the answer denotes, tolerant of synonyms (`joy/joyful/happy→happiness`, `angry→anger`, `sad→sadness`, `fearful/afraid→fear`, `surprised→surprise`, `disgusted→disgust`, `calm/none→neutral`) and of extra words ("The emotion is happiness."). Return the canonical word, or `None` if nothing matches (abstain). Deterministic; first canonical match by scanning the known synonym set.

- [ ] **Step 1: Write the failing test**

Create `tests/test_zeroshot_prompt.py`:
```python
from app.models.speech.zeroshot.prompt import build_prompt, parse_response, CANONICAL_WORDS


def test_build_prompt_lists_all_seven_words():
    p = build_prompt().lower()
    for w in CANONICAL_WORDS:
        assert w in p


def test_parse_exact_words():
    assert parse_response("anger") == "anger"
    assert parse_response("Neutral") == "neutral"


def test_parse_from_sentence_and_synonyms():
    assert parse_response("The emotion is happiness.") == "happiness"
    assert parse_response("joyful") == "happiness"
    assert parse_response("The speaker sounds angry") == "anger"
    assert parse_response("surprised") == "surprise"
    assert parse_response("afraid") == "fear"
    assert parse_response("calm") == "neutral"


def test_parse_no_match_returns_none():
    assert parse_response("I cannot tell") is None
    assert parse_response("") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_zeroshot_prompt.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `app/models/speech/zeroshot/__init__.py`:
```python
"""Zero-shot audio-LLM speech emotion (Qwen2-Audio, run offline on GPU)."""
```

Create `app/models/speech/zeroshot/prompt.py`:
```python
"""Canonical 7-class prompt + robust response parser for the Qwen2-Audio zero-shot model.

This module is the SOURCE OF TRUTH for the prompt and the answer parser. The standalone
Windows inference script (scripts/qwen_windows/run_qwen.py) copies this logic verbatim;
tests here guard the canonical version.
"""
CANONICAL_WORDS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# Synonym / surface-form -> canonical word. Ordered longest-first at match time.
_SYNONYMS = {
    "anger": "anger", "angry": "anger", "annoyed": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear", "afraid": "fear", "scared": "fear",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness", "joyful": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "calm": "neutral", "none": "neutral",
}


def build_prompt() -> str:
    words = ", ".join(CANONICAL_WORDS)
    return ("Listen to the speech and classify the speaker's emotion into exactly one of "
            f"these categories: {words}. Answer with only one word from that list.")


def parse_response(raw: str) -> str | None:
    if not raw:
        return None
    text = raw.strip().lower()
    # scan tokens; longest synonym keys first so 'surprised' beats a substring match
    for key in sorted(_SYNONYMS, key=len, reverse=True):
        # word-ish containment: the key appears as a standalone token or substring of a word
        if key in text:
            return _SYNONYMS[key]
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_zeroshot_prompt.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add app/models/speech/zeroshot/__init__.py app/models/speech/zeroshot/prompt.py tests/test_zeroshot_prompt.py
git commit -m "feat: Qwen2-Audio zero-shot canonical prompt + response parser"
```

---

### Task 2: import-predictions CLI (Mac)

**Files:**
- Create: `app/data/import_preds.py`
- Modify: `app/cli.py` (add `import-predictions` subcommand)
- Test: `tests/test_import_preds.py`

**Interfaces:**
- `import_predictions(preds_path: Path, store: PredictionStore, study: int, corpus: str, model_id: str = "speech_zeroshot", config_hash: str = "qwen2-audio-7b-instruct@zeroshot") -> int` — reads a parquet with columns `[sample_id, corpus, raw_response, pred]`, builds a `Prediction` per row (pred non-empty → `pred=Emotion(pred)`, `scores={that:1.0}`, `confidence=None`, `abstained=False`; empty pred → abstained), filters to the requested `corpus`, writes to the store via `write(...)`, returns the count. Unknown `pred` value (not a canonical Emotion) raises (fail-loud).
- CLI `import-predictions --file <parquet> --study N --corpus X [--results-root R]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_import_preds.py`:
```python
from pathlib import Path
import pandas as pd
from app.config.emotions import Emotion
from app.predict.store import PredictionStore
from app.data.import_preds import import_predictions


def test_import_builds_predictions(tmp_path):
    df = pd.DataFrame([
        {"sample_id": "nemo:a", "corpus": "nemo", "raw_response": "happiness", "pred": "happiness"},
        {"sample_id": "nemo:b", "corpus": "nemo", "raw_response": "I cannot tell", "pred": ""},
        {"sample_id": "cafe:c", "corpus": "cafe", "raw_response": "anger", "pred": "anger"},
    ])
    pf = tmp_path / "qwen_predictions.parquet"
    df.to_parquet(pf, index=False)
    store = PredictionStore(tmp_path / "store")
    n = import_predictions(pf, store, study=2, corpus="nemo")   # only nemo rows
    assert n == 2
    preds = {p.sample_id: p for p in store.read(2, "nemo", "speech_zeroshot")}
    assert preds["nemo:a"].pred is Emotion.HAPPINESS
    assert preds["nemo:a"].scores[Emotion.HAPPINESS] == 1.0
    assert preds["nemo:a"].confidence is None
    assert preds["nemo:b"].abstained is True and preds["nemo:b"].pred is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_import_preds.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `app/data/import_preds.py`:
```python
"""Import externally-computed predictions (e.g. Qwen2-Audio from the Windows GPU box)."""
from pathlib import Path

import pandas as pd

from app.config.emotions import Emotion
from app.models.base import Prediction
from app.predict.store import PredictionStore


def import_predictions(preds_path: Path, store: PredictionStore, study: int, corpus: str,
                       model_id: str = "speech_zeroshot",
                       config_hash: str = "qwen2-audio-7b-instruct@zeroshot") -> int:
    df = pd.read_parquet(preds_path)
    rows = [r for r in df.to_dict(orient="records") if r["corpus"] == corpus]
    predictions = []
    for r in rows:
        pred_str = (r.get("pred") or "").strip()
        if pred_str:
            emo = Emotion(pred_str)  # raises ValueError on a non-canonical value (fail loud)
            predictions.append(Prediction(
                sample_id=r["sample_id"], model_id=model_id, modality="speech",
                pred_raw=str(r.get("raw_response") or ""), pred=emo, confidence=None,
                scores={emo: 1.0}, abstained=False, abstain_reason=None))
        else:
            predictions.append(Prediction(
                sample_id=r["sample_id"], model_id=model_id, modality="speech",
                pred_raw=str(r.get("raw_response") or ""), pred=None, confidence=None,
                scores={}, abstained=True, abstain_reason="zero-shot response unparseable"))
    sig = store.signature(model_id, config_hash, [p.sample_id for p in predictions])
    store.write(study, corpus, model_id, predictions, sig)
    return len(predictions)
```

Add to `app/cli.py` a subcommand (inside `main`'s subparsers, mirroring `predict`):
```python
    ip = sub.add_parser("import-predictions", help="Import external predictions (e.g. Qwen from Windows) into the store")
    ip.add_argument("--file", required=True)
    ip.add_argument("--study", type=int, required=True)
    ip.add_argument("--corpus", required=True)
    ip.add_argument("--results-root", default=None)
    ip.set_defaults(func=_import_predictions)
```
and the handler:
```python
def _import_predictions(args) -> int:
    from app.data.import_preds import import_predictions
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    store = PredictionStore(results_root / "predictions")
    n = import_predictions(Path(args.file), store, args.study, args.corpus)
    print(f"imported {n} speech_zeroshot predictions for {args.corpus} (study {args.study})")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_import_preds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/data/import_preds.py app/cli.py tests/test_import_preds.py
git commit -m "feat: import-predictions CLI for external (Qwen) speech predictions"
```

---

### Task 3: Standalone Windows inference bundle (delivered, not run on Mac)

**Files:**
- Create: `scripts/qwen_windows/run_qwen.py`, `scripts/qwen_windows/README.md`
- Test: none runnable on Mac — reviewer checks logic/consistency.

**Interfaces:** `run_qwen.py` reads CAMEO parquets and/or RAVDESS wavs (paths as CLI args), runs Qwen2-Audio-7B-Instruct (fp16, CUDA, greedy), classifies each clip with the SAME prompt+parser as `app/models/speech/zeroshot/prompt.py` (copied verbatim), and writes/append `qwen_predictions.parquet` with `[sample_id, corpus, raw_response, pred]`. Resumable (skips sample_ids already in the output). `sample_id` construction matches the P2 loaders exactly.

- [ ] **Step 1: Write `scripts/qwen_windows/run_qwen.py`**

Create it (standalone — no `app` imports; prompt/parser copied verbatim from `app/models/speech/zeroshot/prompt.py`):
```python
"""Standalone Qwen2-Audio zero-shot emotion inference for the Windows + NVIDIA box.

Run on the GPU machine (NOT on the Mac). Produces qwen_predictions.parquet that is copied
back to the repo and imported via `python -m app.cli import-predictions`.

Usage (from the transferred data root):
  python run_qwen.py --cameo-dir CAMEO/data --ravdess-audio-dir RAVDESS/Audio_Speech_Actors_01-24 \\
                     --out qwen_predictions.parquet

Resumable: re-run to continue; already-predicted sample_ids are skipped.
"""
import argparse
import glob
import io
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ---- prompt + parser: COPIED VERBATIM from app/models/speech/zeroshot/prompt.py ----
CANONICAL_WORDS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
_SYNONYMS = {
    "anger": "anger", "angry": "anger", "annoyed": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear", "afraid": "fear", "scared": "fear",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness", "joyful": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "calm": "neutral", "none": "neutral",
}
def build_prompt():
    return ("Listen to the speech and classify the speaker's emotion into exactly one of "
            f"these categories: {', '.join(CANONICAL_WORDS)}. Answer with only one word from that list.")
def parse_response(raw):
    if not raw:
        return ""
    text = raw.strip().lower()
    for key in sorted(_SYNONYMS, key=len, reverse=True):
        if key in text:
            return _SYNONYMS[key]
    return ""
# --------------------------------------------------------------------------------

RAVDESS_EMOTIONS = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
TARGET_SR = 16000


def corpus_from_cameo_path(path):
    return re.sub(r"-\d+-of-\d+\.parquet$", "", os.path.basename(path))


def iter_cameo(cameo_dir):
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(os.path.join(cameo_dir, "*.parquet"))):
        corpus = corpus_from_cameo_path(path)
        cols = pq.read_table(path, columns=["file_id", "audio"]).to_pydict()
        for i in range(len(cols["file_id"])):
            sid = f"{corpus}:{cols['file_id'][i]}"
            audio_bytes = cols["audio"][i]["bytes"]
            yield sid, corpus, audio_bytes, None


def iter_ravdess(audio_dir):
    for wav in sorted(glob.glob(os.path.join(audio_dir, "Actor_*", "03-*.wav"))):
        c = os.path.basename(wav).split(".")[0].split("-")
        key = f"{c[1]}-{c[2]}-{c[3]}-{c[4]}-{c[5]}-{c[6]}"
        yield f"RAVDESS:{key}", "RAVDESS", None, wav


def load_audio(audio_bytes, wav_path):
    if wav_path is not None:
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        # lightweight linear resample to 16 kHz (processor expects 16 kHz)
        import math
        n = int(round(len(data) * TARGET_SR / sr))
        data = np.interp(np.linspace(0, len(data), n, endpoint=False), np.arange(len(data)), data).astype("float32")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameo-dir", default=None)
    ap.add_argument("--ravdess-audio-dir", default=None)
    ap.add_argument("--out", default="qwen_predictions.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_parquet(args.out, columns=["sample_id"])["sample_id"].tolist())
        print(f"resuming: {len(done)} already done")

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    prompt = build_prompt()

    sources = []
    if args.cameo_dir:
        sources.append(iter_cameo(args.cameo_dir))
    if args.ravdess_audio_dir:
        sources.append(iter_ravdess(args.ravdess_audio_dir))

    buffer, n = [], 0
    for source in sources:
        for sid, corpus, audio_bytes, wav_path in source:
            if sid in done:
                continue
            audio = load_audio(audio_bytes, wav_path)
            conversation = [{"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audios=[audio], sampling_rate=TARGET_SR,
                               return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            reply = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            buffer.append({"sample_id": sid, "corpus": corpus,
                           "raw_response": reply, "pred": parse_response(reply)})
            n += 1
            if len(buffer) >= 100:
                _flush(buffer, args.out)
                buffer = []
                print(f"\rpredicted {n}", end="")
    if buffer:
        _flush(buffer, args.out)
    print(f"\ndone: wrote/updated {args.out}")


def _flush(buffer, out):
    new = pd.DataFrame(buffer)
    if os.path.exists(out):
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True)
    new.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/qwen_windows/README.md`**

Create it with the exact Windows commands:
````markdown
# Qwen2-Audio zero-shot — run on the Windows + RTX 3090 box

Produces `qwen_predictions.parquet`, which you copy back to the Mac repo and import with
`python -m app.cli import-predictions --file qwen_predictions.parquet --study 2 --corpus <name>`.

## 1. Transfer these from the Mac repo to a folder on Windows (e.g. C:\qwen\data)
- `datasets/CAMEO/data/*.parquet`   -> `C:\qwen\data\CAMEO\data\`
- `datasets/RAVDESS/Audio_Speech_Actors_01-24\` -> `C:\qwen\data\RAVDESS\Audio_Speech_Actors_01-24\`
- `scripts/qwen_windows/run_qwen.py` -> `C:\qwen\run_qwen.py`

## 2. Environment (PowerShell)
```powershell
cd C:\qwen
python -m venv .venv-qwen
.\.venv-qwen\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate soundfile pyarrow pandas numpy
python -c "import torch; print('CUDA ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `CUDA ok: True NVIDIA GeForce RTX 3090`.

## 3. Run (downloads the ~16 GB model on first run; fp16 fits in 24 GB VRAM)
```powershell
python run_qwen.py --cameo-dir data\CAMEO\data --ravdess-audio-dir data\RAVDESS\Audio_Speech_Actors_01-24 --out qwen_predictions.parquet
```
- Resumable: if interrupted, re-run the same command — it skips already-done `sample_id`s.
- ~40k clips; expect several hours. Watch VRAM with `nvidia-smi` (should sit ~17 GB).

## 4. Copy `qwen_predictions.parquet` back to the Mac repo, then per corpus:
```bash
.venv/bin/python -m app.cli import-predictions --file qwen_predictions.parquet --study 2 --corpus nemo
.venv/bin/python -m app.cli import-predictions --file qwen_predictions.parquet --study 2 --corpus cafe
# ... one per CAMEO split; and for Study 1:
.venv/bin/python -m app.cli import-predictions --file qwen_predictions.parquet --study 1 --corpus RAVDESS
```
````

- [ ] **Step 3: Sanity-check the script compiles (Mac, no CUDA)**

Run: `.venv/bin/python -m py_compile scripts/qwen_windows/run_qwen.py && echo "syntax OK"`
Expected: `syntax OK` (imports of torch/transformers are inside `main`/at top — py_compile only checks syntax; do NOT try to run it on the Mac).
NOTE: torch/transformers ARE imported at module top here, so `py_compile` is fine (it doesn't import). Do not execute.

- [ ] **Step 4: Verify prompt/parser match the app source of truth**

Run:
```bash
.venv/bin/python - <<'PY'
import re
from app.models.speech.zeroshot import prompt as canon
src = open("scripts/qwen_windows/run_qwen.py").read()
# the canonical words + synonyms + prompt sentence must appear identically
assert canon.build_prompt() in src, "prompt drifted between app and windows script"
for k, v in canon._SYNONYMS.items():
    assert f'"{k}": "{v}"' in src, f"synonym {k} drifted"
print("prompt+parser in run_qwen.py match app source of truth")
PY
```
Expected: match confirmation.

- [ ] **Step 5: Commit**

```bash
git add scripts/qwen_windows/run_qwen.py scripts/qwen_windows/README.md
git commit -m "feat: standalone Windows Qwen2-Audio zero-shot inference bundle"
```

---

## Self-Review

**1. Spec coverage (P3b scope):**
- Zero-shot speech model for all 8 languages + PL/EN (Study 2 + Study 1) → offline Windows fp16 + import. ✅
- Canonical 7-class prompt + robust parser (source of truth, tested) → T1. ✅
- Import into PredictionStore as `speech_zeroshot` (hard label, confidence=None) → T2. ✅
- sample_id alignment with P2 loaders; resumable Windows run → T3. ✅
- Out of scope: analysis/report (P4). Confidence-correlation for zero-shot intentionally skipped (unreliable).

**2. Placeholder scan:** No TBD/TODO in the Mac tasks. T3 is a delivered script (can't run on Mac); Steps 3–4 syntax-check + a drift-guard that the prompt/parser match `app`.

**3. Type consistency:** `Prediction`/`Emotion`/`PredictionStore` (P1) used per signatures; `import_predictions` builds valid Predictions; `speech_zeroshot` config_hash fixed. run_qwen.py's `sample_id` matches `cameo.corpus_from_path`/`load_cameo_split` and `ravdess.load_ravdess` (`{split}:{file_id}`, `RAVDESS:{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}`).

## Otwarte punkty
- Czas na Windowsie (~kilkanaście h) — skrypt wznawialny; puszczać na raty.
- Qwen2-Audio jakość zero-shot: możliwy niższy pułap (spec §7) — akceptowalne, oznaczone.
- Weryfikacja `apply_chat_template`/`processor` API Qwen2-Audio na realnej wersji transformers na Windowsie — jeśli API się różni, dostosować (skrypt to jedyne miejsce; drift-guard pilnuje tylko promptu/parsera).
