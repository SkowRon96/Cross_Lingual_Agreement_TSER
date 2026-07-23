# Qwen2-Audio zero-shot — run on the Windows + RTX 3090 box

This produces `qwen_predictions.parquet`, which you copy back to the Mac repo and import
into the `PredictionStore` as the `speech_zeroshot` model.

The Windows box has **no Claude Code** — everything here is manual copy/paste. The script is
**standalone** (no `app` import): the prompt + response parser are copied verbatim from
`app/models/speech/zeroshot/prompt.py` (a Mac test guards that they stay identical).

---

## 1. Transfer these from the Mac repo to a folder on Windows (e.g. `C:\qwen`)

| From the Mac repo | To Windows |
|---|---|
| `datasets/CAMEO/data\*.parquet` (~5.3 GB, audio embedded) | `C:\qwen\data\CAMEO\data\` |
| `datasets/nEMO/data\*.parquet` (Study-1 Polish speech) | `C:\qwen\data\nEMO\data\` |
| `datasets/RAVDESS/Audio_Speech_Actors_01-24\` (wav files) | `C:\qwen\data\RAVDESS\Audio_Speech_Actors_01-24\` |
| `scripts/qwen_windows/run_qwen.py` | `C:\qwen\run_qwen.py` |
| `scripts/qwen_windows/sanity_check.py` | `C:\qwen\sanity_check.py` |

The model (~16 GB) downloads itself from Hugging Face on first run — no need to transfer it.

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

## 2.5 Sanity check FIRST (strongly recommended — ~2 min after the model downloads)

Runs the model on 3 clips per source and prints the raw + parsed answer. Writes nothing.
Do this before committing to the multi-hour run, and to confirm the Qwen2-Audio processor
API on your `transformers` version matches the script.

```powershell
python sanity_check.py --cameo-dir data\CAMEO\data --nemo-dir data\nEMO\data `
                       --ravdess-audio-dir data\RAVDESS\Audio_Speech_Actors_01-24 --n 3
```

Expect a small table of `sample_id | raw_response | parsed`. If `parsed` shows sensible
emotions (not all `(abstain)`), proceed. If the model call errors, the only thing to adjust
is the `apply_chat_template` / `processor(...)` block in `run_qwen.py`'s `main()` (and it is
imported by `sanity_check.py`, so fixing it once fixes both).

## 3. Run (downloads the ~16 GB model on first run; fp16 fits in 24 GB VRAM)

```powershell
python run_qwen.py `
  --cameo-dir data\CAMEO\data `
  --nemo-dir data\nEMO\data `
  --ravdess-audio-dir data\RAVDESS\Audio_Speech_Actors_01-24 `
  --out qwen_predictions.parquet
```

- **Resumable:** if interrupted (or to add a source later), re-run the same command — it
  skips `sample_id`s already present in `qwen_predictions.parquet`.
- ~40k clips total; expect several hours. Watch VRAM with `nvidia-smi` (should sit ~17 GB).
- Output columns: `sample_id, corpus, raw_response, pred` (`pred` = canonical word, or `""`
  when the model's answer could not be parsed = abstain).

## 4. Copy `qwen_predictions.parquet` back to the Mac repo, then import per corpus

The file holds every corpus in one parquet; `import-predictions` filters by `--corpus` and
writes one store entry per (study, corpus). Run from the Mac repo root:

```bash
# Study 2 (CAMEO) — one command per split:
for c in cafe crema_d emns emozionalmente enterface jl_corpus mesd nemo oreau pavoque ravdess resd subesco; do
  .venv/bin/python -m app.cli import-predictions --file qwen_predictions.parquet --study 2 --corpus "$c"
done

# Study 1 (deep PL+EN):
.venv/bin/python -m app.cli import-predictions --file qwen_predictions.parquet --study 1 --corpus nEMO
.venv/bin/python -m app.cli import-predictions --file qwen_predictions.parquet --study 1 --corpus RAVDESS
```

> Note the two distinct RAVDESS/nEMO namespaces: CAMEO's `ravdess`/`nemo` splits are Study 2
> (`sample_id` `ravdess:...` / `nemo:...`), while Study 1 uses the standalone RAVDESS wavs
> (`RAVDESS:...`) and standalone nEMO parquets (`nEMO:...`). The script emits all of them; the
> import commands above route each to the right study.

## Notes / limitations

- Zero-shot confidence is unreliable (spec §2), so we import a **hard label** only
  (`scores={pred:1.0}`, `confidence=None`); Study-2 confidence-correlation analyses skip this model.
- If the installed `transformers` exposes a different Qwen2-Audio processor API, the only place
  to adjust is `run_qwen.py`'s `main()` (the `apply_chat_template` / `processor(...)` calls).
  The prompt + parser must stay byte-identical to the Mac source of truth.
