# Datasets — how to obtain and where to put them

`datasets/` is **not** versioned (see [`.gitignore`](../.gitignore)). This document is the
complete recipe for recreating it, so that `python -m app.cli ingest` finds everything it
expects. Every path and column below is what the loaders actually require — the loaders are
fail-loud, so a wrong layout raises rather than silently skipping data.

All four corpora are redistributed by their own authors under their own terms; obtain each
from its original source and respect its licence.

The root can be moved out of the repository with the `TSER_DATASETS_ROOT` environment
variable (likewise `TSER_RESULTS_ROOT` for outputs). The paths below assume the default,
`<repo>/datasets`.

---

## nEMO — Polish acted speech (Study 1)

- **Source:** <https://huggingface.co/datasets/amu-cai/nEMO> (clone with Git-LFS)
- **Layout:** `datasets/nEMO/data/*.parquet` (the repository ships two shards)
- **Required columns:** `file_id`, `emotion`, `normalized_text`, `speaker_id`, `gender`,
  `age`, plus an `audio` struct (`{bytes, path}`) — the waveform is embedded in the parquet,
  there is no separate wav tree.
- **Loader:** [`app/data/loaders/nemo.py`](../app/data/loaders/nemo.py)

```bash
git lfs install
git clone https://huggingface.co/datasets/amu-cai/nEMO datasets/nEMO
python -m app.cli ingest --corpus nemo
```

## CAMEO — eight languages (Study 2)

- **Source:** <https://huggingface.co/datasets/amu-cai/CAMEO> (clone with Git-LFS)
- **Layout:** `datasets/CAMEO/data/<corpus>-00000-of-00001.parquet`, one shard per source
  corpus. The `-<n>-of-<m>.parquet` suffix is stripped to derive the split name, so the file
  names must not be changed.
- **Required columns:** `file_id`, `emotion`, `transcription`, `speaker_id`, `gender`, `age`,
  `language`, plus the embedded `audio` struct. `language` must be one of the eight full
  names in `CAMEO_LANGUAGES` (English, Polish, French, Italian, Spanish, German, Russian,
  Bengali) — an unknown value raises `KeyError` by design.
- **Loader:** [`app/data/loaders/cameo.py`](../app/data/loaders/cameo.py)

```bash
git clone https://huggingface.co/datasets/amu-cai/CAMEO datasets/CAMEO
python -m app.cli ingest --corpus cameo
```

## RAVDESS — English acted speech + video (Study 1, the only tri-modal corpus)

- **Source:** Zenodo record 1188976 — <https://zenodo.org/records/1188976>. Download both the
  speech **audio** archive and the speech **video** archive.
- **Layout — note the two trees sit at different depths:**
  - audio: `datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_<NN>/03-*.wav`
  - video: `datasets/RAVDESS/Actor_<NN>/01-*.mp4`  (actor folders directly under `RAVDESS/`)
- **Pairing is mandatory.** Every `.wav` must have the matching `.mp4`; a missing video
  raises `FileNotFoundError` instead of dropping the sample. Labels and the spoken sentence
  are decoded from the 7-field RAVDESS filename; 60 speech utterances per actor.
- **Loader:** [`app/data/ravdess.py`](../app/data/ravdess.py)

```bash
python -m app.cli ingest --corpus ravdess
```

## MELD — English natural dialogue (Study 1)

- **Source:** the MELD project page, <https://affective-meld.github.io/> — the `MELD.Raw`
  archive (videos + the three sentence-emotion CSVs).
- **Layout:** `datasets/MELD/MELD.Raw/` containing

  | split | video subdirectory | CSV |
  | --- | --- | --- |
  | train | `train/train_splits` | `train/train_sent_emo.csv` |
  | dev | `dev_splits_complete` | `dev_sent_emo.csv` |
  | test | `output_repeated_splits_test` | `test_sent_emo.csv` |

  Video files are named `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`. The CSVs are
  **Windows-1252** encoded — keep them byte-identical to the distribution.
- **Extra step — audio extraction.** MELD ships video only, so the 16 kHz mono wavs have to
  be produced before ingest. This needs **`ffmpeg` on `PATH`** (an external dependency, not
  installed by `requirements.txt`). The script is idempotent and safe to re-run:

  ```bash
  PYTHONPATH=. python scripts/extract_meld_audio.py   # -> datasets/MELD/audio/{split}/dia{D}_utt{U}.wav
  python -m app.cli ingest --corpus meld
  ```

  Utterances whose video (or, at ingest, whose wav) is missing are skipped and counted in the
  harmonization report rather than failing the run.
- **Loader:** [`app/data/loaders/meld.py`](../app/data/loaders/meld.py)

---

## Models

No model weights need to be downloaded by hand.

- **Own trained models** are versioned in [`weights/`](../weights): the CNN-Transformer speech
  checkpoints (`weights/speech_cnn/`) and the Keras LSTM Polish text model
  (`weights/text_lstm/`). The LSTM leaf directory name is part of the model's cache
  signature, so do not rename it.
- **Pretrained models** (four Hugging Face text/face checkpoints and the SpeechBrain IEMOCAP
  model) download themselves on first use into `models/hf_cache`, which is not versioned.
- **Qwen2-Audio zero-shot** (`speech_zeroshot`, the headline speech model) is not run on this
  machine at all. It is executed on a separate GPU box and the predictions are brought in with
  `python -m app.cli import-predictions`; see
  [`scripts/qwen_windows/README.md`](../scripts/qwen_windows/README.md).

## Disk footprint, for planning

nEMO is roughly half a gigabyte, CAMEO a few gigabytes, RAVDESS a few gigabytes with video,
and MELD about ten gigabytes of video that reduces to ~1.5 GB of extracted wavs. Only MELD
requires the extraction step; the other three are read straight from the distribution.
