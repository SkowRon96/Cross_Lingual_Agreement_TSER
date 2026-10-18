# Cross-Lingual Agreement Between Text and Speech Emotion Recognition

**Towards Robust Multimodal Perception in Social Robots**

A reproducible research pipeline for measuring **how far emotion-recognition models agree
with each other across modalities and across languages**, and what a social robot should do
when they disagree. The goal is not to ship a product but to quantify *when and how*
modalities diverge, and to turn that measurement into a decision rule. The code backs a
doctoral dissertation chapter and an accompanying article.

> This is a research project. Every reported number comes from the cached predictions in
> `results/`; nothing is estimated, rounded up, or filled in by hand.

Three studies share one pipeline:

1. **Cross-modal agreement** — Cohen's/Fleiss' κ between the text, speech and face channels
   on **nEMO** (Polish, acted), **RAVDESS** (English, acted, tri-modal) and **MELD**
   (English, natural dialogue), with bootstrap confidence intervals and conflict analysis.
2. **Cross-lingual agreement** — the same measurement across eight languages on **CAMEO**,
   asking whether text↔speech agreement tracks language or data provenance.
3. **Decision strategies** — late-fusion arbitration variants against the best single
   channel, plus **selective prediction**: abstaining when the channels conflict, scored on a
   risk–coverage curve.

A full description of the stages, their inputs and outputs is in
[`docs/PIPELINE.md`](docs/PIPELINE.md).

## Modules

| Path | Responsibility |
| --- | --- |
| [`app/cli.py`](app/cli.py) | The whole pipeline as four cached stages: `ingest`, `predict`, `analyze`, `report`, plus `import-predictions` for models run off-box. |
| [`app/config/`](app/config/) | Settings and paths, the canonical emotion set (Ekman-6 + neutral), and the static study design: which model ran where, headline vs. secondary, and which classes each model can emit. |
| [`app/labels/`](app/labels/) | Label harmonization. Every corpus- and model-specific label is mapped onto the canonical set; anything unmappable is dropped explicitly and counted, never guessed. |
| [`app/data/`](app/data/) | Corpus loaders (nEMO, CAMEO, RAVDESS, MELD), audio/video access, and the corpus index with its harmonization report. |
| [`app/models/`](app/models/) | One `EmotionModel` protocol over ten models in three modalities: Hugging Face text classifiers, a CNN-Transformer speech model, SpeechBrain, a ViT face model, a subprocess-isolated Keras LSTM, and the zero-shot audio-LLM imported from outside. |
| [`app/predict/`](app/predict/) | `PredictionStore` — a versioned parquet cache keyed by a model config signature, holding the **full harmonized probability vector** per sample, so analyses can be redone without re-running inference. |
| [`app/analysis/`](app/analysis/) | The statistics: common-support alignment, Cohen's and Fleiss' κ, percentile bootstrap CIs, McNemar and two-proportion tests, conflict and confidence analysis, and the Study-3 arbitration and selective-prediction logic. All results are Pydantic-typed. |
| [`app/report/`](app/report/) | The dissertation's visual system (palette, typography, number conventions) and the figure/table writers that render Markdown, LaTeX and PNG. |
| [`scripts/`](scripts/) | One-off tooling around the pipeline: the publication figures and tables, dataset/model overviews, MELD audio extraction, and the standalone Qwen2-Audio runner for a separate GPU machine. |
| [`tests/`](tests/) | Unit tests for the correctness-critical maths and label mapping; integration tests that touch real data or download models are marked `slow`. |
| [`weights/`](weights/) | Trained weights of the models developed for this work (CNN-Transformer speech, Keras LSTM Polish text). Versioned, ~25 MB. |
| [`results/`](results/) | Output directory. Not versioned — everything in it is regenerable from the code. |

## Data

The pipeline is a chain of cached artifacts, each stage reading the previous one:

```
corpus  ──ingest──▶  results/index/study{N}/<corpus>.parquet        gold labels + harmonization report
        ──predict─▶  results/predictions/study{N}/<corpus>/<model>.parquet
                                                                    predicted label, confidence,
                                                                    full harmonized score vector
        ──analyze─▶  results/analysis/study{N}/bundle.json + agreements.md
        ──report──▶  results/figures/… and results/tables/…
```

Because the prediction cache keeps the complete score vector, new analyses (including all of
Study 3) are computed offline from the cache — no re-inference, no model downloads, no
datasets. Predictions are invalidated automatically when a model's configuration changes.
Nothing under `results/` is versioned.

## Datasets

| Corpus | Role |
| --- | --- |
| **nEMO** | Polish acted speech — the Polish reference point for Study 1. |
| **RAVDESS** | English acted speech **and video** — the only fully tri-modal corpus, so the only place the face channel appears. |
| **MELD** | English natural dialogue — the contrast to acted speech, and where emotionally loaded text actually exists. |
| **CAMEO** | Eight languages (en, pl, de, fr, it, es, ru, bn) — the breadth layer for Study 2. |

None of them is redistributed here. **[`docs/DATASETS.md`](docs/DATASETS.md) is the complete
recipe:** where to download each corpus, the exact directory layout and columns the loaders
require, and the extra `ffmpeg` audio-extraction step MELD needs. Put them under `datasets/`
(or point `TSER_DATASETS_ROOT` elsewhere).

## How to run

```bash
# 1. environment (Python 3.12)
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. the four pipeline stages -- always run from the repository root
python -m app.cli ingest  --corpus nemo
python -m app.cli predict --study 1 --corpus nEMO --model mling_tabularis
python -m app.cli analyze --study 1
python -m app.cli report  --study 1

# 3. the publication figures/tables, once analyze has produced a bundle
PYTHONPATH=. python scripts/article_tables.py

# 4. tests (two are skipped without the RAVDESS files; `slow` ones download models)
pytest -m "not slow"
```

`predict` is idempotent: re-running it is a no-op unless the model configuration changed, so
the stages can be repeated freely. `python -m app.cli --help` lists every option. Paths in
the tests and in `scripts/` are resolved relative to the working directory, so both must be
invoked from the repository root.

Three things are optional and only needed for specific models:

- **Keras LSTM (Polish text).** Runs in its own interpreter to keep TensorFlow out of the
  main environment: create `.venv-tf` from `requirements-tf.txt`. As pinned
  (`tensorflow-macos`, Keras 2.15, numpy 1.26.4) this environment is macOS / Apple Silicon
  only. Every other model works without it.
- **Qwen2-Audio zero-shot speech.** Too large for this machine; it is run on a separate GPU
  box and the predictions are imported with `python -m app.cli import-predictions`. See
  [`scripts/qwen_windows/README.md`](scripts/qwen_windows/README.md).
- **MELD.** Needs `ffmpeg` on `PATH` for the one-off audio extraction, see
  [`docs/DATASETS.md`](docs/DATASETS.md).

Pretrained Hugging Face and SpeechBrain models download themselves on first use into
`models/hf_cache`. Inference runs on CPU, with Apple Silicon (`mps`) used when available.

## Citation

If you use this code or build on these results, please cite the dissertation chapter:

```bibtex
@phdthesis{skowron_cross_lingual_agreement,
  author  = {Skowron, Kamil},
  title   = {{TODO: dissertation title}},
  school  = {{TODO: institution}},
  year    = {{TODO: year}},
  note    = {Chapter: Cross-Lingual Agreement Between Text and Speech Emotion
             Recognition: Towards Robust Multimodal Perception in Social Robots}
}
```

The four corpora and all pretrained models have their own licences and citation
requirements; cite them from their original sources as listed in
[`docs/DATASETS.md`](docs/DATASETS.md).
