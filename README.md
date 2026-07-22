# Cross-Lingual Agreement Between Text and Speech Emotion Recognition

**Towards Robust Multimodal Perception in Social Robots**

Research codebase for a study on **cross-modal agreement between text and speech
emotion recognition**, multilingual, with Polish as the reference point. The goal
is *not* to build a system, but to quantify **when and how modalities disagree** —
material for an article and a component of a doctoral dissertation.

> ⚠️ This is a research project. Results must reflect reality — no fabricated,
> guessed, or altered numbers, ever.

## Studies

The work splits into two independent studies (different data, mostly different
models, different research questions):

- **Study 1 — deep layer (PL, EN).** How exactly modalities diverge in concrete
  cases: where, how often, and which modality is closer to the truth on conflict.
  Datasets: **nEMO** (PL) and **RAVDESS** (EN, the only fully tri-modal point,
  incl. face). Metrics: accuracy / macro-F1, confusion matrices, Cohen's &
  Fleiss' kappa, conflict analysis, bootstrap CIs, McNemar / chi-square.
- **Study 2 — broad layer (8 languages).** Where Polish sits among other
  languages on text–speech agreement, and whether the pattern depends on
  language family or data resourcefulness. Dataset: **CAMEO** (PL, EN, FR, IT,
  ES, DE, RU, BN), zero-shot audio-LLM for speech + XLM-RoBERTa for text.

The canonical research plan lives in
[`instructions/konspekt_artykulu_ESM.md`](instructions/konspekt_artykulu_ESM.md);
the developer instruction in
[`instructions/programming_insctruction.md`](instructions/programming_insctruction.md).
Instruction/plan files are in Polish; all code, results and tests are in English.

## Repository layout

| Folder          | Purpose                                                        | Tracked? |
| --------------- | -------------------------------------------------------------- | -------- |
| `app/`          | All helper scripts and the research application itself.        | ✅       |
| `datasets/`     | Datasets we build on (nEMO, CAMEO; RAVDESS TBD).               | ❌ (§7)  |
| `instructions/` | Developer instructions, generated checkpoints, plans.          | ✅       |
| `models/`       | All models (own ones extracted from `old_projects/`).          | ❌ (§7)  |
| `old_projects/` | Legacy projects — source for own models. **Read-only.**        | ❌ (§7)  |
| `results/`      | Results grouped per study (well described, article-ready).     | ✅       |

Large assets (`datasets/`, `models/`, `old_projects/`) are excluded from version
control by design — see [`.gitignore`](.gitignore).

## Environment & reproduction

- Python **3.12** (`python3.12`, verified 3.12.1). One shared `venv` for the
  whole project. **No GPU — Apple M1 only** (a Windows + GPU fallback is allowed
  for prohibitively long runs).
- Uses **Pydantic** for configuration/validation.

```bash
# from the repository root
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` is provisional at this stage and will be pinned during the
> planning/implementation phase, once the concrete pipelines and model backends
> are fixed.

### Datasets

`datasets/` is not versioned. To reproduce:

- **nEMO** (PL) and **CAMEO** (8 languages) — clone from Hugging Face into
  `datasets/` (Git-LFS).
- **RAVDESS** (EN) — not yet downloaded (large). Source:
  <https://zenodo.org/records/1188976>. Scope (which files, how to extract face
  frames + emotion labels) to be decided during planning.
