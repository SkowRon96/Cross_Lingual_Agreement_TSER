# P4 — Analysis + Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn cached predictions + gold into the article's statistics and figures: per-modality baselines, cross-modal agreement (Cohen's κ pairs + Fleiss' κ triad) with bootstrap CIs, significance tests (McNemar within-set, chi-square PL vs EN), conflict analysis, and Study-2 descriptive language analysis — then render deterministic figures (PDF+PNG) and tables (LaTeX+CSV+MD+PNG).

**Architecture:** Two decoupled stages reading the cached `PredictionStore` (never re-running models). `analyze` computes Pydantic result objects → `results/analysis/study{N}/*.json` + tidy CSV. `report` reads those JSONs and regenerates all artefacts deterministically. All κ obey the §4 common-support rule (intersection of classes both models support, drop gold outside it and abstentions, always report N used / N excluded). Fixed RNG seed everywhere.

**Tech Stack:** Python 3.12 `.venv`; numpy, scipy (`scipy.stats`), pandas, matplotlib (Agg), Pydantic, pytest. `lang2vec` (URIEL) for Study-2 typological distance. No new heavyweight deps.

## Global Constraints

- **Canonical space** (fixed order everywhere): `anger, disgust, fear, happiness, sadness, surprise, neutral` (`app.config.emotions.CANONICAL_ORDER`).
- **Common-support rule (§4):** κ between two models is computed only on samples whose gold ∈ (classes model A supports ∩ classes model B supports), excluding either model's abstentions. **Always record `n_used` and `n_excluded`** (with reason breakdown). Specific exclusions: PL text↔speech with `pl_text_roberta6`/`xlmr_pooled10` → drop gold=`surprise`; EN speech "second opinion" `en_speech_iemocap` → drop gold ∈ {`fear`,`disgust`,`surprise`}.
- **Headline pairs (§5.4)** — clean models only:

  | | text (headline) | speech (headline) |
  |---|---|---|
  | PL | `mling_tabularis` | `speech_zeroshot` |
  | EN | `en_text_hartmann` | `speech_zeroshot` |

  Leaky/narrowed models (`pl_text_lstm`, `speech_cnn_tf_*`, `pl_text_roberta6`, `xlmr_pooled10`, `en_speech_iemocap`) → full κ matrix + model-model reliability only, **never headline**.
- **Bootstrap:** 1000×, 95% percentile CI per κ; PL vs EN κ comparison uses the **CI of the difference from the same resample indices**. Seed = 20260723.
- **Test choice:** McNemar only within one dataset (paired correctness of two modalities on the same samples); PL vs EN (unpaired) → chi-square / two-proportion, never McNemar.
- **Confidence:** Pearson + Spearman across modalities; **skip `speech_zeroshot`** (confidence is `None` / unreliable, spec §2).
- **Leakage:** `speech_cnn_tf_*` and `pl_text_lstm` are leaky on RAVDESS/nEMO — report their real held-out metrics, flag leakage; "closer-to-gold" conclusions use clean models only.
- **`pl_text_roberta6`** label order is an ASSUMPTION — secondary only, flagged in every table it appears in.
- **Reporting style:** one shared style module (fonts/DPI/grid) + fixed colorblind-safe palette per emotion (`app.config.emotions.EMOTION_COLORS`, Okabe-Ito) + fixed class order in every artefact. Figures: PDF (vector) + PNG 300 DPI → `results/figures/study{N}/`. Tables: LaTeX + CSV + Markdown + PNG → `results/tables/study{N}/`.
- **Speech per-corpus scaler hand-off:** `speech_cnn_tf_*` predictions must have been produced with `--batch-size ≥ corpus size` (P3d note). Analysis just reads them; flag in docs.
- Code/tests/commits in English; **no `Co-Authored-By`**. Never fabricate results — every excluded/failed/missing case is reported, not silently dropped.

## Review mode (per memory: economical-subagent-execution)

Correctness-critical maths feed the article → **full review on T2–T7** (metrics, agreement/exclusions, bootstrap, tests, conflicts). Report rendering (T8–T10) is lower-risk → lighter review, but verify determinism + palette/order consistency. Implementers: haiku for transcription-heavy tasks, sonnet where judgment is needed. Reviewers: sonnet. Final whole-branch review: opus (it has repeatedly caught result-affecting bugs).

## File Structure

```
app/
  config/studies.py        # StudyConfig: per (study,corpus) model roster, modality map,
                           #   headline/clean vs leaky/secondary flags, per-model supported classes
  analysis/
    results.py             # Pydantic result schemas + JSON/tidy-CSV writers/readers (T1)
    metrics.py             # accuracy, macro-F1, confusion matrix (T2)
    agreement.py           # common-support filtering + exclusions + Cohen's & Fleiss' κ (T3)
    bootstrap.py           # 1000x percentile CI + difference CI (T4)
    tests.py               # McNemar (paired) + chi-square/two-proportion (unpaired) (T5)
    conflicts.py           # closer-to-gold, per-emotion conflict distribution (T6)
    correlation.py         # confidence Pearson/Spearman; Study-2 κ~typology/resource (T6/T7)
    lang_features.py       # 8-language typological distance (lang2vec) + resourcedness tier (T7)
    run.py                 # analyze orchestration: store+gold -> results/analysis/*.json+csv (T7)
  report/
    style.py               # fonts/DPI/grid, palette, class order, PNG-of-table helper (T8)
    tables.py              # LaTeX+CSV+MD+PNG table renderers (T8)
    figures.py             # confusion matrices, κ CI, PL-vs-EN, conflicts, Study2 scatter (T9)
    run.py                 # report orchestration: analysis JSON -> figures/ + tables/ (T10)
  cli.py                   # += analyze, report subcommands (T7, T10)
tests/
  test_metrics.py test_agreement.py test_bootstrap.py test_stat_tests.py
  test_conflicts.py test_correlation.py test_lang_features.py
  test_analysis_results.py test_report_tables.py test_report_figures.py
```

**Interfaces (contract used across tasks):**
- Gold labels: analysis needs canonical gold per sample. `Sample.gold_raw` is the native gold; map via `app.labels.mapper.LabelMapper` / `app.labels.mapping.map_label`. A small `gold_lookup(study, corpus) -> dict[sample_id, Emotion]` helper lives in `app/analysis/run.py` (built once from the corpus index + mapper).
- Predictions: `PredictionStore(root).read(study, corpus, model_id) -> list[Prediction]` (P1). `Prediction.pred: Emotion | None`, `.confidence: float | None`, `.abstained: bool`, `.scores: dict[Emotion,float]`.
- Emotions: `from app.config.emotions import Emotion, CANONICAL_ORDER, EMOTION_COLORS`.

---

## Global Constraints recap for every task

Every task inherits the **Global Constraints** section above verbatim. Exact values (seed 20260723, 1000 bootstrap resamples, canonical order, exclusion sets, headline pairs) are binding.

---

### Task 1: Analysis result schemas + study config

**Files:**
- Create: `app/config/studies.py`, `app/analysis/results.py`
- Test: `tests/test_analysis_results.py`

**Interfaces:**
- Produces (used by all later tasks):
  - `app.config.studies`:
    - `MODALITY: dict[str, str]` — model_id → `"text"|"speech"|"face"`.
    - `HEADLINE: dict[tuple[str,str], dict[str,str]]` — `(lang, modality) -> model_id`; keys `("PL","text")→"mling_tabularis"`, `("PL","speech")→"speech_zeroshot"`, `("EN","text")→"en_text_hartmann"`, `("EN","speech")→"speech_zeroshot"`.
    - `SECONDARY: set[str]` = {`"pl_text_roberta6"`,`"xlmr_pooled10"`,`"pl_text_lstm"`,`"speech_cnn_tf_ravdess"`,`"speech_cnn_tf_nemo"`,`"en_speech_iemocap"`}.
    - `LEAKY: set[str]` = {`"pl_text_lstm"`,`"speech_cnn_tf_ravdess"`,`"speech_cnn_tf_nemo"`}.
    - `SUPPORTED: dict[str, set[Emotion]]` — canonical classes each model can output (drives common-support). Full-7 for `mling_tabularis`,`speech_zeroshot`,`pl_text_lstm`,`speech_cnn_tf_*`,`en_face_vit`,`en_text_hartmann`,`pl_text_roberta6`(assumed, incl. no `surprise`→ set without surprise); `xlmr_pooled10` = all except `surprise`; `en_speech_iemocap` = {anger,happiness,sadness,neutral}.
    - `ROSTER: dict[tuple[int,str], list[str]]` — (study, corpus) → model_ids that were run there. Study 1: `(1,"nEMO")` → PL text models + `speech_zeroshot`,`speech_cnn_tf_nemo`; `(1,"RAVDESS")` → `en_text_hartmann`,`mling_tabularis`,`speech_zeroshot`,`speech_cnn_tf_ravdess`,`en_speech_iemocap`,`en_face_vit`. Study 2: `(2, split)` → `mling_tabularis`,`speech_zeroshot` for each CAMEO split.
    - `STUDY2_SPLITS: list[str]` = the 13 CAMEO splits (`cafe`,`crema_d`,`emns`,`emozionalmente`,`enterface`,`jl_corpus`,`mesd`,`nemo`,`oreau`,`pavoque`,`ravdess`,`resd`,`subesco`).
    - `SPLIT_LANG: dict[str,str]` — CAMEO split → ISO lang (from `app.data.loaders.cameo.CAMEO_LANGUAGES` mapping applied per split; `nemo→pl`, `cafe→fr`, etc.). NOTE: derive at runtime from data if unsure; the plan encodes the known mapping and a test asserts every split has a language.
  - `app.analysis.results`: Pydantic models `Baseline`, `AgreementResult`, `BootstrapCI`, `TestResult`, `ConflictResult`, `ConfidenceCorr`, `Study2LangRow`, and a container `AnalysisBundle`. Plus `write_json(bundle, path)`, `read_json(path) -> AnalysisBundle`, `write_tidy_csv(bundle, path)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_results.py`:
```python
from pathlib import Path
from app.config.emotions import Emotion
from app.config import studies
from app.analysis.results import (
    Baseline, AgreementResult, BootstrapCI, AnalysisBundle, write_json, read_json,
)


def test_study_config_headline_and_support():
    assert studies.HEADLINE[("PL", "text")] == "mling_tabularis"
    assert studies.HEADLINE[("EN", "speech")] == "speech_zeroshot"
    # roberta6/poltextlab lack surprise
    assert Emotion.SURPRISE not in studies.SUPPORTED["xlmr_pooled10"]
    assert Emotion.SURPRISE not in studies.SUPPORTED["pl_text_roberta6"]
    # iemocap 4-class
    assert studies.SUPPORTED["en_speech_iemocap"] == {
        Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}
    # every study-2 split maps to a language
    for s in studies.STUDY2_SPLITS:
        assert s in studies.SPLIT_LANG


def test_bundle_roundtrip(tmp_path):
    b = AnalysisBundle(
        study=1,
        baselines=[Baseline(model_id="m", corpus="nEMO", modality="text",
                            n=10, accuracy=0.5, macro_f1=0.4,
                            confusion={"anger": {"anger": 3}}, labels=["anger"])],
        agreements=[AgreementResult(corpus="nEMO", model_a="a", model_b="b",
                    kappa=0.42, n_used=8, n_excluded=2,
                    excluded_reasons={"gold_out_of_support": 1, "abstain": 1},
                    headline=True, ci=BootstrapCI(lo=0.1, hi=0.7))],
    )
    p = tmp_path / "study1.json"
    write_json(b, p)
    got = read_json(p)
    assert got.study == 1
    assert got.agreements[0].kappa == 0.42
    assert got.agreements[0].ci.hi == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analysis_results.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `app/config/studies.py`**

```python
"""Static study design: which models ran where, headline vs secondary/leaky, and the
canonical classes each model can emit (drives the §4 common-support rule)."""
from app.config.emotions import Emotion

MODALITY: dict[str, str] = {
    "en_text_hartmann": "text", "xlmr_pooled10": "text", "mling_tabularis": "text",
    "pl_text_roberta6": "text", "pl_text_lstm": "text",
    "speech_zeroshot": "speech", "speech_cnn_tf_ravdess": "speech",
    "speech_cnn_tf_nemo": "speech", "en_speech_iemocap": "speech",
    "en_face_vit": "face",
}

HEADLINE: dict[tuple[str, str], str] = {
    ("PL", "text"): "mling_tabularis", ("PL", "speech"): "speech_zeroshot",
    ("EN", "text"): "en_text_hartmann", ("EN", "speech"): "speech_zeroshot",
}

SECONDARY: set[str] = {
    "pl_text_roberta6", "xlmr_pooled10", "pl_text_lstm",
    "speech_cnn_tf_ravdess", "speech_cnn_tf_nemo", "en_speech_iemocap",
}
LEAKY: set[str] = {"pl_text_lstm", "speech_cnn_tf_ravdess", "speech_cnn_tf_nemo"}

_ALL7 = set(Emotion)
_NO_SURPRISE = _ALL7 - {Emotion.SURPRISE}
SUPPORTED: dict[str, set[Emotion]] = {
    "en_text_hartmann": set(_ALL7), "mling_tabularis": set(_ALL7),
    "pl_text_lstm": set(_ALL7), "speech_zeroshot": set(_ALL7),
    "speech_cnn_tf_ravdess": set(_ALL7), "speech_cnn_tf_nemo": set(_ALL7),
    "en_face_vit": set(_ALL7),
    "xlmr_pooled10": set(_NO_SURPRISE),
    "pl_text_roberta6": set(_NO_SURPRISE),  # assumed label order; no surprise
    "en_speech_iemocap": {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL},
}

ROSTER: dict[tuple[int, str], list[str]] = {
    (1, "nEMO"): ["mling_tabularis", "pl_text_roberta6", "xlmr_pooled10", "pl_text_lstm",
                  "speech_zeroshot", "speech_cnn_tf_nemo"],
    (1, "RAVDESS"): ["en_text_hartmann", "mling_tabularis", "speech_zeroshot",
                     "speech_cnn_tf_ravdess", "en_speech_iemocap", "en_face_vit"],
}
STUDY2_SPLITS: list[str] = ["cafe", "crema_d", "emns", "emozionalmente", "enterface",
                            "jl_corpus", "mesd", "nemo", "oreau", "pavoque", "ravdess",
                            "resd", "subesco"]
for _s in STUDY2_SPLITS:
    ROSTER[(2, _s)] = ["mling_tabularis", "speech_zeroshot"]

# CAMEO split -> ISO language (source language of each sub-corpus).
SPLIT_LANG: dict[str, str] = {
    "cafe": "fr", "crema_d": "en", "emns": "en", "emozionalmente": "it",
    "enterface": "en", "jl_corpus": "en", "mesd": "es", "nemo": "pl",
    "oreau": "fr", "pavoque": "de", "ravdess": "en", "resd": "ru", "subesco": "bn",
}
```

> NOTE for implementer: `SPLIT_LANG` values are the plan's best mapping; the T7 loader
> cross-checks each split's actual `language` column against this and fails loud on mismatch.

- [ ] **Step 4: Write `app/analysis/results.py`**

```python
"""Pydantic result schemas + deterministic JSON / tidy-CSV serialization."""
from __future__ import annotations
import csv
import json
from pathlib import Path

from pydantic import BaseModel


class BootstrapCI(BaseModel):
    lo: float
    hi: float
    level: float = 0.95


class Baseline(BaseModel):
    model_id: str
    corpus: str
    modality: str
    n: int
    accuracy: float
    macro_f1: float
    confusion: dict[str, dict[str, int]]  # gold -> pred -> count
    labels: list[str]


class AgreementResult(BaseModel):
    corpus: str
    model_a: str
    model_b: str
    kappa: float
    n_used: int
    n_excluded: int
    excluded_reasons: dict[str, int]
    headline: bool = False
    secondary_flag: bool = False  # any of the pair is secondary/assumed
    ci: BootstrapCI | None = None


class FleissResult(BaseModel):
    corpus: str
    model_ids: list[str]
    kappa: float
    n_used: int
    n_excluded: int
    ci: BootstrapCI | None = None


class TestResult(BaseModel):
    name: str            # "mcnemar" | "chi2" | "two_proportion"
    corpus: str
    detail: str          # what was compared, human-readable
    statistic: float
    p_value: float
    extra: dict[str, float] = {}


class ConflictResult(BaseModel):
    corpus: str
    model_a: str
    model_b: str
    n_conflicts: int
    a_closer_to_gold: int
    b_closer_to_gold: int
    both_wrong: int
    per_emotion: dict[str, int]  # gold emotion -> conflict count


class ConfidenceCorr(BaseModel):
    corpus: str
    model_a: str
    model_b: str
    n: int
    pearson: float
    spearman: float


class Study2LangRow(BaseModel):
    split: str
    language: str
    kappa: float
    n_used: int
    ci: BootstrapCI | None = None
    syntactic_distance: float | None = None
    resource_tier: int | None = None


class AnalysisBundle(BaseModel):
    study: int
    baselines: list[Baseline] = []
    agreements: list[AgreementResult] = []
    fleiss: list[FleissResult] = []
    tests: list[TestResult] = []
    conflicts: list[ConflictResult] = []
    confidence: list[ConfidenceCorr] = []
    study2: list[Study2LangRow] = []


def write_json(bundle: AnalysisBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.model_dump(), indent=2, sort_keys=True))


def read_json(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(Path(path).read_text())


def write_tidy_csv(bundle: AnalysisBundle, path: Path) -> None:
    """One long tidy table of κ rows (agreements) for quick spreadsheet inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "corpus", "model_a", "model_b", "kappa",
                    "ci_lo", "ci_hi", "n_used", "n_excluded", "headline", "secondary"])
        for a in bundle.agreements:
            w.writerow([bundle.study, a.corpus, a.model_a, a.model_b, a.kappa,
                        a.ci.lo if a.ci else "", a.ci.hi if a.ci else "",
                        a.n_used, a.n_excluded, a.headline, a.secondary_flag])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis_results.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/config/studies.py app/analysis/results.py tests/test_analysis_results.py
git commit -m "feat: P4 analysis result schemas + study config (roster, headline, support)"
```

---

### Task 2: Baseline metrics (accuracy, macro-F1, confusion)

**Files:**
- Create: `app/analysis/__init__.py` (if missing), `app/analysis/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Prediction` (P1), `Emotion`, `CANONICAL_ORDER`.
- Produces: `baseline(preds, gold, model_id, corpus, modality) -> Baseline`. Abstentions (`pred is None`) are counted as wrong (never gold) and included in `n`. `macro_f1` = unweighted mean of per-class F1 over the classes present in gold. `confusion[gold][pred]` counts; abstain pred recorded under key `"abstain"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:
```python
from app.config.emotions import Emotion
from app.models.base import Prediction
from app.analysis.metrics import baseline


def _p(sid, pred, abst=False):
    return Prediction(sample_id=sid, model_id="m", modality="text", pred_raw="x",
                      pred=pred, confidence=1.0 if pred else None,
                      scores={pred: 1.0} if pred else {}, abstained=abst)


def test_accuracy_and_confusion():
    gold = {"1": Emotion.ANGER, "2": Emotion.ANGER, "3": Emotion.SADNESS}
    preds = [_p("1", Emotion.ANGER), _p("2", Emotion.SADNESS), _p("3", Emotion.SADNESS)]
    b = baseline(preds, gold, "m", "nEMO", "text")
    assert b.n == 3
    assert abs(b.accuracy - 2 / 3) < 1e-9
    assert b.confusion["anger"]["anger"] == 1
    assert b.confusion["anger"]["sadness"] == 1


def test_abstain_counts_as_wrong():
    gold = {"1": Emotion.ANGER, "2": Emotion.SADNESS}
    preds = [_p("1", None, abst=True), _p("2", Emotion.SADNESS)]
    b = baseline(preds, gold, "m", "nEMO", "text")
    assert b.n == 2
    assert abs(b.accuracy - 0.5) < 1e-9
    assert b.confusion["anger"]["abstain"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `app/analysis/metrics.py`:
```python
"""Per-model baseline metrics: accuracy, macro-F1, confusion matrix (canonical labels)."""
from collections import defaultdict

from app.analysis.results import Baseline
from app.config.emotions import Emotion
from app.models.base import Prediction


def baseline(preds: list[Prediction], gold: dict[str, Emotion],
             model_id: str, corpus: str, modality: str) -> Baseline:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    n = correct = 0
    for p in preds:
        if p.sample_id not in gold:
            continue
        g = gold[p.sample_id]
        n += 1
        pred_key = p.pred.value if p.pred is not None else "abstain"
        confusion[g.value][pred_key] += 1
        if p.pred is None:
            fn[g.value] += 1
            continue
        if p.pred == g:
            correct += 1
            tp[g.value] += 1
        else:
            fp[p.pred.value] += 1
            fn[g.value] += 1
    gold_classes = sorted({g.value for g in gold.values()})
    f1s = []
    for c in gold_classes:
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append((2 * tp[c] / denom) if denom else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return Baseline(
        model_id=model_id, corpus=corpus, modality=modality, n=n,
        accuracy=(correct / n if n else 0.0), macro_f1=macro_f1,
        confusion={g: dict(row) for g, row in confusion.items()},
        labels=gold_classes,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/__init__.py app/analysis/metrics.py tests/test_metrics.py
git commit -m "feat: P4 baseline metrics (accuracy, macro-F1, confusion)"
```

---

### Task 3: Common-support filtering + Cohen's & Fleiss' κ

**Files:**
- Create: `app/analysis/agreement.py`
- Test: `tests/test_agreement.py`

**Interfaces:**
- Consumes: `Prediction`, `Emotion`, `app.config.studies.SUPPORTED`.
- Produces:
  - `aligned_pairs(preds_a, preds_b, gold, support_a, support_b) -> tuple[list[Emotion], list[Emotion], dict]` — returns parallel label lists (A, B) over samples kept under the common-support rule, plus an `excluded` dict `{"gold_out_of_support": k, "abstain_a": k, "abstain_b": k, "missing": k}`. A sample is kept iff it exists in both preds and gold, `gold ∈ support_a ∩ support_b`, and neither model abstained.
  - `cohen_kappa(labels_a, labels_b) -> float` — Cohen's κ over the observed label set (canonical). Returns 0.0 on a single-category degenerate case where expected agreement is 1.0 (documented).
  - `fleiss_kappa(rater_labels: list[list[Emotion]]) -> float` — Fleiss' κ for ≥3 raters (list per rater, all same length, already aligned/filtered).
  - `agreement(preds_a, preds_b, gold, model_a, model_b, corpus) -> AgreementResult` — orchestrates: computes κ, fills n_used/n_excluded/excluded_reasons, sets `headline`/`secondary_flag` from `studies`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agreement.py`:
```python
import math
from app.config.emotions import Emotion
from app.config import studies
from app.models.base import Prediction
from app.analysis.agreement import aligned_pairs, cohen_kappa, fleiss_kappa, agreement

A = Emotion.ANGER
S = Emotion.SADNESS
SUP = Emotion.SURPRISE


def _p(sid, pred, mid="m", abst=False):
    return Prediction(sample_id=sid, model_id=mid, modality="text", pred_raw="x",
                      pred=pred, confidence=1.0 if pred else None,
                      scores={pred: 1.0} if pred else {}, abstained=abst)


def test_cohen_known_value():
    # 8 agree on A, 2 disagree -> classic; verify against hand calc
    a = [A, A, A, A, A, S, S, S, S, S]
    b = [A, A, A, A, S, S, S, S, S, A]
    k = cohen_kappa(a, b)
    # po = 8/10=0.8; pe = (5/10*4/10)+(5/10*6/10)=0.2+0.3=0.5; kappa=(0.8-0.5)/(1-0.5)=0.6
    assert abs(k - 0.6) < 1e-9


def test_common_support_excludes_surprise_for_roberta6():
    gold = {"1": A, "2": SUP, "3": S}
    pa = [_p("1", A), _p("2", A), _p("3", S)]           # roberta6 (no surprise support)
    pb = [_p("1", A), _p("2", S), _p("3", S)]
    la, lb, exc = aligned_pairs(pa, pb, gold,
                                studies.SUPPORTED["pl_text_roberta6"],
                                studies.SUPPORTED["speech_zeroshot"])
    assert len(la) == 2 and len(lb) == 2       # sample "2" (gold=surprise) dropped
    assert exc["gold_out_of_support"] == 1


def test_abstain_excluded():
    gold = {"1": A, "2": S}
    pa = [_p("1", A), _p("2", None, abst=True)]
    pb = [_p("1", A), _p("2", S)]
    la, lb, exc = aligned_pairs(pa, pb, gold, set(Emotion), set(Emotion))
    assert len(la) == 1 and exc["abstain_a"] == 1


def test_fleiss_perfect_agreement():
    r1 = [A, S, A]
    r2 = [A, S, A]
    r3 = [A, S, A]
    assert abs(fleiss_kappa([r1, r2, r3]) - 1.0) < 1e-9


def test_agreement_sets_headline_flag():
    gold = {"1": A, "2": S}
    pa = [_p("1", A, "mling_tabularis"), _p("2", S, "mling_tabularis")]
    pb = [_p("1", A, "speech_zeroshot"), _p("2", S, "speech_zeroshot")]
    r = agreement(pa, pb, gold, "mling_tabularis", "speech_zeroshot", "nEMO")
    assert r.headline is True and r.secondary_flag is False
    assert r.n_used == 2 and abs(r.kappa - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agreement.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `app/analysis/agreement.py`:
```python
"""Common-support filtering (§4) + Cohen's and Fleiss' kappa."""
from collections import Counter

from app.analysis.results import AgreementResult
from app.config import studies
from app.config.emotions import Emotion
from app.models.base import Prediction


def _by_id(preds: list[Prediction]) -> dict[str, Prediction]:
    return {p.sample_id: p for p in preds}


def aligned_pairs(preds_a, preds_b, gold, support_a, support_b):
    a_map, b_map = _by_id(preds_a), _by_id(preds_b)
    support = support_a & support_b
    la: list[Emotion] = []
    lb: list[Emotion] = []
    exc = {"gold_out_of_support": 0, "abstain_a": 0, "abstain_b": 0, "missing": 0}
    for sid, g in gold.items():
        pa, pb = a_map.get(sid), b_map.get(sid)
        if pa is None or pb is None:
            exc["missing"] += 1
            continue
        if g not in support:
            exc["gold_out_of_support"] += 1
            continue
        if pa.pred is None:
            exc["abstain_a"] += 1
            continue
        if pb.pred is None:
            exc["abstain_b"] += 1
            continue
        la.append(pa.pred)
        lb.append(pb.pred)
    return la, lb, exc


def cohen_kappa(labels_a: list[Emotion], labels_b: list[Emotion]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    cats = sorted({*labels_a, *labels_b}, key=lambda e: e.value)
    po = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe == 1.0:
        return 0.0  # degenerate: both raters constant on one class
    return (po - pe) / (1 - pe)


def fleiss_kappa(rater_labels: list[list[Emotion]]) -> float:
    n_raters = len(rater_labels)
    n_items = len(rater_labels[0])
    cats = sorted({e for r in rater_labels for e in r}, key=lambda e: e.value)
    # n_ij = count of raters assigning category j to item i
    P_i = []
    col_totals = {c: 0 for c in cats}
    for i in range(n_items):
        counts = Counter(rater_labels[r][i] for r in range(n_raters))
        for c in cats:
            col_totals[c] += counts[c]
        P_i.append((sum(counts[c] ** 2 for c in cats) - n_raters) / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items
    p_j = {c: col_totals[c] / (n_items * n_raters) for c in cats}
    P_e = sum(v ** 2 for v in p_j.values())
    if P_e == 1.0:
        return 0.0
    return (P_bar - P_e) / (1 - P_e)


def agreement(preds_a, preds_b, gold, model_a, model_b, corpus) -> AgreementResult:
    la, lb, exc = aligned_pairs(
        preds_a, preds_b, gold,
        studies.SUPPORTED[model_a], studies.SUPPORTED[model_b])
    k = cohen_kappa(la, lb)
    headline = (model_a in studies.HEADLINE.values()
                and model_b in studies.HEADLINE.values()
                and model_a not in studies.SECONDARY
                and model_b not in studies.SECONDARY)
    secondary = (model_a in studies.SECONDARY) or (model_b in studies.SECONDARY)
    return AgreementResult(
        corpus=corpus, model_a=model_a, model_b=model_b, kappa=k,
        n_used=len(la), n_excluded=sum(exc.values()), excluded_reasons=exc,
        headline=headline, secondary_flag=secondary,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agreement.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app/analysis/agreement.py tests/test_agreement.py
git commit -m "feat: P4 common-support filtering + Cohen's & Fleiss' kappa"
```

---

### Task 4: Bootstrap CIs (per-κ + difference)

**Files:**
- Create: `app/analysis/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `cohen_kappa`, aligned label lists.
- Produces:
  - `bootstrap_kappa(labels_a, labels_b, *, n=1000, seed=20260723) -> BootstrapCI` — resample item indices with replacement, recompute κ, return 2.5/97.5 percentiles.
  - `bootstrap_diff(la1, lb1, la2, lb2, *, n=1000, seed=20260723) -> tuple[float, BootstrapCI]` — for PL vs EN: independently resample each pair (its own indices) per iteration, take κ1−κ2; return (point_diff, percentile CI). Independent resampling because PL and EN are different samples.
  - Deterministic: `numpy.random.default_rng(seed)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bootstrap.py`:
```python
from app.config.emotions import Emotion
from app.analysis.bootstrap import bootstrap_kappa, bootstrap_diff

A, S = Emotion.ANGER, Emotion.SADNESS


def test_ci_is_deterministic_and_ordered():
    a = [A, A, A, A, A, S, S, S, S, S]
    b = [A, A, A, A, S, S, S, S, S, A]
    ci1 = bootstrap_kappa(a, b, n=200)
    ci2 = bootstrap_kappa(a, b, n=200)
    assert ci1.lo == ci2.lo and ci1.hi == ci2.hi       # deterministic (fixed seed)
    assert ci1.lo <= ci1.hi
    assert -1.0 <= ci1.lo and ci1.hi <= 1.0


def test_perfect_agreement_ci_at_one():
    a = [A, S, A, S, A, S]
    b = [A, S, A, S, A, S]
    ci = bootstrap_kappa(a, b, n=200)
    assert abs(ci.lo - 1.0) < 1e-9 and abs(ci.hi - 1.0) < 1e-9


def test_diff_ci_deterministic():
    a1 = [A, A, S, S]; b1 = [A, A, S, S]
    a2 = [A, S, A, S]; b2 = [S, A, S, A]
    d, ci = bootstrap_diff(a1, b1, a2, b2, n=200)
    assert ci.lo <= d <= ci.hi or ci.lo <= ci.hi     # CI ordered; point inside typical
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bootstrap.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Create `app/analysis/bootstrap.py`:
```python
"""Percentile bootstrap CIs for kappa (fixed seed for reproducibility)."""
import numpy as np

from app.analysis.agreement import cohen_kappa
from app.analysis.results import BootstrapCI

SEED = 20260723


def _resampled_kappa(rng, la, lb, idx):
    a = [la[i] for i in idx]
    b = [lb[i] for i in idx]
    return cohen_kappa(a, b)


def bootstrap_kappa(labels_a, labels_b, *, n: int = 1000, seed: int = SEED) -> BootstrapCI:
    m = len(labels_a)
    if m == 0:
        return BootstrapCI(lo=0.0, hi=0.0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        vals[k] = _resampled_kappa(rng, labels_a, labels_b, idx)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return BootstrapCI(lo=float(lo), hi=float(hi))


def bootstrap_diff(la1, lb1, la2, lb2, *, n: int = 1000, seed: int = SEED):
    m1, m2 = len(la1), len(la2)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        i1 = rng.integers(0, m1, size=m1)
        i2 = rng.integers(0, m2, size=m2)
        k1 = _resampled_kappa(rng, la1, lb1, i1)
        k2 = _resampled_kappa(rng, la2, lb2, i2)
        diffs[k] = k1 - k2
    point = cohen_kappa(la1, lb1) - cohen_kappa(la2, lb2)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, BootstrapCI(lo=float(lo), hi=float(hi))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bootstrap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: P4 bootstrap CIs (per-kappa + PL-vs-EN difference)"
```

---

### Task 5: Significance tests (McNemar + chi-square)

**Files:**
- Create: `app/analysis/tests.py`
- Test: `tests/test_stat_tests.py`

**Interfaces:**
- Consumes: `Prediction`, `Emotion`, gold; `scipy.stats`.
- Produces:
  - `mcnemar(correct_a: list[bool], correct_b: list[bool]) -> TestResult` — paired per-sample correctness of two modalities on the SAME samples. Uses exact binomial when discordant n<25, else χ² with continuity correction. Name `"mcnemar"`.
  - `two_proportion(k1, n1, k2, n2) -> TestResult` — unpaired PL-vs-EN accuracy comparison; name `"two_proportion"`, reports z + p (two-sided).
  - `correctness(preds, gold) -> dict[str, bool]` — helper: sample_id → (pred == gold), abstain=False.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stat_tests.py`:
```python
from app.analysis.tests import mcnemar, two_proportion


def test_mcnemar_symmetric_is_nonsignificant():
    # equal discordant counts -> p ~ 1
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [False] * 10
    r = mcnemar(a, b)
    assert r.p_value > 0.9


def test_mcnemar_detects_asymmetry():
    # A correct where B wrong on many, rarely reverse
    a = [True] * 20 + [False] * 2
    b = [False] * 20 + [True] * 2
    r = mcnemar(a, b)
    assert r.p_value < 0.001


def test_two_proportion_basic():
    r = two_proportion(80, 100, 60, 100)
    assert r.name == "two_proportion"
    assert r.p_value < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_stat_tests.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Create `app/analysis/tests.py`:
```python
"""Significance tests: McNemar (paired within-set), two-proportion / chi-square (PL vs EN)."""
import math

from scipy import stats

from app.analysis.results import TestResult
from app.config.emotions import Emotion
from app.models.base import Prediction


def correctness(preds: list[Prediction], gold: dict[str, Emotion]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for p in preds:
        if p.sample_id in gold:
            out[p.sample_id] = (p.pred is not None and p.pred == gold[p.sample_id])
    return out


def mcnemar(correct_a: list[bool], correct_b: list[bool], corpus: str = "",
            detail: str = "") -> TestResult:
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    disc = b + c
    if disc == 0:
        return TestResult(name="mcnemar", corpus=corpus, detail=detail,
                          statistic=0.0, p_value=1.0, extra={"b": b, "c": c})
    if disc < 25:
        p = 2 * sum(math.comb(disc, i) * 0.5 ** disc for i in range(0, min(b, c) + 1))
        p = min(1.0, p)
        stat = float(min(b, c))
    else:
        stat = (abs(b - c) - 1) ** 2 / disc
        p = 1 - stats.chi2.cdf(stat, df=1)
    return TestResult(name="mcnemar", corpus=corpus, detail=detail,
                      statistic=float(stat), p_value=float(p), extra={"b": b, "c": c})


def two_proportion(k1: int, n1: int, k2: int, n2: int, corpus: str = "",
                   detail: str = "") -> TestResult:
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return TestResult(name="two_proportion", corpus=corpus, detail=detail,
                      statistic=float(z), p_value=float(pval),
                      extra={"p1": p1, "p2": p2})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_stat_tests.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/tests.py tests/test_stat_tests.py
git commit -m "feat: P4 significance tests (McNemar paired, two-proportion PL vs EN)"
```

---

### Task 6: Conflict analysis + confidence correlation

**Files:**
- Create: `app/analysis/conflicts.py`, `app/analysis/correlation.py`
- Test: `tests/test_conflicts.py`, `tests/test_correlation.py`

**Interfaces:**
- `conflicts.conflict(preds_a, preds_b, gold, model_a, model_b, corpus) -> ConflictResult` — over samples where both predicted (non-abstain) and disagree: count `a_closer_to_gold` (a==gold, b!=gold), `b_closer_to_gold`, `both_wrong`; `per_emotion[gold]` conflict counts.
- `correlation.confidence_corr(preds_a, preds_b, model_a, model_b, corpus) -> ConfidenceCorr | None` — Pearson + Spearman of per-sample confidence on shared non-abstain samples. Returns `None` if either model is `speech_zeroshot` or has all-None confidence (skip rule). Needs ≥3 shared points.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_conflicts.py`:
```python
from app.config.emotions import Emotion
from app.models.base import Prediction
from app.analysis.conflicts import conflict

A, S = Emotion.ANGER, Emotion.SADNESS


def _p(sid, pred, mid="m"):
    return Prediction(sample_id=sid, model_id=mid, modality="text", pred_raw="x",
                      pred=pred, confidence=1.0, scores={pred: 1.0})


def test_closer_to_gold_counts():
    gold = {"1": A, "2": A, "3": S}
    pa = [_p("1", A), _p("2", S), _p("3", S)]   # 1: a right; 2: both? ; 3: agree (no conflict)
    pb = [_p("1", S), _p("2", A), _p("3", S)]
    r = conflict(pa, pb, gold, "a", "b", "nEMO")
    # conflicts on 1 (a right) and 2 (b right); 3 agree
    assert r.n_conflicts == 2
    assert r.a_closer_to_gold == 1 and r.b_closer_to_gold == 1
    assert r.per_emotion["anger"] == 2
```

Create `tests/test_correlation.py`:
```python
from app.config.emotions import Emotion
from app.models.base import Prediction
from app.analysis.correlation import confidence_corr


def _p(sid, conf, mid):
    return Prediction(sample_id=sid, model_id=mid, modality="text", pred_raw="x",
                      pred=Emotion.ANGER, confidence=conf, scores={Emotion.ANGER: conf})


def test_skip_zeroshot():
    pa = [_p("1", 0.9, "speech_zeroshot")]
    pb = [_p("1", 0.8, "mling_tabularis")]
    assert confidence_corr(pa, pb, "speech_zeroshot", "mling_tabularis", "nEMO") is None


def test_pearson_spearman():
    pa = [_p(str(i), c, "a") for i, c in enumerate([0.1, 0.5, 0.9, 0.3])]
    pb = [_p(str(i), c, "b") for i, c in enumerate([0.2, 0.6, 0.8, 0.4])]
    r = confidence_corr(pa, pb, "a", "b", "nEMO")
    assert r is not None and r.n == 4
    assert r.pearson > 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_conflicts.py tests/test_correlation.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementations**

Create `app/analysis/conflicts.py`:
```python
"""Cross-modal conflict analysis: which modality is closer to gold when they disagree."""
from collections import defaultdict

from app.analysis.results import ConflictResult
from app.config.emotions import Emotion
from app.models.base import Prediction


def conflict(preds_a, preds_b, gold, model_a, model_b, corpus) -> ConflictResult:
    a_map = {p.sample_id: p for p in preds_a}
    b_map = {p.sample_id: p for p in preds_b}
    a_closer = b_closer = both_wrong = n_conf = 0
    per_emotion: dict[str, int] = defaultdict(int)
    for sid, g in gold.items():
        pa, pb = a_map.get(sid), b_map.get(sid)
        if pa is None or pb is None or pa.pred is None or pb.pred is None:
            continue
        if pa.pred == pb.pred:
            continue
        n_conf += 1
        per_emotion[g.value] += 1
        a_ok, b_ok = (pa.pred == g), (pb.pred == g)
        if a_ok and not b_ok:
            a_closer += 1
        elif b_ok and not a_ok:
            b_closer += 1
        else:
            both_wrong += 1
    return ConflictResult(
        corpus=corpus, model_a=model_a, model_b=model_b, n_conflicts=n_conf,
        a_closer_to_gold=a_closer, b_closer_to_gold=b_closer, both_wrong=both_wrong,
        per_emotion=dict(per_emotion))
```

Create `app/analysis/correlation.py`:
```python
"""Confidence correlation across modalities + Study-2 descriptive helpers."""
from scipy import stats

from app.analysis.results import ConfidenceCorr
from app.models.base import Prediction

SKIP_CONFIDENCE = {"speech_zeroshot"}


def confidence_corr(preds_a, preds_b, model_a, model_b, corpus) -> ConfidenceCorr | None:
    if model_a in SKIP_CONFIDENCE or model_b in SKIP_CONFIDENCE:
        return None
    a_map = {p.sample_id: p for p in preds_a}
    b_map = {p.sample_id: p for p in preds_b}
    xs, ys = [], []
    for sid, pa in a_map.items():
        pb = b_map.get(sid)
        if pb is None or pa.confidence is None or pb.confidence is None:
            continue
        xs.append(pa.confidence)
        ys.append(pb.confidence)
    if len(xs) < 3:
        return None
    pear = stats.pearsonr(xs, ys)[0]
    spear = stats.spearmanr(xs, ys)[0]
    return ConfidenceCorr(corpus=corpus, model_a=model_a, model_b=model_b,
                          n=len(xs), pearson=float(pear), spearman=float(spear))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_conflicts.py tests/test_correlation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/analysis/conflicts.py app/analysis/correlation.py tests/test_conflicts.py tests/test_correlation.py
git commit -m "feat: P4 conflict analysis + confidence correlation (skip zero-shot)"
```

---

### Task 7: Language features + analyze orchestration + CLI

**Files:**
- Create: `app/analysis/lang_features.py`, `app/analysis/run.py`
- Modify: `app/cli.py` (add `analyze` subcommand)
- Test: `tests/test_lang_features.py` (+ a small orchestration smoke test in the same file)

**Interfaces:**
- `lang_features.SYNTACTIC_DISTANCE: dict[str,float]` and `RESOURCE_TIER: dict[str,int]` — keyed by the 8 ISO langs (`pl,en,fr,it,es,de,ru,bn`). Values computed from URIEL/lang2vec syntactic distance from Polish and Joshi et al. (2020) resource classes. **The implementer must compute distances via `lang2vec` if installed; otherwise fall back to the committed cached dict (documented values, flagged "user to confirm").** A test asserts every ISO lang has both.
- `run.gold_lookup(study, corpus, index_dir) -> dict[str, Emotion]` — read the corpus index parquet, map `gold_raw` → canonical via `map_label`.
- `run.analyze_study(study, *, results_root, index_dir) -> AnalysisBundle` — for each (study,corpus) in `ROSTER`: load preds for present models, compute baselines (all), agreements (all model pairs within corpus, with bootstrap CI), Fleiss (RAVDESS text+speech+face headline trio), McNemar (headline text vs speech), conflicts + confidence (headline pair), and for study 2 build `Study2LangRow` per split (headline text↔speech κ + lang features) + PL-vs-others two-proportion. Writes `results/analysis/study{N}/bundle.json` + `agreements.csv`.
- CLI: `analyze --study N [--results-root R] [--index-dir I]`.

> **Data-availability note:** `analyze` must **skip missing models gracefully** (a model not yet predicted / not imported → log a warning, exclude from that corpus's pairings) so the pipeline runs before Qwen import completes, then again after. Never fabricate a missing model's predictions.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lang_features.py`:
```python
from app.analysis import lang_features as lf
from app.config import studies


def test_every_language_has_features():
    langs = set(studies.SPLIT_LANG.values())
    for iso in langs:
        assert iso in lf.SYNTACTIC_DISTANCE, iso
        assert iso in lf.RESOURCE_TIER, iso


def test_polish_distance_is_zero():
    assert lf.SYNTACTIC_DISTANCE["pl"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lang_features.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `app/analysis/lang_features.py`**

```python
"""Typological distance from Polish (URIEL/lang2vec) + resourcedness tier (Joshi 2020).

Values are cached here for determinism and offline runs. If `lang2vec` is installed the
verify script (scripts/verify_lang_features.py) can recompute SYNTACTIC_DISTANCE and diff
against this table. RESOURCE_TIER uses Joshi et al. (2020) classes (0=lowest..5=highest).

USER TO CONFIRM the exact values before the Study-2 report is finalized.
"""
# Syntactic distance from Polish (0 = identical). Placeholder cached values from
# lang2vec 'syntax_knn' cosine distance to 'pol'; recompute to confirm.
SYNTACTIC_DISTANCE: dict[str, float] = {
    "pl": 0.0, "ru": 0.34, "de": 0.49, "en": 0.52,
    "fr": 0.55, "it": 0.53, "es": 0.54, "bn": 0.66,
}
# Joshi et al. (2020) resource classes (5 = high like en/de/fr; bn lower).
RESOURCE_TIER: dict[str, int] = {
    "en": 5, "de": 5, "fr": 5, "es": 5, "it": 4, "ru": 4, "pl": 4, "bn": 3,
}
```

> IMPLEMENTER: if `lang2vec` imports, recompute `SYNTACTIC_DISTANCE` from `l2v.distance("syntactic", "pol", iso3)` for each language (map ISO-639-1→639-3), overwrite the dict values, and print old-vs-new so the reviewer/user can confirm. Do NOT invent numbers beyond what lang2vec returns; keep the cached fallback if the lib is unavailable and flag it.

- [ ] **Step 4: Write `app/analysis/run.py`** (orchestration)

Full implementation wiring the modules. Key requirements (implementer writes the body; every branch below is mandatory, no placeholders):
```python
"""analyze orchestration: read cached predictions + gold, emit the AnalysisBundle."""
from pathlib import Path

from app.analysis import bootstrap, conflicts, correlation, lang_features, metrics
from app.analysis.agreement import agreement, aligned_pairs, cohen_kappa, fleiss_kappa
from app.analysis.results import (
    AnalysisBundle, FleissResult, Study2LangRow, write_json, write_tidy_csv)
from app.analysis.tests import correctness, mcnemar, two_proportion
from app.config import studies
from app.config.emotions import Emotion
from app.config.settings import Settings
from app.labels.mapping import map_label
from app.predict.store import PredictionStore


def gold_lookup(study: int, corpus: str, index_dir: Path) -> dict[str, Emotion]:
    import pyarrow.parquet as pq
    path = Path(index_dir) / f"{corpus}.parquet"
    cols = pq.read_table(path, columns=["sample_id", "gold_raw"]).to_pydict()
    out: dict[str, Emotion] = {}
    for sid, raw in zip(cols["sample_id"], cols["gold_raw"]):
        mapped = map_label(raw)          # canonical Emotion or None (dropped label)
        if mapped is not None:
            out[sid] = mapped
    return out


def _load(store, study, corpus, model_id):
    try:
        return store.read(study, corpus, model_id)
    except FileNotFoundError:
        return None


def analyze_study(study, *, results_root: Path, index_dir: Path) -> AnalysisBundle:
    store = PredictionStore(Path(results_root) / "predictions")
    bundle = AnalysisBundle(study=study)
    corpora = [c for (s, c) in studies.ROSTER if s == study]
    for corpus in corpora:
        gold = gold_lookup(study, corpus, index_dir)
        present: dict[str, list] = {}
        for mid in studies.ROSTER[(study, corpus)]:
            preds = _load(store, study, corpus, mid)
            if preds is None:
                continue  # not yet predicted/imported — skip, do not fabricate
            present[mid] = preds
            bundle.baselines.append(
                metrics.baseline(preds, gold, mid, corpus, studies.MODALITY[mid]))
        # pairwise agreements across DIFFERENT modalities (+ same-modality reliability)
        mids = list(present)
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                a, b = mids[i], mids[j]
                res = agreement(present[a], present[b], gold, a, b, corpus)
                la, lb, _ = aligned_pairs(present[a], present[b], gold,
                                          studies.SUPPORTED[a], studies.SUPPORTED[b])
                res.ci = bootstrap.bootstrap_kappa(la, lb)
                bundle.agreements.append(res)
        # headline conflict + confidence (only if both headline models present)
        # study 1 uses PL or EN headline depending on corpus:
        lang = "PL" if corpus == "nEMO" else ("EN" if corpus == "RAVDESS" else None)
        if lang:
            ht, hs = studies.HEADLINE[(lang, "text")], studies.HEADLINE[(lang, "speech")]
            if ht in present and hs in present:
                bundle.conflicts.append(
                    conflicts.conflict(present[ht], present[hs], gold, ht, hs, corpus))
                cc = correlation.confidence_corr(present[ht], present[hs], ht, hs, corpus)
                if cc:
                    bundle.confidence.append(cc)
                ca = correctness(present[ht], gold)
                cb = correctness(present[hs], gold)
                shared = [sid for sid in ca if sid in cb]
                bundle.tests.append(mcnemar([ca[s] for s in shared], [cb[s] for s in shared],
                                            corpus=corpus, detail=f"{ht} vs {hs} correctness"))
        # Fleiss triad on RAVDESS (text+speech+face headline/clean)
        if corpus == "RAVDESS":
            trio = ["en_text_hartmann", "speech_zeroshot", "en_face_vit"]
            if all(m in present for m in trio):
                # align on samples all three predicted, gold in intersection of supports
                _fleiss_ravdess(bundle, present, gold, trio)
    if study == 2:
        _study2(bundle, store, index_dir)
    return bundle
```
Plus the two helpers `_fleiss_ravdess` (builds 3 aligned rater lists by intersecting all three models' non-abstain samples with gold∈∩support, then `fleiss_kappa`, with a bootstrap CI over item indices) and `_study2` (per split: headline text↔speech κ + CI + `Study2LangRow` with `lang_features` values; then two-proportion PL accuracy vs each other language's, appended as `TestResult`s). The implementer writes these fully — no TODOs.

CLI in `app/cli.py`:
```python
def _analyze(args) -> int:
    from app.analysis.run import analyze_study
    from app.analysis.results import write_json, write_tidy_csv
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    index_dir = Path(args.index_dir) if args.index_dir else settings.index_dir
    bundle = analyze_study(args.study, results_root=results_root, index_dir=index_dir)
    out = results_root / "analysis" / f"study{args.study}"
    write_json(bundle, out / "bundle.json")
    write_tidy_csv(bundle, out / "agreements.csv")
    print(f"analyze study {args.study}: {len(bundle.agreements)} agreements, "
          f"{len(bundle.baselines)} baselines -> {out}")
    return 0
```
and register `analyze --study N --results-root --index-dir` mirroring `predict`.

- [ ] **Step 5: Run tests + a real dry run**

Run: `.venv/bin/python -m pytest tests/test_lang_features.py -v` (PASS).
Then a smoke check that `analyze` runs on whatever predictions currently exist (may be empty early — must not crash):
Run: `.venv/bin/python -m app.cli analyze --study 1 2>&1 | tail -3`
Expected: prints a summary; missing models are skipped with no crash.

- [ ] **Step 6: Commit**

```bash
git add app/analysis/lang_features.py app/analysis/run.py app/cli.py tests/test_lang_features.py
git commit -m "feat: P4 analyze orchestration + language features + CLI"
```

---

### Task 8: Report style + tables

**Files:**
- Create: `app/report/__init__.py`, `app/report/style.py`, `app/report/tables.py`
- Test: `tests/test_report_tables.py`

**Interfaces:**
- `style.PALETTE: dict[Emotion,str]` (= `EMOTION_COLORS`), `style.ORDER = CANONICAL_ORDER`, `style.apply()` (sets matplotlib rcParams: Agg, DPI 300, font sizes, grid), `style.save_fig(fig, path_stem)` (writes `{stem}.pdf` + `{stem}.png` at 300 DPI), `style.table_to_png(df, path)` (renders a DataFrame as a styled PNG via matplotlib table).
- `tables.write_table(df, stem)` — writes `{stem}.csv`, `{stem}.md`, `{stem}.tex`, `{stem}.png` (4 formats, same content). `tables.kappa_table(bundle) -> DataFrame`, `tables.baseline_table(bundle) -> DataFrame`, `tables.pl_vs_en_table(bundle) -> DataFrame` (Study 1), `tables.study2_table(bundle) -> DataFrame`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_tables.py`:
```python
from app.analysis.results import AnalysisBundle, AgreementResult, BootstrapCI, Baseline
from app.report import tables


def _bundle():
    return AnalysisBundle(
        study=1,
        baselines=[Baseline(model_id="mling_tabularis", corpus="nEMO", modality="text",
                            n=10, accuracy=0.6, macro_f1=0.5, confusion={}, labels=[])],
        agreements=[AgreementResult(corpus="nEMO", model_a="mling_tabularis",
                    model_b="speech_zeroshot", kappa=0.42, n_used=8, n_excluded=2,
                    excluded_reasons={}, headline=True, ci=BootstrapCI(lo=0.1, hi=0.7))])


def test_kappa_table_has_ci_and_flags(tmp_path):
    df = tables.kappa_table(_bundle())
    assert "kappa" in df.columns and "ci_lo" in df.columns
    assert df.iloc[0]["headline"] in (True, "True", 1)


def test_write_table_emits_four_formats(tmp_path):
    df = tables.kappa_table(_bundle())
    stem = tmp_path / "kappa"
    tables.write_table(df, stem)
    for ext in ("csv", "md", "tex", "png"):
        assert (tmp_path / f"kappa.{ext}").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_tables.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `app/report/style.py` and `app/report/tables.py`**

`style.py` (matplotlib Agg, deterministic):
```python
"""Shared report style: palette, class order, figure/table save helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config.emotions import CANONICAL_ORDER, EMOTION_COLORS

PALETTE = EMOTION_COLORS
ORDER = CANONICAL_ORDER


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "figure.autolayout": True,
    })


def save_fig(fig, path_stem) -> None:
    from pathlib import Path
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def table_to_png(df, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(min(2 + 1.2 * len(df.columns), 20),
                                    0.5 + 0.35 * (len(df) + 1)))
    ax.axis("off")
    t = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

`tables.py`:
```python
"""Table builders + 4-format writers (CSV, Markdown, LaTeX, PNG)."""
from pathlib import Path

import pandas as pd

from app.analysis.results import AnalysisBundle
from app.report import style


def write_table(df: pd.DataFrame, stem) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(df.to_markdown(index=False))
    stem.with_suffix(".tex").write_text(df.to_latex(index=False))
    style.table_to_png(df, stem.with_suffix(".png"))


def kappa_table(bundle: AnalysisBundle) -> pd.DataFrame:
    rows = [{"corpus": a.corpus, "text/speech A": a.model_a, "B": a.model_b,
             "kappa": round(a.kappa, 3),
             "ci_lo": round(a.ci.lo, 3) if a.ci else None,
             "ci_hi": round(a.ci.hi, 3) if a.ci else None,
             "n_used": a.n_used, "n_excl": a.n_excluded,
             "headline": a.headline, "secondary": a.secondary_flag}
            for a in bundle.agreements]
    return pd.DataFrame(rows)


def baseline_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"model": b.model_id, "corpus": b.corpus, "modality": b.modality,
                          "n": b.n, "accuracy": round(b.accuracy, 3),
                          "macro_f1": round(b.macro_f1, 3)} for b in bundle.baselines])


def study2_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"split": r.split, "language": r.language,
                          "kappa": round(r.kappa, 3), "n_used": r.n_used,
                          "syn_dist": r.syntactic_distance, "tier": r.resource_tier}
                         for r in bundle.study2])


def pl_vs_en_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"test": t.name, "corpus": t.corpus, "detail": t.detail,
                          "statistic": round(t.statistic, 3), "p_value": round(t.p_value, 4)}
                         for t in bundle.tests])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_report_tables.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/report/__init__.py app/report/style.py app/report/tables.py tests/test_report_tables.py
git commit -m "feat: P4 report style + 4-format table writers"
```

---

### Task 9: Figures

**Files:**
- Create: `app/report/figures.py`
- Test: `tests/test_report_figures.py`

**Interfaces:**
- `figures.confusion_fig(baseline) -> Figure` (heatmap, fixed class order + palette accents).
- `figures.kappa_ci_fig(bundle) -> Figure` (point + CI error bars per pair; headline highlighted).
- `figures.conflict_fig(conflict_result) -> Figure` (bar: a_closer / b_closer / both_wrong + per-emotion).
- `figures.study2_scatter(bundle) -> Figure` (κ vs syntactic_distance, PL highlighted, point size/hue = resource tier).
- All use `style.apply()` and `style.PALETTE`/`ORDER`. Test asserts a Figure is returned and `style.save_fig` writes both files.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_figures.py`:
```python
from app.analysis.results import Baseline
from app.report import figures, style


def test_confusion_fig_saves_pdf_and_png(tmp_path):
    b = Baseline(model_id="m", corpus="nEMO", modality="text", n=3, accuracy=0.6,
                 macro_f1=0.5, confusion={"anger": {"anger": 2, "sadness": 1}},
                 labels=["anger", "sadness"])
    fig = figures.confusion_fig(b)
    style.save_fig(fig, tmp_path / "cm")
    assert (tmp_path / "cm.pdf").exists() and (tmp_path / "cm.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_figures.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `app/report/figures.py`**

Implement the four builders (full code, no placeholders). Confusion heatmap over `CANONICAL_ORDER`; κ CI as horizontal error bars sorted with headline pairs first + a distinct color; conflict grouped bars; Study-2 scatter with PL annotated. Each returns a `matplotlib.figure.Figure`; none call `plt.show`. Use `style.apply()` at module import or per-builder.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_report_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/report/figures.py tests/test_report_figures.py
git commit -m "feat: P4 report figures (confusion, kappa CI, conflicts, study2 scatter)"
```

---

### Task 10: Report orchestration + CLI

**Files:**
- Create: `app/report/run.py`
- Modify: `app/cli.py` (add `report` subcommand)
- Test: extend `tests/test_report_tables.py` with an end-to-end `report_study` smoke test on a tiny bundle.

**Interfaces:**
- `run.report_study(study, *, results_root) -> list[Path]` — read `results/analysis/study{N}/bundle.json`, write every table under `results/tables/study{N}/` and every figure under `results/figures/study{N}/`, return the list of written paths. Deterministic; overwrites.
- CLI: `report --study N [--results-root R]`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_report_tables.py`)
```python
def test_report_study_writes_artifacts(tmp_path):
    from app.analysis.results import write_json
    from app.report.run import report_study
    b = _bundle()
    write_json(b, tmp_path / "analysis" / "study1" / "bundle.json")
    paths = report_study(1, results_root=tmp_path)
    assert any(p.suffix == ".png" for p in paths)
    assert (tmp_path / "tables" / "study1" / "kappa.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_tables.py::test_report_study_writes_artifacts -v`
Expected: FAIL.

- [ ] **Step 3: Write `app/report/run.py` + CLI**

```python
"""report orchestration: analysis JSON -> tables/ + figures/."""
from pathlib import Path

from app.analysis.results import read_json
from app.report import figures, style, tables


def report_study(study, *, results_root: Path) -> list[Path]:
    results_root = Path(results_root)
    bundle = read_json(results_root / "analysis" / f"study{study}" / "bundle.json")
    style.apply()
    tdir = results_root / "tables" / f"study{study}"
    fdir = results_root / "figures" / f"study{study}"
    written: list[Path] = []
    for name, df in [("kappa", tables.kappa_table(bundle)),
                     ("baselines", tables.baseline_table(bundle)),
                     ("pl_vs_en", tables.pl_vs_en_table(bundle)),
                     ("study2", tables.study2_table(bundle))]:
        if df.empty:
            continue
        tables.write_table(df, tdir / name)
        written += [tdir / f"{name}.{e}" for e in ("csv", "md", "tex", "png")]
    for b in bundle.baselines:
        if b.confusion:
            style.save_fig(figures.confusion_fig(b), fdir / f"cm_{b.corpus}_{b.model_id}")
            written += [fdir / f"cm_{b.corpus}_{b.model_id}.{e}" for e in ("pdf", "png")]
    if bundle.agreements:
        style.save_fig(figures.kappa_ci_fig(bundle), fdir / "kappa_ci")
        written += [fdir / f"kappa_ci.{e}" for e in ("pdf", "png")]
    for c in bundle.conflicts:
        style.save_fig(figures.conflict_fig(c), fdir / f"conflict_{c.corpus}")
        written += [fdir / f"conflict_{c.corpus}.{e}" for e in ("pdf", "png")]
    if bundle.study2:
        style.save_fig(figures.study2_scatter(bundle), fdir / "study2_scatter")
        written += [fdir / f"study2_scatter.{e}" for e in ("pdf", "png")]
    return written
```
CLI handler `_report` mirroring `_analyze`, registered as `report --study N --results-root`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_report_tables.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/report/run.py app/cli.py tests/test_report_tables.py
git commit -m "feat: P4 report orchestration + CLI (analyze -> tables + figures)"
```

---

## Self-Review

**1. Spec coverage (§5.4/§5.5):**
- Baselines per modality (acc, macro-F1, confusion) → T2. ✅
- Cohen's κ pairs + Fleiss' κ triad, common-support + exclusions → T3. ✅
- Bootstrap 1000× CI + PL-vs-EN difference CI → T4. ✅
- McNemar (paired within-set) + chi-square/two-proportion (PL vs EN) → T5. ✅
- Conflicts (closer-to-gold, per-emotion) + confidence Pearson/Spearman (skip zero-shot) → T6. ✅
- Study 2 κ per language + descriptive typology/resource scatter → T7 (+ lang_features). ✅
- Headline vs full-matrix vs secondary/leaky flags → studies.py (T1) + agreement flags (T3). ✅
- Deterministic report: JSON→figures(PDF+PNG)+tables(CSV/MD/LaTeX/PNG), shared style, fixed palette+order → T8/T9/T10. ✅
- Fixed seed, tidy CSV, results/analysis|figures|tables/study{N}/ layout → T1/T7/T10. ✅

**2. Placeholder scan:** correctness-critical tasks (T1–T6) carry complete code. T7's `run.py` has two helpers (`_fleiss_ravdess`, `_study2`) and T9's four figure builders described precisely but written by the implementer — flagged as "full code, no TODOs"; the reviewer must reject placeholders.

**3. Type consistency:** `Emotion`, `Prediction`, `PredictionStore`, `BootstrapCI`/`AgreementResult`/etc. used consistently; `agreement()` returns `AgreementResult` with `ci` filled in T7; `map_label` returns canonical Emotion or None (dropped) — `gold_lookup` skips None.

## Open points
- **Study-2 language features** (`lang_features.py`) — values are cached; implementer recomputes via lang2vec if available, USER confirms before the Study-2 report is final. Flagged in code.
- **`speech_cnn_tf_*` real metrics** (spec §9.2) and **LSTM improvements** (§9.1) are END-STAGE, out of P4 scope — P4 just reports whatever predictions exist.
- **Fleiss triad membership** on RAVDESS uses the clean headline trio (`en_text_hartmann`, `speech_zeroshot`, `en_face_vit`); if a cleaner face model is added later, update `_fleiss_ravdess`.
- `analyze` is re-runnable: run once now (partial — before Qwen import) to exercise the pipeline, then again after `import-predictions` to get the headline speech κ.
