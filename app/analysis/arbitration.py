"""Study 3 — modality arbitration (S0-S3) and the abstention mechanism.

Everything here is computed OFFLINE from the Study-1 prediction cache
(`results/predictions/study1/<corpus>/<model_id>.parquet`). No inference, no model
downloads, no raw media. `Prediction.scores` in that cache is already the HARMONISED
canonical vector (the label mapper maps before the store writes), which is the
precondition for summing vectors in S1/S2 (trap #2 of the concept note); the loader
re-checks it per sample and refuses anything outside the canonical space.

Methodological decisions that are visible in the code on purpose
---------------------------------------------------------------
D-A  Speech channel = `speech_zeroshot` (Qwen2-Audio) in the headline variant. Qwen is a
     GENERATIVE model: it emits a label, not a simplex, so its cached vector carries mass
     1.0 on exactly one class. It therefore enters S1/S2 as an explicitly DEGENERATE
     distribution - a hard vote with full confidence. This is a deliberate choice (Qwen is
     the only speech channel without train-set leakage on nEMO/RAVDESS); it is flagged per
     channel as `degenerate=True` and must be declared in the write-up.
D-B  `speech_cnn_tf_nemo` / `speech_cnn_tf_ravdess` do have genuine soft vectors but are
     LEAKY (trained on nEMO / RAVDESS). They are used only through the `speech_cnn`
     control variant (trap #4), never as the headline result.
D-C  nEMO has TWO channels (text + speech; no face), so there is no majority vote to be
     had - a 2-way disagreement is always a tie. Arbitration on nEMO is therefore resolved
     by CONFIDENCE (S1: summed vectors; S2: weighted summed vectors), which is stated
     explicitly rather than hidden behind a generic voting rule. Majority voting is not
     implemented anywhere in this module for exactly that reason.
D-D  Every strategy on a corpus is evaluated on the SAME sample set: gold known AND all
     channels present AND no channel abstained. Otherwise S3 (oracle) would be measured on
     a different denominator than S0-S2 and the ceiling gate would be meaningless. The
     dropped samples are counted and reported in `excluded_reasons`.
D-E  S2 weights are fitted OUT-OF-FOLD (k-fold, default k=5): the prediction for a sample
     never sees weights fitted on that sample (trap #1). The in-sample (leaky) figure is
     computed too, but only as a diagnostic of how large the optimism would have been -
     it is labelled `s2_insample` and is not the reported result.
D-F  S0 picks the single best channel by accuracy ON THE EVALUATION CORPUS. That is a
     deliberately generous baseline (it is chosen with hindsight), which makes the Delta of
     S1/S2 over S0 conservative.
D-G  Oracle (S3) is a ceiling, not a method. Its accuracy is exactly "some channel was
     right". For macro-F1 it needs a concrete label per sample, so the realisation used is
     gold where some channel is right, and the S0 channel's prediction otherwise.
D-H  Part B uses the BINARY cross-modal conflict flag as the uncertainty signal (concept
     section 7). A binary signal only yields two coverage levels, so to draw a continuous
     risk-coverage curve ties are broken INSIDE each group by the fused confidence of the
     strategy being evaluated (descending). The headline number reported alongside is the
     purely binary one: accuracy on the agreeing subset vs accuracy at full coverage.
D-I  The post-audit text model (concept section 5.1) exists as THREE seeds. Arbitration needs
     one prediction per modality, so each seed is arbitrated as its own variant
     (`audit_s7`/`audit_s17`/`audit_s27`) and reported as mean +- sd over the three; the
     vectors are never averaged into a synthetic fourth model. The seed spread on nEMO is
     itself part of the result. Only the text channel is swapped (speech stays Qwen), so the
     contrast is attributable to the audited model alone (trap #5: identical test split,
     identical taxonomy). `pair_complementarity` measures whether the errors of the audited
     text channel coincide more or less with the speech channel's - the direction is
     measured, never assumed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.analysis import metrics
from app.analysis.bootstrap import SEED
from app.analysis.tests import mcnemar, two_proportion
from app.analysis.results import TestResult
from app.config import studies
from app.config.emotions import CANONICAL_ORDER, Emotion
from app.models.base import Prediction
from app.predict.store import PredictionStore

STUDY = 1  # the prediction cache Study 3 reads from
MODALITY_ORDER: list[str] = ["text", "speech", "face"]
COVERAGE_POINTS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
CURVE_MAX_POINTS = 200

# variant -> corpus -> modality -> model_id
CHANNEL_SETS: dict[str, dict[str, dict[str, str]]] = {
    # Headline: clean, no train-set leakage. Speech = Qwen (degenerate, see D-A).
    "headline": {
        "nEMO": {"text": studies.HEADLINE[("PL", "text")],
                 "speech": studies.HEADLINE[("PL", "speech")]},
        "RAVDESS": {"text": studies.HEADLINE[("EN", "text")],
                    "speech": studies.HEADLINE[("EN", "speech")],
                    "face": "en_face_vit"},
        "MELD": {"text": studies.HEADLINE[("EN", "text")],
                 "speech": studies.HEADLINE[("EN", "speech")],
                 "face": "en_face_vit"},
    },
    # Control (trap #4): genuine soft speech vectors, but LEAKY on nEMO/RAVDESS.
    "speech_cnn": {
        "nEMO": {"text": studies.HEADLINE[("PL", "text")], "speech": "speech_cnn_tf_nemo"},
        "RAVDESS": {"text": studies.HEADLINE[("EN", "text")],
                    "speech": "speech_cnn_tf_ravdess", "face": "en_face_vit"},
        "MELD": {"text": studies.HEADLINE[("EN", "text")],
                 "speech": "speech_cnn_tf_ravdess", "face": "en_face_vit"},
    },
}
LEAKY_VARIANT_CORPORA: dict[str, set[str]] = {"speech_cnn": {"nEMO", "RAVDESS"}}

# Post-audit text model (concept section 5.1): RoBERTa-large trained on the audited
# chatbot_dataset, three seeds. D-I: arbitration needs ONE prediction per modality, so each
# seed is arbitrated SEPARATELY as its own variant and reported as mean +- sd over the three
# (decision D2 of the chapter plan). The score vectors are never averaged into a synthetic
# fourth model - the seed spread on nEMO is large (neutral share 0.600 / 0.644 / 0.900) and
# collapsing it would hide a property of the result.
AUDIT_SEEDS: tuple[int, ...] = (7, 17, 27)
AUDIT_VARIANTS: list[str] = [f"audit_s{s}" for s in AUDIT_SEEDS]
AUDIT_BASELINE_VARIANT = "headline"   # what the post-audit channel is compared against
AUDIT_CORPUS = "nEMO"
for _seed in AUDIT_SEEDS:
    CHANNEL_SETS[f"audit_s{_seed}"] = {
        # only the TEXT channel changes; speech stays Qwen so the contrast is attributable
        # to the audited text model alone (trap #5: identical test split, identical taxonomy)
        AUDIT_CORPUS: {"text": f"pl_text_roberta_audit_s{_seed}",
                       "speech": studies.HEADLINE[("PL", "speech")]},
    }
# Variants that cover the full Study-1 corpus roster (the audit variants are nEMO-only).
FULL_VARIANTS: list[str] = ["headline", "speech_cnn"]


# --------------------------------------------------------------------------- schemas
class ChannelSpec(BaseModel):
    modality: str
    model_id: str
    accuracy: float           # on the common evaluated sample set
    macro_f1: float
    degenerate_share: float   # share of samples whose vector is one-hot (max p >= 0.999)
    degenerate: bool          # True when effectively every vector is one-hot
    leaky: bool


class StrategyResult(BaseModel):
    corpus: str
    variant: str
    strategy: str             # "S0" | "S1" | "S2" | "S3" | "S2_insample"
    label: str
    n: int
    accuracy: float
    macro_f1: float
    delta_accuracy_vs_s0: float
    delta_macro_f1_vs_s0: float
    detail: str = ""


class FoldWeights(BaseModel):
    fold: int
    n_fit: int
    n_fit_conflicts: int
    n_eval: int
    weights: dict[str, float]


class CurvePoint(BaseModel):
    coverage: float
    risk: float
    n: int


class SelectiveResult(BaseModel):
    corpus: str
    variant: str
    strategy: str
    uncertainty: str = "binary_cross_modal_conflict (ties broken by fused confidence)"
    n: int
    accuracy_full_coverage: float
    accuracy_agree_subset: float
    accuracy_conflict_subset: float
    n_agree: int
    n_conflict: int
    selective_accuracy: dict[str, float]   # "1.00"/"0.90"/"0.80"/"0.70" -> accuracy
    aurc: float
    curve: list[CurvePoint]


class ComplementarityResult(BaseModel):
    """How two channels' errors relate - the quantity section 5.1 of the concept asks about.

    A channel can be better in isolation yet contribute LESS to fusion if its errors start
    to coincide with the other channel's. `error_overlap_ratio` is the observed
    both-wrong rate divided by the rate expected if the two error events were independent:
    1.0 = independent errors, >1 = errors coincide (redundant channels, less to gain from
    fusion), <1 = errors are complementary.
    """
    corpus: str
    variant: str
    modality_a: str
    modality_b: str
    model_a: str
    model_b: str
    n: int
    kappa_predictions: float      # Cohen's kappa between the two channels' predicted labels
    accuracy_a: float
    accuracy_b: float
    both_correct: int
    a_only_correct: int
    b_only_correct: int
    both_wrong: int
    pair_oracle_accuracy: float   # 1 - both_wrong/n
    expected_both_wrong_if_independent: float
    error_overlap_ratio: float


class MetricDelta(BaseModel):
    corpus: str
    variant: str
    baseline_variant: str
    metric: str
    value: float
    baseline_value: float
    delta: float


class SeedAggregate(BaseModel):
    """Mean +- sd over the per-seed variants (D-I / plan decision D2)."""
    group: str
    corpus: str
    metric: str
    n_seeds: int
    mean: float
    sd: float                     # sample sd (ddof=1)
    per_variant: dict[str, float]
    baseline_variant: str | None = None
    baseline_value: float | None = None
    mean_delta: float | None = None
    sd_delta: float | None = None


class ArbitrationCorpus(BaseModel):
    corpus: str
    variant: str
    leaky: bool
    n_used: int
    n_excluded: int
    excluded_reasons: dict[str, int]
    channels: list[ChannelSpec]
    s0_modality: str
    s0_model_id: str
    n_folds: int
    seed: int
    fold_weights: list[FoldWeights]
    mean_weights: dict[str, float]
    strategies: list[StrategyResult]
    selective: list[SelectiveResult]
    complementarity: list[ComplementarityResult] = []
    tests: list[TestResult]
    notes: list[str] = []


class ArbitrationBundle(BaseModel):
    study: int = 3
    source_study: int = STUDY
    corpora: list[ArbitrationCorpus] = []
    audit_deltas: list[MetricDelta] = []
    seed_aggregates: list[SeedAggregate] = []
    notes: list[str] = []


# ----------------------------------------------------------------------- per-sample table
@dataclass(frozen=True)
class SampleRow:
    sample_id: str
    gold: Emotion
    preds: dict[str, Emotion]                  # modality -> predicted label
    scores: dict[str, dict[Emotion, float]]    # modality -> harmonised vector


def _check_harmonised(p: Prediction, model_id: str) -> None:
    """Refuse anything that is not already in the canonical space (trap #2)."""
    for k in p.scores:
        if not isinstance(k, Emotion):
            raise ValueError(f"{model_id}: non-canonical score key {k!r} - vector not harmonised")
    total = sum(p.scores.values())
    if p.scores and not (0.99 <= total <= 1.01):
        raise ValueError(f"{model_id}/{p.sample_id}: score vector sums to {total!r}, expected 1")


def build_table(store: PredictionStore, corpus: str, channels: dict[str, str],
                gold: dict[str, Emotion], *, study: int = STUDY,
                ) -> tuple[list[SampleRow], dict[str, int]]:
    """Per-sample table: sample_id | gold | pred+full vector per modality.

    Keeps only samples with gold and a usable prediction from EVERY channel (D-D).
    """
    loaded: dict[str, dict[str, Prediction]] = {}
    for modality, model_id in channels.items():
        path = store.path(study, corpus, model_id)
        if not path.exists():
            raise FileNotFoundError(f"no cached predictions: {path}")
        preds = store.read(study, corpus, model_id)
        for p in preds:
            _check_harmonised(p, model_id)
        loaded[modality] = {p.sample_id: p for p in preds}

    rows: list[SampleRow] = []
    excluded: dict[str, int] = {}
    for sid in sorted(gold):
        reason = None
        for modality in sorted(channels):
            p = loaded[modality].get(sid)
            if p is None:
                reason = f"missing_prediction:{modality}"
                break
            if p.pred is None or p.abstained or not p.scores:
                reason = f"abstained:{modality}"
                break
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        rows.append(SampleRow(
            sample_id=sid, gold=gold[sid],
            preds={m: loaded[m][sid].pred for m in channels},
            scores={m: dict(loaded[m][sid].scores) for m in channels},
        ))
    return rows, excluded


# ------------------------------------------------------------------------------ helpers
def argmax_vector(vec: dict[Emotion, float]) -> Emotion | None:
    """Deterministic argmax: ties broken by CANONICAL_ORDER, never by dict order."""
    best: Emotion | None = None
    best_v = float("-inf")
    for e in CANONICAL_ORDER:
        v = vec.get(e, 0.0)
        if v > best_v:
            best, best_v = e, v
    return best if best_v > float("-inf") else None


def fuse(row: SampleRow, weights: dict[str, float]) -> dict[Emotion, float]:
    """Weighted sum of the harmonised vectors (S1 = uniform weights, S2 = fitted)."""
    out: dict[Emotion, float] = {e: 0.0 for e in CANONICAL_ORDER}
    for modality, vec in row.scores.items():
        w = weights.get(modality, 0.0)
        if w == 0.0:
            continue
        for e, v in vec.items():
            out[e] += w * v
    return out


def is_conflict(row: SampleRow) -> bool:
    """Binary cross-modal conflict flag: the channels do not all agree."""
    return len(set(row.preds.values())) > 1


def _accuracy(preds: list[Emotion | None], rows: list[SampleRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for p, r in zip(preds, rows) if p == r.gold) / len(rows)


def fit_weights(rows: list[SampleRow], modalities: list[str]) -> tuple[dict[str, float], int]:
    """Channel weights from the Study-1 conflict analysis.

    A channel's weight is its accuracy ON THE CONFLICT SUBSET of the fitting rows - i.e.
    how reliable it is exactly when the modalities disagree, which is the quantity
    `conflicts.conflict()` reports pairwise as `*_closer_to_gold`. Weights are normalised
    to sum to 1 (a monotone rescaling; it does not change any argmax, only readability).
    Samples where every channel agrees carry no arbitration information, so they are not
    used for fitting.
    """
    conflict_rows = [r for r in rows if is_conflict(r)]
    fit_rows = conflict_rows if conflict_rows else rows
    raw = {m: (sum(1 for r in fit_rows if r.preds[m] == r.gold) / len(fit_rows)
               if fit_rows else 0.0)
           for m in modalities}
    total = sum(raw.values())
    if total <= 0:  # degenerate corner: nothing is ever right on conflicts -> uniform
        n = len(modalities)
        return {m: 1.0 / n for m in modalities}, len(conflict_rows)
    return {m: v / total for m, v in raw.items()}, len(conflict_rows)


def fold_assignment(n: int, n_folds: int, seed: int = SEED) -> np.ndarray:
    """Deterministic fold ids for rows already sorted by sample_id."""
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.empty(n, dtype=int)
    for position, i in enumerate(idx):
        folds[i] = position % n_folds
    return folds


# ------------------------------------------------------------------------- risk-coverage
def risk_coverage(order: list[int], correct: list[bool]) -> tuple[list[tuple[float, float, int]], float]:
    """Risk-coverage curve and AURC for an accept-order over samples.

    `order` lists sample positions from most certain to least certain. Point k covers the
    first k accepted samples; risk is the error rate among them. AURC is the mean risk
    over all coverage levels k = 1..N (the discrete area under the curve).
    """
    n = len(order)
    if n == 0:
        return [], 0.0
    curve: list[tuple[float, float, int]] = []
    errors = 0
    risks = np.empty(n)
    for k, i in enumerate(order, start=1):
        if not correct[i]:
            errors += 1
        risk = errors / k
        risks[k - 1] = risk
        curve.append((k / n, risk, k))
    return curve, float(risks.mean())


def selective_accuracy(curve: list[tuple[float, float, int]], coverage: float) -> float:
    n = curve[-1][2] if curve else 0
    if n == 0:
        return 0.0
    k = max(1, min(n, int(round(coverage * n))))
    return 1.0 - curve[k - 1][1]


def _thin_curve(curve: list[tuple[float, float, int]]) -> list[CurvePoint]:
    n = len(curve)
    if n <= CURVE_MAX_POINTS:
        keep = range(n)
    else:
        keep = sorted({int(round(x)) for x in np.linspace(0, n - 1, CURVE_MAX_POINTS)})
    return [CurvePoint(coverage=curve[i][0], risk=curve[i][1], n=curve[i][2]) for i in keep]


# ------------------------------------------------------------------------------ metrics
def _score(preds: list[Emotion | None], rows: list[SampleRow], corpus: str,
           name: str) -> tuple[float, float]:
    """accuracy + macro-F1 through the same code path as the Study-1 baselines."""
    gold = {r.sample_id: r.gold for r in rows}
    objs = [Prediction(sample_id=r.sample_id, model_id=name, modality="fusion",
                       pred_raw=None, pred=p, confidence=None, scores={})
            for p, r in zip(preds, rows)]
    b = metrics.baseline(objs, gold, name, corpus, "fusion")
    return b.accuracy, b.macro_f1


# --------------------------------------------------------------------------- main driver
def arbitrate_corpus(store: PredictionStore, corpus: str, gold: dict[str, Emotion],
                     *, variant: str = "headline", channels: dict[str, str] | None = None,
                     n_folds: int = 5, seed: int = SEED,
                     study: int = STUDY) -> ArbitrationCorpus:
    channels = channels if channels is not None else CHANNEL_SETS[variant][corpus]
    modalities = [m for m in MODALITY_ORDER if m in channels]
    rows, excluded = build_table(store, corpus, channels, gold, study=study)
    if not rows:
        raise ValueError(f"{corpus}/{variant}: no usable samples")
    n = len(rows)
    notes: list[str] = []

    # ---- channel descriptions (incl. the degenerate-distribution flag, D-A)
    specs: list[ChannelSpec] = []
    for m in modalities:
        preds = [r.preds[m] for r in rows]
        acc, f1 = _score(preds, rows, corpus, channels[m])
        deg = sum(1 for r in rows if max(r.scores[m].values()) >= 0.999) / n
        specs.append(ChannelSpec(
            modality=m, model_id=channels[m], accuracy=acc, macro_f1=f1,
            degenerate_share=deg, degenerate=deg >= 0.999,
            leaky=channels[m] in studies.LEAKY))
    for s in specs:
        if s.degenerate:
            notes.append(
                f"channel {s.modality} ({s.model_id}) is a DEGENERATE distribution: "
                f"{s.degenerate_share:.4f} of its vectors are one-hot - it enters S1/S2 as a "
                f"hard vote with full confidence (decision D-A).")
    if len(modalities) == 2:
        notes.append(
            "two channels only: a disagreement is always a tie, so there is no majority "
            "vote; arbitration is resolved by confidence (S1) and by conflict-fitted "
            "channel weights (S2). Decision D-C.")

    # ---- S0: best single channel by accuracy on this corpus (D-F)
    best = max(specs, key=lambda s: (s.accuracy, -MODALITY_ORDER.index(s.modality)))
    s0_preds = [r.preds[best.modality] for r in rows]

    # ---- S1: argmax of the plain sum of harmonised vectors
    uniform = {m: 1.0 for m in modalities}
    s1_fused = [fuse(r, uniform) for r in rows]
    s1_preds = [argmax_vector(v) for v in s1_fused]

    # ---- S2: conflict-fitted channel weights, evaluated OUT-OF-FOLD (D-E / trap #1)
    folds = fold_assignment(n, n_folds, seed)
    s2_preds: list[Emotion | None] = [None] * n
    s2_fused: list[dict[Emotion, float]] = [{} for _ in range(n)]
    fold_weights: list[FoldWeights] = []
    for f in range(n_folds):
        fit_rows = [r for r, g in zip(rows, folds) if g != f]
        eval_idx = [i for i, g in enumerate(folds) if g == f]
        w, n_conf = fit_weights(fit_rows, modalities)
        fold_weights.append(FoldWeights(fold=f, n_fit=len(fit_rows), n_fit_conflicts=n_conf,
                                        n_eval=len(eval_idx), weights=w))
        for i in eval_idx:
            v = fuse(rows[i], w)
            s2_fused[i] = v
            s2_preds[i] = argmax_vector(v)
    mean_weights = {m: float(np.mean([fw.weights[m] for fw in fold_weights])) for m in modalities}

    # in-sample S2 (LEAKY): only a diagnostic of how large the optimism would have been
    w_all, _ = fit_weights(rows, modalities)
    s2_in_preds = [argmax_vector(fuse(r, w_all)) for r in rows]

    # ---- S3: oracle ceiling (D-G)
    oracle_hit = [any(r.preds[m] == r.gold for m in modalities) for r in rows]
    s3_preds = [r.gold if hit else p0 for hit, r, p0 in zip(oracle_hit, rows, s0_preds)]

    acc0, f10 = _score(s0_preds, rows, corpus, "S0")
    results: list[StrategyResult] = []
    for key, label, preds, detail in (
        ("S0", "best single modality", s0_preds,
         f"channel {best.modality} ({best.model_id}) chosen by accuracy on this corpus"),
        ("S1", "naive fusion (argmax of summed vectors)", s1_preds,
         "unweighted sum of the harmonised score vectors"),
        ("S2", "modality priority (conflict-fitted weights, out-of-fold)", s2_preds,
         f"{n_folds}-fold: weights fitted on the other folds, never on the evaluated sample"),
        ("S3", "oracle (any modality correct)", s3_preds,
         "ceiling for late fusion; not implementable - requires the label"),
        ("S2_insample", "S2 with weights fitted on the full set (LEAKY diagnostic)",
         s2_in_preds, "reported only to quantify the optimism avoided by D-E; not a result"),
    ):
        acc, f1 = _score(preds, rows, corpus, key)
        results.append(StrategyResult(
            corpus=corpus, variant=variant, strategy=key, label=label, n=n,
            accuracy=acc, macro_f1=f1, delta_accuracy_vs_s0=acc - acc0,
            delta_macro_f1_vs_s0=f1 - f10, detail=detail))

    # ---- gate: the oracle is the ceiling of late fusion
    by_key = {r.strategy: r for r in results}
    for key in ("S0", "S1", "S2"):
        if by_key["S3"].accuracy + 1e-12 < by_key[key].accuracy:
            raise AssertionError(
                f"{corpus}/{variant}: oracle ceiling violated - S3={by_key['S3'].accuracy!r} "
                f"< {key}={by_key[key].accuracy!r}. This is an implementation bug, not a result.")

    # ---- Part B: abstention on the binary cross-modal conflict flag (D-H)
    conflict = [is_conflict(r) for r in rows]
    confidences = {
        "S0": [max(r.scores[best.modality].values()) for r in rows],
        "S1": [max(v.values()) / len(modalities) for v in s1_fused],
        "S2": [max(v.values()) if v else 0.0 for v in s2_fused],
    }
    if best.degenerate:
        notes.append(
            f"the S0 channel ({best.model_id}) is degenerate, so its confidence is a "
            f"constant 1.0: inside the agree/conflict groups the S0 risk-coverage order is "
            f"arbitrary (sample_id) and its AURC must not be read as a ranking quality. "
            f"Only the fused (S1/S2) confidences carry a usable ordering here.")
    selective: list[SelectiveResult] = []
    for key, preds in (("S0", s0_preds), ("S1", s1_preds), ("S2", s2_preds)):
        correct = [p == r.gold for p, r in zip(preds, rows)]
        conf = confidences[key]
        # accept order: agreeing samples first, then by descending fused confidence,
        # sample_id last so the order is fully deterministic.
        order = sorted(range(n), key=lambda i: (conflict[i], -conf[i], rows[i].sample_id))
        curve, aurc = risk_coverage(order, correct)
        agree_idx = [i for i in range(n) if not conflict[i]]
        conf_idx = [i for i in range(n) if conflict[i]]
        selective.append(SelectiveResult(
            corpus=corpus, variant=variant, strategy=key, n=n,
            accuracy_full_coverage=sum(correct) / n,
            accuracy_agree_subset=(sum(correct[i] for i in agree_idx) / len(agree_idx)
                                   if agree_idx else 0.0),
            accuracy_conflict_subset=(sum(correct[i] for i in conf_idx) / len(conf_idx)
                                      if conf_idx else 0.0),
            n_agree=len(agree_idx), n_conflict=len(conf_idx),
            selective_accuracy={f"{c:.2f}": selective_accuracy(curve, c)
                                for c in COVERAGE_POINTS},
            aurc=aurc, curve=_thin_curve(curve)))

    # ---- complementarity between every channel pair (concept section 5.1)
    complementarity: list[ComplementarityResult] = []
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            complementarity.append(
                pair_complementarity(rows, modalities[i], modalities[j], channels,
                                     corpus=corpus, variant=variant))

    # ---- tests
    tests: list[TestResult] = []
    c_s2 = [p == r.gold for p, r in zip(s2_preds, rows)]
    c_s1 = [p == r.gold for p, r in zip(s1_preds, rows)]
    c_s0 = [p == r.gold for p, r in zip(s0_preds, rows)]
    tests.append(mcnemar(c_s2, c_s0, corpus=corpus,
                         detail=f"[{variant}] S2 (modality priority) vs S0 (best single modality)"))
    tests.append(mcnemar(c_s1, c_s0, corpus=corpus,
                         detail=f"[{variant}] S1 (naive fusion) vs S0 (best single modality)"))
    sel_s2 = next(s for s in selective if s.strategy == "S2")
    if sel_s2.n_agree and sel_s2.n_conflict:
        tests.append(two_proportion(
            int(round(sel_s2.accuracy_agree_subset * sel_s2.n_agree)), sel_s2.n_agree,
            int(round(sel_s2.accuracy_conflict_subset * sel_s2.n_conflict)), sel_s2.n_conflict,
            corpus=corpus,
            detail=f"[{variant}] S2 accuracy: cross-modal agreement vs conflict subset"))

    leaky = corpus in LEAKY_VARIANT_CORPORA.get(variant, set())
    if leaky:
        notes.append(f"variant {variant!r} on {corpus} uses a LEAKY speech model "
                     f"({channels['speech']}, trained on this corpus) - control only, trap #4.")
    return ArbitrationCorpus(
        corpus=corpus, variant=variant, leaky=leaky, n_used=n,
        n_excluded=sum(excluded.values()), excluded_reasons=excluded, channels=specs,
        s0_modality=best.modality, s0_model_id=best.model_id, n_folds=n_folds, seed=seed,
        fold_weights=fold_weights, mean_weights=mean_weights, strategies=results,
        selective=selective, complementarity=complementarity, tests=tests, notes=notes)


# ------------------------------------------------------ complementarity / seed aggregation
def pair_complementarity(rows: list[SampleRow], mod_a: str, mod_b: str,
                         channels: dict[str, str], *, corpus: str,
                         variant: str) -> ComplementarityResult:
    from app.analysis.agreement import cohen_kappa
    n = len(rows)
    ok_a = [r.preds[mod_a] == r.gold for r in rows]
    ok_b = [r.preds[mod_b] == r.gold for r in rows]
    both = sum(1 for a, b in zip(ok_a, ok_b) if a and b)
    a_only = sum(1 for a, b in zip(ok_a, ok_b) if a and not b)
    b_only = sum(1 for a, b in zip(ok_a, ok_b) if b and not a)
    neither = n - both - a_only - b_only
    err_a, err_b = 1 - sum(ok_a) / n, 1 - sum(ok_b) / n
    expected = err_a * err_b
    observed = neither / n
    return ComplementarityResult(
        corpus=corpus, variant=variant, modality_a=mod_a, modality_b=mod_b,
        model_a=channels[mod_a], model_b=channels[mod_b], n=n,
        kappa_predictions=cohen_kappa([r.preds[mod_a] for r in rows],
                                      [r.preds[mod_b] for r in rows]),
        accuracy_a=sum(ok_a) / n, accuracy_b=sum(ok_b) / n,
        both_correct=both, a_only_correct=a_only, b_only_correct=b_only, both_wrong=neither,
        pair_oracle_accuracy=1.0 - observed,
        expected_both_wrong_if_independent=expected,
        error_overlap_ratio=(observed / expected if expected > 0 else 0.0),
    )


def corpus_metrics(block: ArbitrationCorpus) -> dict[str, float]:
    """Flat metric view of one corpus x variant block, for deltas and seed aggregation."""
    out: dict[str, float] = {}
    for ch in block.channels:
        out[f"channel_{ch.modality}_accuracy"] = ch.accuracy
        out[f"channel_{ch.modality}_macro_f1"] = ch.macro_f1
    for s in block.strategies:
        out[f"{s.strategy}_accuracy"] = s.accuracy
        out[f"{s.strategy}_macro_f1"] = s.macro_f1
    if "S3_accuracy" in out and "S2_accuracy" in out:
        out["oracle_gap_S3_minus_S2"] = out["S3_accuracy"] - out["S2_accuracy"]
        out["oracle_gap_S3_minus_S0"] = out["S3_accuracy"] - out["S0_accuracy"]
    for c in block.complementarity:
        tag = f"{c.modality_a}_{c.modality_b}"
        out[f"kappa_{tag}"] = c.kappa_predictions
        out[f"pair_oracle_{tag}"] = c.pair_oracle_accuracy
        out[f"both_wrong_share_{tag}"] = c.both_wrong / c.n
        out[f"error_overlap_ratio_{tag}"] = c.error_overlap_ratio
        out[f"{c.modality_a}_only_correct_share_{tag}"] = c.a_only_correct / c.n
        out[f"{c.modality_b}_only_correct_share_{tag}"] = c.b_only_correct / c.n
    for s in block.selective:
        out[f"{s.strategy}_aurc"] = s.aurc
        out[f"{s.strategy}_accuracy_agree_subset"] = s.accuracy_agree_subset
        out[f"{s.strategy}_accuracy_conflict_subset"] = s.accuracy_conflict_subset
        out[f"{s.strategy}_n_conflict_share"] = s.n_conflict / s.n
    return out


def _block(bundle: ArbitrationBundle, variant: str, corpus: str) -> ArbitrationCorpus | None:
    for b in bundle.corpora:
        if b.variant == variant and b.corpus == corpus:
            return b
    return None


def audit_comparison(bundle: ArbitrationBundle, *, corpus: str = AUDIT_CORPUS,
                     variants: list[str] | None = None,
                     baseline_variant: str = AUDIT_BASELINE_VARIANT,
                     group: str = "audit",
                     ) -> tuple[list[MetricDelta], list[SeedAggregate]]:
    """Per-seed deltas vs the baseline text channel + mean/sd over seeds (D-I).

    Answers section 5.1 numerically: does a better-audited text model move only its own
    accuracy, or also the cross-modal agreement, the complementarity and the arbitration
    result? Direction is measured, never assumed.
    """
    variants = variants if variants is not None else AUDIT_VARIANTS
    base = _block(bundle, baseline_variant, corpus)
    present = [(v, _block(bundle, v, corpus)) for v in variants]
    present = [(v, b) for v, b in present if b is not None]
    if base is None or not present:
        return [], []
    base_m = corpus_metrics(base)
    deltas: list[MetricDelta] = []
    per_metric: dict[str, dict[str, float]] = {}
    for variant, block in present:
        m = corpus_metrics(block)
        for key in sorted(m):
            per_metric.setdefault(key, {})[variant] = m[key]
            if key in base_m:
                deltas.append(MetricDelta(
                    corpus=corpus, variant=variant, baseline_variant=baseline_variant,
                    metric=key, value=m[key], baseline_value=base_m[key],
                    delta=m[key] - base_m[key]))
    aggregates: list[SeedAggregate] = []
    for key in sorted(per_metric):
        vals = [per_metric[key][v] for v, _ in present]
        arr = np.asarray(vals, dtype=float)
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        agg = SeedAggregate(
            group=group, corpus=corpus, metric=key, n_seeds=len(vals),
            mean=float(arr.mean()), sd=sd, per_variant=per_metric[key])
        if key in base_m:
            d = arr - base_m[key]
            agg.baseline_variant = baseline_variant
            agg.baseline_value = base_m[key]
            agg.mean_delta = float(d.mean())
            agg.sd_delta = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        aggregates.append(agg)
    return deltas, aggregates


def run_study3(*, results_root: Path, index_dir: Path,
               variants: list[str] | None = None,
               corpora: list[str] | None = None,
               n_folds: int = 5, seed: int = SEED) -> ArbitrationBundle:
    from app.analysis.run import gold_lookup
    store = PredictionStore(Path(results_root) / "predictions")
    variants = variants if variants is not None else list(CHANNEL_SETS)
    bundle = ArbitrationBundle(notes=[
        "Study 3 is computed offline from the Study-1 prediction cache; no inference.",
        "Score vectors are read back from the cache already harmonised (verified per "
        "sample by build_table) - vectors are summed in the canonical space (trap #2).",
        "S2 weights are fitted out-of-fold; the in-sample figure is kept only as the "
        "'S2_insample' leakage diagnostic (trap #1).",
        "Oracle (S3) is the late-fusion ceiling, not a method (concept section 6); its value "
        "grows with the number of channels, so nEMO (2 channels) and RAVDESS/MELD "
        "(3 channels) oracles are not comparable across corpora (risk R7).",
        "The post-audit text model enters as three separate per-seed variants "
        "(audit_s7/s17/s27), arbitrated independently and reported as mean +- sd; score "
        "vectors are never averaged into a synthetic single model (decision D-I).",
    ])
    gold_cache: dict[str, dict[str, Emotion]] = {}
    for variant in variants:
        want = corpora if corpora is not None else list(CHANNEL_SETS[variant])
        for corpus in want:
            if corpus not in CHANNEL_SETS[variant]:
                continue  # e.g. the audit variants only define nEMO
            if corpus not in gold_cache:
                gold_cache[corpus] = gold_lookup(STUDY, corpus, Path(index_dir))
            bundle.corpora.append(arbitrate_corpus(
                store, corpus, gold_cache[corpus], variant=variant,
                n_folds=n_folds, seed=seed))
    bundle.audit_deltas, bundle.seed_aggregates = audit_comparison(bundle)
    return bundle


def write_json(bundle: ArbitrationBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.model_dump(), indent=2, sort_keys=True))


def read_json(path: Path) -> ArbitrationBundle:
    return ArbitrationBundle.model_validate_json(Path(path).read_text())


def write_tidy_md(bundle: ArbitrationBundle, path: Path) -> None:
    """One tidy Markdown table of strategy rows, for quick inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "corpus", "strategy", "n", "accuracy", "macro_f1",
            "d_acc_vs_s0", "d_f1_vs_s0", "detail"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for c in bundle.corpora:
        for s in c.strategies:
            lines.append("| " + " | ".join(str(v) for v in [
                s.variant, s.corpus, s.strategy, s.n, round(s.accuracy, 4),
                round(s.macro_f1, 4), round(s.delta_accuracy_vs_s0, 4),
                round(s.delta_macro_f1_vs_s0, 4), s.detail]) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    import argparse

    from app.config.settings import Settings
    parser = argparse.ArgumentParser(prog="app.analysis.arbitration",
                                     description="Study 3: modality arbitration + abstention")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--variant", action="append", default=None,
                        choices=sorted(CHANNEL_SETS))
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    index_dir = Path(args.index_dir) if args.index_dir else settings.index_dir
    bundle = run_study3(results_root=results_root, index_dir=index_dir,
                        variants=args.variant, corpora=args.corpus, n_folds=args.folds)
    out = Path(results_root) / "analysis" / "study3"
    write_json(bundle, out / "bundle.json")
    write_tidy_md(bundle, out / "strategies.md")
    print(f"study 3: {len(bundle.corpora)} corpus x variant blocks -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
