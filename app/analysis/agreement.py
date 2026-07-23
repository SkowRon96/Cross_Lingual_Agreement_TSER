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
