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
