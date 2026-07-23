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
