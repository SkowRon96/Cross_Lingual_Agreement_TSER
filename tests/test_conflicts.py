from app.config.emotions import Emotion
from app.models.base import Prediction
from app.analysis.conflicts import conflict

A, S = Emotion.ANGER, Emotion.SADNESS


def _p(sid, pred, mid="m"):
    return Prediction(sample_id=sid, model_id=mid, modality="text", pred_raw="x",
                      pred=pred, confidence=1.0, scores={pred: 1.0})


def test_closer_to_gold_counts():
    gold = {"1": A, "2": A, "3": S}
    pa = [_p("1", A), _p("2", S), _p("3", S)]   # 1: a right; 2: b right; 3: agree (no conflict)
    pb = [_p("1", S), _p("2", A), _p("3", S)]
    r = conflict(pa, pb, gold, "a", "b", "nEMO")
    # conflicts on 1 (a right) and 2 (b right); 3 agree
    assert r.n_conflicts == 2
    assert r.a_closer_to_gold == 1 and r.b_closer_to_gold == 1
    assert r.per_emotion["anger"] == 2
