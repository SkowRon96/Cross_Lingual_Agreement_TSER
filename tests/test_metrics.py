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
