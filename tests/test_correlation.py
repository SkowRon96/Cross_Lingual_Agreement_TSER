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
