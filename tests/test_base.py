import math
from app.config.emotions import Emotion
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction


def test_to_prediction_picks_argmax_and_confidence():
    mapper = LabelMapper(["angry", "calm", "neutral"])
    p = to_prediction(
        sample_id="s1", model_id="dummy", modality="speech",
        native_labels=["angry", "calm", "neutral"],
        scores=[0.2, 0.3, 0.5], mapper=mapper,
    )
    assert p.pred is Emotion.NEUTRAL           # 0.3 (calm) + 0.5 (neutral) = 0.8
    assert math.isclose(p.confidence, 0.8, rel_tol=1e-9)
    assert p.abstained is False
    assert p.pred_raw == "neutral"             # native argmax before mapping


def test_to_prediction_abstains_when_all_mass_dropped():
    mapper = LabelMapper(["contempt", "love"])
    p = to_prediction(
        sample_id="s2", model_id="dummy", modality="text",
        native_labels=["contempt", "love"],
        scores=[0.6, 0.4], mapper=mapper,
    )
    assert p.abstained is True
    assert p.pred is None
    assert p.confidence is None
    assert p.scores == {}
    assert p.abstain_reason is not None


def test_prediction_serializes_scores_with_emotion_keys():
    p = Prediction(
        sample_id="s3", model_id="m", modality="text",
        pred_raw="joy", pred=Emotion.HAPPINESS, confidence=0.9,
        scores={Emotion.HAPPINESS: 0.9, Emotion.NEUTRAL: 0.1},
    )
    dumped = p.model_dump(mode="json")
    assert dumped["pred"] == "happiness"
    assert dumped["scores"]["happiness"] == 0.9
