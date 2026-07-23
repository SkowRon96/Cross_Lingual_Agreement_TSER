import math
import pytest
from app.config.emotions import Emotion
from app.labels.mapper import LabelMapper


def approx_sum_one(d):
    return math.isclose(sum(d.values()), 1.0, rel_tol=1e-9)


def test_supported_reflects_mapped_labels_only():
    m = LabelMapper(["angry", "happy", "sad", "neutral"])  # IEMOCAP-style
    assert m.supported == {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}


def test_map_label_passthrough():
    m = LabelMapper(["calm", "neutral"])
    assert m.map_label("calm") is Emotion.NEUTRAL


def test_scores_aggregate_when_two_natives_map_to_one_emotion():
    # RAVDESS-style: calm and neutral both -> NEUTRAL, summed.
    labels = ["angry", "calm", "neutral"]
    m = LabelMapper(labels)
    out = m.map_scores([0.2, 0.3, 0.5])
    assert approx_sum_one(out)
    assert math.isclose(out[Emotion.NEUTRAL], 0.8, rel_tol=1e-9)
    assert math.isclose(out[Emotion.ANGER], 0.2, rel_tol=1e-9)


def test_dropped_mass_is_removed_and_renormalized():
    # contempt is dropped; remaining mass renormalizes to sum 1.
    labels = ["anger", "contempt", "neutral"]
    m = LabelMapper(labels)
    out = m.map_scores([0.5, 0.3, 0.2])
    assert set(out) == {Emotion.ANGER, Emotion.NEUTRAL}
    assert approx_sum_one(out)
    # 0.5 and 0.2 renormalized over 0.7 total.
    assert math.isclose(out[Emotion.ANGER], 0.5 / 0.7, rel_tol=1e-9)
    assert math.isclose(out[Emotion.NEUTRAL], 0.2 / 0.7, rel_tol=1e-9)


def test_all_mass_dropped_returns_empty():
    m = LabelMapper(["contempt", "love"])
    assert m.map_scores([0.6, 0.4]) == {}


def test_none_of_them_override_to_neutral():
    # visegradmedia-style: no explicit neutral; none-of-them -> NEUTRAL.
    labels = ["anger", "fear", "disgust", "sadness", "happiness", "none of them"]
    m = LabelMapper(labels, overrides={"none of them": Emotion.NEUTRAL})
    assert Emotion.NEUTRAL in m.supported
    out = m.map_scores([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    assert math.isclose(out[Emotion.NEUTRAL], 0.5, rel_tol=1e-9)


def test_length_mismatch_raises():
    m = LabelMapper(["anger", "neutral"])
    with pytest.raises(ValueError):
        m.map_scores([0.5, 0.3, 0.2])
