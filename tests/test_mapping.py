import pytest
from app.config.emotions import Emotion
from app.labels.mapping import map_label, MASTER_MAP, DROP_LABELS


def test_direct_ekman_labels():
    assert map_label("anger") is Emotion.ANGER
    assert map_label("disgust") is Emotion.DISGUST
    assert map_label("fear") is Emotion.FEAR
    assert map_label("sadness") is Emotion.SADNESS
    assert map_label("neutral") is Emotion.NEUTRAL


def test_synonyms_normalize_to_canonical():
    assert map_label("joy") is Emotion.HAPPINESS
    assert map_label("happy") is Emotion.HAPPINESS
    assert map_label("happiness") is Emotion.HAPPINESS
    assert map_label("surprised") is Emotion.SURPRISE
    assert map_label("angry") is Emotion.ANGER
    assert map_label("sad") is Emotion.SADNESS


def test_established_merges():
    assert map_label("calm") is Emotion.NEUTRAL
    assert map_label("excitement") is Emotion.HAPPINESS
    assert map_label("enthusiasm") is Emotion.HAPPINESS


def test_case_and_whitespace_insensitive():
    assert map_label("  Anger ") is Emotion.ANGER
    assert map_label("HAPPY") is Emotion.HAPPINESS


def test_dropped_labels_return_none():
    for lbl in ["hope", "pride", "contempt", "frustration", "gratitude",
                "love", "sarcasm", "anxiety", "apology", "assertiveness",
                "concern", "encouragement", "poker"]:
        assert map_label(lbl) is None


def test_none_of_them_dropped_by_default_but_overridable():
    assert map_label("none of them") is None
    assert map_label("none of them", overrides={"none of them": Emotion.NEUTRAL}) is Emotion.NEUTRAL


def test_override_can_drop_a_normally_mapped_label():
    assert map_label("none of them", overrides={"none of them": None}) is None


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        map_label("banana")


def test_override_beats_master_map():
    # override must win even over a label that MASTER_MAP maps normally
    assert map_label("anger", overrides={"anger": Emotion.NEUTRAL}) is Emotion.NEUTRAL
    assert map_label("joy", overrides={"joy": None}) is None


def test_master_map_and_drop_labels_are_disjoint():
    assert set(MASTER_MAP).isdisjoint(DROP_LABELS)


def test_ravdess_fearful_maps_to_fear():
    from app.config.emotions import Emotion
    assert map_label("fearful") is Emotion.FEAR
    assert map_label("FEARFUL") is Emotion.FEAR  # normalization
