from app.config.emotions import Emotion, CANONICAL_ORDER, EMOTION_COLORS


def test_seven_canonical_emotions():
    assert [e.value for e in CANONICAL_ORDER] == [
        "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
    ]


def test_emotion_is_str_enum():
    assert Emotion.HAPPINESS == "happiness"
    assert Emotion("surprise") is Emotion.SURPRISE


def test_every_emotion_has_a_distinct_color():
    assert set(EMOTION_COLORS) == set(CANONICAL_ORDER)
    assert len(set(EMOTION_COLORS.values())) == len(CANONICAL_ORDER)
    for hex_code in EMOTION_COLORS.values():
        assert hex_code.startswith("#") and len(hex_code) == 7
