from app.models.speech.zeroshot.prompt import build_prompt, parse_response, CANONICAL_WORDS


def test_build_prompt_lists_all_seven_words():
    p = build_prompt().lower()
    for w in CANONICAL_WORDS:
        assert w in p


def test_parse_exact_words():
    assert parse_response("anger") == "anger"
    assert parse_response("Neutral") == "neutral"


def test_parse_from_sentence_and_synonyms():
    assert parse_response("The emotion is happiness.") == "happiness"
    assert parse_response("joyful") == "happiness"
    assert parse_response("The speaker sounds angry") == "anger"
    assert parse_response("surprised") == "surprise"
    assert parse_response("afraid") == "fear"
    assert parse_response("calm") == "neutral"


def test_parse_no_match_returns_none():
    assert parse_response("I cannot tell") is None
    assert parse_response("") is None
