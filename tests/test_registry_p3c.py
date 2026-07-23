import pytest
from app.models.registry import MODEL_IDS, build_model, ROBERTA6_ID2LABEL


def test_new_text_ids_registered():
    for mid in ["xlmr_pooled10", "mling_tabularis", "pl_text_roberta6"]:
        assert mid in MODEL_IDS


def test_roberta6_assumed_order_constant():
    # The documented ASSUMPTION (unverified) — guards against accidental reordering.
    assert ROBERTA6_ID2LABEL == {0:"anger",1:"fear",2:"disgust",3:"sadness",4:"joy",5:"none of them"}


@pytest.mark.slow
def test_build_roberta6_applies_override():
    m = build_model("pl_text_roberta6")
    assert m.native_labels == ["anger","fear","disgust","sadness","joy","none of them"]
    from app.config.emotions import Emotion
    assert Emotion.NEUTRAL in m.mapper.supported and Emotion.SURPRISE not in m.mapper.supported


@pytest.mark.slow
def test_build_poltextlab_supported():
    from app.config.emotions import Emotion
    m = build_model("xlmr_pooled10")
    assert m.mapper.supported == {Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST,
                                  Emotion.SADNESS, Emotion.HAPPINESS, Emotion.NEUTRAL}
    assert Emotion.SURPRISE not in m.mapper.supported


@pytest.mark.slow
def test_build_tabularisai_supported():
    from app.config.emotions import Emotion
    m = build_model("mling_tabularis")
    assert m.mapper.supported == set(Emotion)   # all 7
