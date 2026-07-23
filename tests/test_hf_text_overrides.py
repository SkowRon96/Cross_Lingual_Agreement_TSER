from app.config.emotions import Emotion
from app.labels.mapper import LabelMapper
from app.models.text.hf_text import native_labels_from_id2label

# Verified id2label sets (grounded from HF configs).
POLTEXTLAB = {"0":"Neutral","1":"Anger","2":"Fear","3":"Disgust","4":"Sadness",
              "5":"Joy","6":"Hope","7":"Pride","8":"Enthusiasm","9":"None of them"}
TABULARIS = {"0":"anger","1":"contempt","2":"disgust","3":"fear","4":"frustration",
             "5":"gratitude","6":"joy","7":"love","8":"neutral","9":"sadness","10":"surprise"}
ROBERTA6 = {0:"anger",1:"fear",2:"disgust",3:"sadness",4:"joy",5:"none of them"}


def test_poltextlab_supported_set():
    m = LabelMapper(native_labels_from_id2label(POLTEXTLAB))
    assert m.supported == {Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST,
                           Emotion.SADNESS, Emotion.HAPPINESS, Emotion.NEUTRAL}  # no surprise


def test_tabularisai_supported_set_is_full_seven():
    m = LabelMapper(native_labels_from_id2label(TABULARIS))
    assert m.supported == set(Emotion)  # all 7


def test_roberta6_override_with_none_of_them_neutral():
    m = LabelMapper(native_labels_from_id2label(ROBERTA6),
                    overrides={"none of them": Emotion.NEUTRAL})
    assert m.supported == {Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST,
                           Emotion.SADNESS, Emotion.HAPPINESS, Emotion.NEUTRAL}  # no surprise
    assert m.map_label("none of them") is Emotion.NEUTRAL
