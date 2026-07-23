import pytest
from app.config.emotions import Emotion
from app.models.text.hf_text import native_labels_from_id2label, HFTextModel


def test_native_labels_ordered_by_int_key():
    id2label = {"2": "fear", "0": "anger", "1": "disgust"}
    assert native_labels_from_id2label(id2label) == ["anger", "disgust", "fear"]


@pytest.mark.slow
def test_hartmann_predicts_happiness_on_happy_text(tmp_path):
    # Downloads the real model (~330MB) into a temp HF cache; cached across runs.
    m = HFTextModel("en_text_hartmann",
                    "j-hartmann/emotion-english-distilroberta-base",
                    cache_dir=tmp_path)
    from app.data.schema import Sample, AudioRef
    from pathlib import Path
    s = Sample(sample_id="t:1", studies=[1], corpus="t", language="en",
               text="I am so happy and delighted today!",
               audio=AudioRef(wav_path=Path("a.wav")), gold_raw="happy", gold=Emotion.HAPPINESS)
    [pred] = m.predict_batch([s])
    assert pred.modality == "text"
    assert pred.pred is Emotion.HAPPINESS        # joy -> happiness, clearly happy text
    assert 0.0 < pred.confidence <= 1.0
    assert set(pred.scores).issubset(set(Emotion))
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
