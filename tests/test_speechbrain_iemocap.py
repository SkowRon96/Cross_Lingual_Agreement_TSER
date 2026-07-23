import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.models.speech.speechbrain_iemocap import IEMOCAP_TO_CANONICAL


def test_iemocap_label_map_covers_four_classes():
    assert set(IEMOCAP_TO_CANONICAL.values()) == {"anger", "happiness", "sadness", "neutral"}


@pytest.mark.slow
@pytest.mark.skipif(not glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"),
                    reason="RAVDESS dataset not present")
def test_speechbrain_predicts_on_real_audio():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("en_speech_iemocap")
    assert m.modality == "speech"
    # IEMOCAP has only 4 classes -> no fear/disgust/surprise
    assert m.mapper.supported == {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}
    wav = sorted(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))[0]
    s = Sample(sample_id="RAVDESS:probe", studies=[1], corpus="RAVDESS", language="en",
               text="Kids are talking by the door", audio=AudioRef(wav_path=Path(wav)),
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])
    assert pred.pred in {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
    assert pred.confidence > 0.8   # real model is confident on this clip; guards against the .exp() flattening bug
