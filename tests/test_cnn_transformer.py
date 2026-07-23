import glob
import numpy as np
import pytest
import torch
from pathlib import Path
from app.config.emotions import Emotion
from app.models.speech.architectures import ParallelModel


def test_parallel_model_output_shapes():
    m = ParallelModel(num_emotions=8).eval()
    x = torch.randn(2, 1, 128, 563)
    with torch.no_grad():
        logits, soft = m(x)
    assert logits.shape == (2, 8)
    assert soft.shape == (2, 8)
    assert torch.allclose(soft.sum(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(not glob.glob("old_projects/polish-speech-emotion-recognition/notebooks/models/cnn_transf_parallel_model_v1.pt"),
                    reason="custom speech weights not present")
def test_ravdess_weights_load_and_predict_on_real_audio():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("speech_cnn_tf_ravdess")
    assert m.modality == "speech"
    assert Emotion.NEUTRAL in m.mapper.supported  # calm+neutral both -> neutral
    # a real RAVDESS wav
    wav = sorted(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))[0]
    s = Sample(sample_id="RAVDESS:probe", studies=[1], corpus="RAVDESS", language="en",
               text="Kids are talking by the door", audio=AudioRef(wav_path=Path(wav)),
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])   # single-sample batch: scaler fit on this one (probe only)
    assert pred.pred in set(Emotion) and pred.modality == "speech"
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
