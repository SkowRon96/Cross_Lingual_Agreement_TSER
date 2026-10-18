import glob
import numpy as np
import pytest
import torch
from pathlib import Path
from app.config.emotions import Emotion
from app.config.settings import Settings
from app.data.schema import AudioRef, Sample
from app.models.speech.architectures import ParallelModel

# The test below builds speech_cnn_tf_ravdess, which loads this exact file --
# guard on it, not on another checkpoint, or the test silently skips forever.
_RAVDESS_WEIGHTS = Settings().speech_weights_dir / "cnn_transf_parallel_model_v2.pt"


def test_parallel_model_output_shapes():
    m = ParallelModel(num_emotions=8).eval()
    x = torch.randn(2, 1, 128, 563)
    with torch.no_grad():
        logits, soft = m(x)
    assert logits.shape == (2, 8)
    assert soft.shape == (2, 8)
    assert torch.allclose(soft.sum(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(not _RAVDESS_WEIGHTS.exists(),
                    reason="custom speech weights not present")
def test_ravdess_weights_load_and_predict_on_real_audio():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("speech_cnn_tf_ravdess")
    assert m.modality == "speech"
    assert Emotion.NEUTRAL in m.mapper.supported  # calm+neutral both -> neutral
    # multi-sample batch: the per-corpus StandardScaler needs >1 sample to be non-degenerate
    wavs = sorted(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))[:3]
    samples = [Sample(sample_id=f"RAVDESS:p{i}", studies=[1], corpus="RAVDESS", language="en",
                      text="x", audio=AudioRef(wav_path=Path(w)), gold_raw="neutral", gold=Emotion.NEUTRAL)
               for i, w in enumerate(wavs)]
    preds = m.predict_batch(samples)
    assert len(preds) == 3
    for pred in preds:
        assert pred.pred in set(Emotion) and pred.modality == "speech"
        assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
