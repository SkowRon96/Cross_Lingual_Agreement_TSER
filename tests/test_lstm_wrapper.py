import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.config.settings import Settings

_S = Settings()
_HAS = (_S.lstm_model_dir / "model.h5").exists() and _S.tf_python.exists()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS, reason="trained LSTM model or .venv-tf not present")
def test_lstm_wrapper_predicts_via_registry():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("pl_text_lstm")
    assert m.modality == "text"
    assert m.mapper.supported == set(Emotion)   # all 7 (happiness merge only)
    s = Sample(sample_id="pl:1", studies=[1], corpus="nEMO", language="pl",
               text="jestem bardzo szczęśliwy i zadowolony z dzisiejszego dnia",
               audio=AudioRef(wav_path=Path("a.wav")), gold_raw="happiness", gold=Emotion.HAPPINESS)
    [pred] = m.predict_batch([s])
    assert pred.pred is Emotion.HAPPINESS
    assert pred.modality == "text"
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5     # renormalized over supported
