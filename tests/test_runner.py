from pathlib import Path
from app.config.emotions import Emotion
from app.data.schema import Sample, AudioRef
from app.models.base import Prediction
from app.predict.store import PredictionStore
from app.predict.runner import run_prediction, chunked


class StubModel:
    model_id = "stub"
    modality = "text"
    def __init__(self):
        self.calls = 0
    def config_signature(self):
        return "cfg-v1"
    def predict_batch(self, samples):
        self.calls += len(samples)
        return [Prediction(sample_id=s.sample_id, model_id=self.model_id, modality="text",
                           pred_raw="joy", pred=Emotion.HAPPINESS, confidence=0.9,
                           scores={Emotion.HAPPINESS: 0.9, Emotion.NEUTRAL: 0.1})
                for s in samples]


def _samples(n):
    return [Sample(sample_id=f"c:{i}", studies=[1], corpus="c", language="en", text="t",
                   audio=AudioRef(wav_path=Path("a.wav")), gold_raw="happy", gold=Emotion.HAPPINESS)
            for i in range(n)]


def test_chunked():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_run_then_cached(tmp_path):
    store = PredictionStore(tmp_path)
    m = StubModel()
    assert run_prediction(_samples(5), m, store, 1, "c", batch_size=2) == "done"
    assert m.calls == 5
    assert len(store.read(1, "c", "stub")) == 5
    # second run: fully cached, model not called again
    m2 = StubModel()
    assert run_prediction(_samples(5), m2, store, 1, "c") == "cached"
    assert m2.calls == 0


def test_resume_only_predicts_missing(tmp_path):
    store = PredictionStore(tmp_path)
    m = StubModel()
    run_prediction(_samples(3), m, store, 1, "c", batch_size=10)  # predicts 3
    m2 = StubModel()
    # now ask for 5 (superset): only the 2 new ones are predicted
    assert run_prediction(_samples(5), m2, store, 1, "c", batch_size=10) == "done"
    assert m2.calls == 2
    assert len(store.read(1, "c", "stub")) == 5


def test_force_recomputes(tmp_path):
    store = PredictionStore(tmp_path)
    run_prediction(_samples(3), StubModel(), store, 1, "c")
    m = StubModel()
    assert run_prediction(_samples(3), m, store, 1, "c", force=True) == "done"
    assert m.calls == 3  # recomputed despite cache
