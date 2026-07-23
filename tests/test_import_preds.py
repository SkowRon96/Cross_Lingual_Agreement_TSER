from app.config.emotions import Emotion
from app.predict.store import PredictionStore
from app.data.import_preds import import_predictions
import pandas as pd


def test_import_builds_predictions(tmp_path):
    df = pd.DataFrame([
        {"sample_id": "nemo:a", "corpus": "nemo", "raw_response": "happiness", "pred": "happiness"},
        {"sample_id": "nemo:b", "corpus": "nemo", "raw_response": "I cannot tell", "pred": ""},
        {"sample_id": "cafe:c", "corpus": "cafe", "raw_response": "anger", "pred": "anger"},
    ])
    pf = tmp_path / "qwen_predictions.parquet"
    df.to_parquet(pf, index=False)
    store = PredictionStore(tmp_path / "store")
    n = import_predictions(pf, store, study=2, corpus="nemo")   # only nemo rows
    assert n == 2
    preds = {p.sample_id: p for p in store.read(2, "nemo", "speech_zeroshot")}
    assert preds["nemo:a"].pred is Emotion.HAPPINESS
    assert preds["nemo:a"].scores[Emotion.HAPPINESS] == 1.0
    assert preds["nemo:a"].confidence is None
    assert preds["nemo:b"].abstained is True and preds["nemo:b"].pred is None


def test_import_cache_uses_plain_config_hash(tmp_path):
    """write() must store the plain config_hash so is_cached/stored_config_hash work."""
    df = pd.DataFrame([
        {"sample_id": "nemo:a", "corpus": "nemo", "raw_response": "anger", "pred": "anger"},
    ])
    pf = tmp_path / "q.parquet"
    df.to_parquet(pf, index=False)
    store = PredictionStore(tmp_path / "store")
    import_predictions(pf, store, study=2, corpus="nemo")
    assert store.stored_config_hash(2, "nemo", "speech_zeroshot") == "qwen2-audio-7b-instruct@zeroshot"
    assert store.is_cached(2, "nemo", "speech_zeroshot",
                           "qwen2-audio-7b-instruct@zeroshot", ["nemo:a"])
