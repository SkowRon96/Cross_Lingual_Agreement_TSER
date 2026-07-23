from pathlib import Path
from app.config.emotions import Emotion
from app.models.base import Prediction
from app.predict.store import PredictionStore


def _pred(sid, pred=Emotion.HAPPINESS):
    return Prediction(
        sample_id=sid, model_id="m1", modality="text",
        pred_raw="joy", pred=pred, confidence=0.9,
        scores={Emotion.HAPPINESS: 0.9, Emotion.NEUTRAL: 0.1},
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    store = PredictionStore(tmp_path)
    preds = [_pred("a"), _pred("b", Emotion.NEUTRAL)]
    sig = store.signature("m1", "cfg1", ["a", "b"])
    store.write(1, "nEMO", "m1", preds, signature=sig)
    got = store.read(1, "nEMO", "m1")
    assert {p.sample_id for p in got} == {"a", "b"}
    got_b = next(p for p in got if p.sample_id == "b")
    assert got_b.pred is Emotion.NEUTRAL
    assert got_b.scores[Emotion.HAPPINESS] == 0.9


def test_signature_is_order_independent_over_sample_ids(tmp_path: Path):
    store = PredictionStore(tmp_path)
    assert store.signature("m1", "cfg1", ["a", "b"]) == store.signature("m1", "cfg1", ["b", "a"])


def test_signature_changes_with_config(tmp_path: Path):
    store = PredictionStore(tmp_path)
    assert store.signature("m1", "cfg1", ["a"]) != store.signature("m1", "cfg2", ["a"])


def test_is_cached_matches_signature(tmp_path: Path):
    store = PredictionStore(tmp_path)
    sig = store.signature("m1", "cfg1", ["a"])
    assert store.is_cached(1, "nEMO", "m1", sig) is False
    store.write(1, "nEMO", "m1", [_pred("a")], signature=sig)
    assert store.is_cached(1, "nEMO", "m1", sig) is True
    assert store.is_cached(1, "nEMO", "m1", "different-signature") is False


def test_existing_sample_ids_and_append(tmp_path: Path):
    store = PredictionStore(tmp_path)
    sig = store.signature("m1", "cfg1", ["a", "b"])
    store.write(1, "nEMO", "m1", [_pred("a")], signature=sig)
    assert store.existing_sample_ids(1, "nEMO", "m1") == {"a"}
    store.append(1, "nEMO", "m1", [_pred("b")])
    assert store.existing_sample_ids(1, "nEMO", "m1") == {"a", "b"}
