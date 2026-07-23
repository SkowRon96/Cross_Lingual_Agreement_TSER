from pathlib import Path

import pytest

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
    store.write(1, "nEMO", "m1", preds, config_hash="cfg1")
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
    assert store.is_cached(1, "nEMO", "m1", "cfg1", ["a"]) is False
    store.write(1, "nEMO", "m1", [_pred("a")], config_hash="cfg1")
    assert store.is_cached(1, "nEMO", "m1", "cfg1", ["a"]) is True
    assert store.is_cached(1, "nEMO", "m1", "cfg1", ["a", "b"]) is False
    assert store.is_cached(1, "nEMO", "m1", "cfg2", ["a"]) is False


def test_partial_write_not_cached_for_full_target(tmp_path: Path):
    store = PredictionStore(tmp_path)
    store.write(1, "nEMO", "m1", [_pred("a")], config_hash="cfg1")
    assert store.is_cached(1, "nEMO", "m1", "cfg1", ["a", "b"]) is False

    store.append(1, "nEMO", "m1", [_pred("b")])

    assert store.is_cached(1, "nEMO", "m1", "cfg1", ["a", "b"]) is True
    assert store.is_cached(1, "nEMO", "m1", "cfg1", ["a"]) is False


def test_append_dedups_sample_id(tmp_path: Path):
    store = PredictionStore(tmp_path)
    store.write(1, "nEMO", "m1", [_pred("a")], config_hash="cfg1")
    store.append(1, "nEMO", "m1", [_pred("a", Emotion.SADNESS)])

    got = store.read(1, "nEMO", "m1")
    a_rows = [p for p in got if p.sample_id == "a"]
    assert len(a_rows) == 1
    # the original (existing) row is kept, not the duplicate
    assert a_rows[0].pred is Emotion.HAPPINESS


def test_write_dedups_intra_batch_sample_id(tmp_path: Path):
    store = PredictionStore(tmp_path)
    preds = [_pred("a"), _pred("a", Emotion.SADNESS)]
    store.write(1, "nEMO", "m1", preds, config_hash="cfg1")

    got = store.read(1, "nEMO", "m1")
    a_rows = [p for p in got if p.sample_id == "a"]
    assert len(a_rows) == 1
    # first occurrence in the batch is kept
    assert a_rows[0].pred is Emotion.HAPPINESS


def test_append_dedups_intra_batch_sample_id(tmp_path: Path):
    store = PredictionStore(tmp_path)
    store.write(1, "nEMO", "m1", [_pred("a")], config_hash="cfg1")
    store.append(1, "nEMO", "m1", [_pred("b"), _pred("b", Emotion.SADNESS)])

    got = store.read(1, "nEMO", "m1")
    b_rows = [p for p in got if p.sample_id == "b"]
    assert len(b_rows) == 1
    assert b_rows[0].pred is Emotion.HAPPINESS


def test_existing_sample_ids_and_append(tmp_path: Path):
    store = PredictionStore(tmp_path)
    store.write(1, "nEMO", "m1", [_pred("a")], config_hash="cfg1")
    assert store.existing_sample_ids(1, "nEMO", "m1") == {"a"}
    store.append(1, "nEMO", "m1", [_pred("b")])
    assert store.existing_sample_ids(1, "nEMO", "m1") == {"a", "b"}


def test_append_without_prior_write_raises(tmp_path: Path):
    store = PredictionStore(tmp_path)
    with pytest.raises(ValueError):
        store.append(1, "nEMO", "m1", [_pred("a")])


def test_none_confidence_abstained_roundtrip(tmp_path: Path):
    store = PredictionStore(tmp_path)
    original = Prediction(
        sample_id="a", model_id="m1", modality="text",
        pred_raw="unknown", pred=None, confidence=None, scores={},
        abstained=True, abstain_reason="all probability mass mapped to dropped labels",
    )
    store.write(1, "nEMO", "m1", [original], config_hash="cfg1")
    got = store.read(1, "nEMO", "m1")
    assert len(got) == 1
    assert got[0] == original
