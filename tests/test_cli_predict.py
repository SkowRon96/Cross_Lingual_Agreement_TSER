import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.data.ravdess import load_ravdess
from app.data.index_io import write_index
from app.predict.store import PredictionStore
from app.cli import main

_HAS_DATA = bool(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA, reason="RAVDESS dataset not present")
def test_cli_predict_en_text_on_ravdess_subset(tmp_path, capsys):
    # Build a tiny real index (3 Actor_01 utterances) and predict EN text end-to-end.
    samples = load_ravdess(Path("datasets/RAVDESS/Audio_Speech_Actors_01-24"),
                           Path("datasets/RAVDESS"))
    subset = [s for s in samples if s.speaker_id == "ravdess_actor_01"][:3]
    index = tmp_path / "RAVDESS.parquet"
    write_index(subset, index)
    results = tmp_path / "results"
    rc = main(["predict", "--study", "1", "--corpus", "RAVDESS",
               "--model", "en_text_hartmann", "--index", str(index),
               "--results-root", str(results)])
    assert rc == 0
    store = PredictionStore(results / "predictions")
    preds = store.read(1, "RAVDESS", "en_text_hartmann")
    assert len(preds) == 3
    for p in preds:
        assert p.pred in set(Emotion) and p.modality == "text"
    # RAVDESS text is the fixed neutral sentence -> model should lean neutral-ish,
    # but we only assert validity here, not a specific label (text carries little emotion).
