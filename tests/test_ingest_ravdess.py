import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.data.ravdess import load_ravdess
from app.data.ingest import ingest_corpus
from app.data.index_io import read_index

_HAS_DATA = bool(glob.glob("datasets/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-*.wav"))


@pytest.mark.skipif(not _HAS_DATA, reason="RAVDESS dataset not present")
def test_ingest_ravdess_real_subset(tmp_path):
    # Load the full corpus, then ingest just Actor_01's samples for speed.
    samples = load_ravdess(Path("datasets/RAVDESS/Audio_Speech_Actors_01-24"),
                           Path("datasets/RAVDESS"))
    subset = [s for s in samples if s.speaker_id == "ravdess_actor_01"]
    assert len(subset) == 60  # RAVDESS: 60 speech utterances per actor
    report = ingest_corpus(subset, "RAVDESS_actor01", tmp_path)
    assert report.total == 60
    assert report.dropped == 0  # every RAVDESS label maps (calm->neutral, fearful->fear, ...)
    assert report.per_raw_label["calm"] == "neutral"
    assert report.per_raw_label["fearful"] == "fear"
    got = read_index(tmp_path / "RAVDESS_actor01.parquet")
    assert all(s.has_face and s.video_path is not None for s in got)
    assert all(s.gold is not None for s in got)  # nothing dropped
