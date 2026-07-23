from pathlib import Path
from app.config.emotions import Emotion
from app.data.schema import Sample, AudioRef
from app.data.ingest import resolve_gold, ingest_corpus
from app.data.index_io import read_index


def _s(sid, raw):
    return Sample(sample_id=sid, studies=[2], corpus="demo", language="en",
                  text="t", audio=AudioRef(wav_path=Path("a.wav")),
                  gold_raw=raw, gold=None)


def test_resolve_gold_maps_merges_and_drops():
    out = resolve_gold([_s("1", "calm"), _s("2", "excitement"), _s("3", "apology"), _s("4", "anger")])
    assert out[0].gold is Emotion.NEUTRAL       # calm -> neutral (merge)
    assert out[1].gold is Emotion.HAPPINESS     # excitement -> happiness (merge)
    assert out[2].gold is None                  # apology -> DROP
    assert out[3].gold is Emotion.ANGER


def test_ingest_corpus_writes_index_and_reports(tmp_path):
    samples = [_s("1", "calm"), _s("2", "apology"), _s("3", "anger"), _s("4", "apology")]
    report = ingest_corpus(samples, "demo", tmp_path)
    assert report.total == 4
    assert report.kept == 2                      # calm(->neutral), anger
    assert report.dropped == 2                   # two apology
    assert report.dropped_labels == {"apology": 2}
    assert report.per_raw_label["calm"] == "neutral"
    assert report.per_raw_label["apology"] == "DROP"
    # index written and round-trips, dropped rows kept with gold=None
    got = read_index(tmp_path / "demo.parquet")
    assert len(got) == 4
    assert sum(1 for s in got if s.gold is None) == 2
