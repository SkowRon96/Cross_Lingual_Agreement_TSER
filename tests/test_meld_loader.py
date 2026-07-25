from pathlib import Path

from app.data.loaders.meld import load_meld


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime\n"
    # write as cp1252 with a curly quote to exercise the decoding
    with open(path, "w", encoding="cp1252") as f:
        f.write(header)
        for r in rows:
            f.write(r + "\n")


def test_load_meld_builds_samples_and_skips_missing_video(tmp_path):
    raw = tmp_path / "MELD.Raw"
    (raw / "train" / "train_splits").mkdir(parents=True)
    (raw / "dev_splits_complete").mkdir(parents=True)
    _write_csv(raw / "train" / "train_sent_emo.csv",
               ["1,“Hello” there,Ross,joy,positive,0,0,1,1,0,0",
                "2,I am scared,Phoebe,fear,negative,0,1,1,1,0,0"])
    _write_csv(raw / "dev_sent_emo.csv",
               ["1,No video here,Joey,anger,negative,5,2,1,1,0,0"])
    # create videos for the two train rows, but NOT for the dev row
    (raw / "train" / "train_splits" / "dia0_utt0.mp4").write_bytes(b"x")
    (raw / "train" / "train_splits" / "dia0_utt1.mp4").write_bytes(b"x")
    # (empty test csv folder omitted -> loader tolerates missing? no: provide empty file)
    _write_csv(raw / "test_sent_emo.csv", [])
    (raw / "output_repeated_splits_test").mkdir(parents=True)

    samples = load_meld(raw, tmp_path / "audio")
    by_id = {s.sample_id: s for s in samples}
    assert set(by_id) == {"MELD:train:dia0_utt0", "MELD:train:dia0_utt1"}  # dev row skipped (no video)
    s0 = by_id["MELD:train:dia0_utt0"]
    assert s0.corpus == "MELD" and s0.language == "en" and s0.studies == [1]
    assert s0.gold_raw == "joy"          # maps to happiness at ingest
    assert "Hello" in s0.text            # cp1252 curly quotes decoded, no crash
    assert s0.audio.wav_path == (tmp_path / "audio" / "train" / "dia0_utt0.wav")
    assert s0.video_path == (raw / "train" / "train_splits" / "dia0_utt0.mp4")
    assert s0.has_face is True
