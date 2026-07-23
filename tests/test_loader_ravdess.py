import pytest
from app.data.ravdess import parse_ravdess_name, load_ravdess


def test_parse_ravdess_name():
    d = parse_ravdess_name("03-01-06-02-02-01-16")
    assert d["modality"] == "03" and d["emotion"] == "06"
    assert d["statement"] == "02" and d["actor"] == "16"


def _make_pair(tmp_path, emotion="06", statement="02", intensity="02", rep="01", actor="16"):
    key = f"01-{emotion}-{intensity}-{statement}-{rep}-{actor}"
    adir = tmp_path / "Audio_Speech_Actors_01-24" / f"Actor_{actor}"
    vdir = tmp_path / f"Actor_{actor}"
    adir.mkdir(parents=True); vdir.mkdir(parents=True)
    wav = adir / f"03-{key}.wav"; wav.write_bytes(b"RIFF")  # content irrelevant; loader only uses the path
    mp4 = vdir / f"01-{key}.mp4"; mp4.write_bytes(b"\x00")
    return tmp_path, wav, mp4


def test_load_ravdess_pairs_and_maps(tmp_path):
    root, wav, mp4 = _make_pair(tmp_path, emotion="06", statement="02", actor="16")
    samples = load_ravdess(root / "Audio_Speech_Actors_01-24", root)
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "RAVDESS:01-06-02-02-01-16"
    assert s.corpus == "RAVDESS" and s.language == "en" and s.studies == [1]
    assert s.gold_raw == "fearful" and s.gold is None
    assert s.text == "Dogs are sitting by the door"
    assert s.audio.wav_path == wav and s.has_face is True and s.video_path == mp4
    assert s.speaker_id == "ravdess_actor_16" and s.gender == "female"  # 16 even
    assert s.intensity == "strong"


def test_load_ravdess_missing_video_raises(tmp_path):
    root, wav, mp4 = _make_pair(tmp_path, actor="16")
    mp4.unlink()  # remove the paired video
    with pytest.raises(FileNotFoundError):
        load_ravdess(root / "Audio_Speech_Actors_01-24", root)


def test_load_ravdess_odd_actor_is_male_and_normal_intensity(tmp_path):
    root, wav, mp4 = _make_pair(tmp_path, emotion="03", intensity="01", actor="07")
    s = load_ravdess(root / "Audio_Speech_Actors_01-24", root)[0]
    assert s.gender == "male"        # actor 07 odd
    assert s.intensity == "normal"   # intensity 01
    assert s.gold_raw == "happy"
