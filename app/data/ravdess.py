"""RAVDESS (English) loader: original Zenodo files -> tri-modal Sample list (Study 1).

Emotion and text are decoded from the 7-code filename; audio (modality 03 WAV)
is paired with the audio-visual video (modality 01 MP4) by the shared identifier.
"""
from pathlib import Path

from app.data.schema import AudioRef, Sample

RAVDESS_EMOTIONS: dict[str, str] = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_STATEMENTS: dict[str, str] = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
_FIELDS = ["modality", "vocal", "emotion", "intensity", "statement", "repetition", "actor"]


def parse_ravdess_name(stem: str) -> dict:
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Not a RAVDESS filename stem: {stem!r}")
    return dict(zip(_FIELDS, parts))


def load_ravdess(audio_dir: Path, video_dir: Path) -> list[Sample]:
    audio_dir = Path(audio_dir)
    video_dir = Path(video_dir)
    samples: list[Sample] = []
    for wav in sorted(audio_dir.glob("Actor_*/03-*.wav")):
        c = parse_ravdess_name(wav.stem)
        key = f"{c['vocal']}-{c['emotion']}-{c['intensity']}-{c['statement']}-{c['repetition']}-{c['actor']}"
        mp4 = video_dir / f"Actor_{c['actor']}" / f"01-{key}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(f"No paired RAVDESS video for {wav.name}: expected {mp4}")
        actor_num = int(c["actor"])
        samples.append(Sample(
            sample_id=f"RAVDESS:{key}",
            studies=[1],
            corpus="RAVDESS",
            language="en",
            text=RAVDESS_STATEMENTS[c["statement"]],
            audio=AudioRef(wav_path=wav),
            has_face=True,
            video_path=mp4,
            gold_raw=RAVDESS_EMOTIONS[c["emotion"]],
            gold=None,
            speaker_id=f"ravdess_actor_{c['actor']}",
            gender="male" if actor_num % 2 == 1 else "female",
            age=None,
            intensity="normal" if c["intensity"] == "01" else "strong",
        ))
    return samples
