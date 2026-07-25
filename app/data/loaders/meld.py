"""MELD (English, real dialogue from Friends) loader -> tri-modal Sample list (Study 1).

MELD's emotionally-loaded conversational TEXT is the point: unlike the acted, fixed-sentence
RAVDESS, here the text channel actually carries emotion, so text<->speech agreement becomes
measurable. 7 emotions map cleanly to the canonical space (joy -> happiness).

Layout (MELD.Raw):
  train/train_sent_emo.csv          + train/train_splits/diaD_uttU.mp4
  dev_sent_emo.csv                  + dev_splits_complete/diaD_uttU.mp4
  test_sent_emo.csv                 + output_repeated_splits_test/diaD_uttU.mp4
CSVs are Windows-1252 encoded (curly quotes as \\x92 etc.). Dialogue/Utterance IDs repeat
across splits, so sample_id includes the split. Audio is extracted to wav beforehand
(scripts/extract_meld_audio.py); this loader only references the wav path.
"""
import csv
from pathlib import Path

from app.data.schema import AudioRef, Sample

# (split, video subdir, csv path relative to MELD.Raw)
SPLITS: list[tuple[str, str, str]] = [
    ("train", "train/train_splits", "train/train_sent_emo.csv"),
    ("dev", "dev_splits_complete", "dev_sent_emo.csv"),
    ("test", "output_repeated_splits_test", "test_sent_emo.csv"),
]


def _read_csv(path: Path) -> list[dict]:
    # MELD CSVs are Windows-1252; decode faithfully so the text channel is clean.
    with open(path, encoding="cp1252", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def load_meld(raw_root: Path, audio_root: Path) -> list[Sample]:
    """raw_root = .../MELD/MELD.Raw ; audio_root = dir of extracted wavs ({split}/diaD_uttU.wav)."""
    raw_root = Path(raw_root)
    audio_root = Path(audio_root)
    samples: list[Sample] = []
    for split, vdir, csv_rel in SPLITS:
        for r in _read_csv(raw_root / csv_rel):
            key = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
            video = raw_root / vdir / f"{key}.mp4"
            if not video.exists():
                continue  # a couple of MELD utterances have no video file — skip (fail-safe)
            samples.append(Sample(
                sample_id=f"MELD:{split}:{key}",
                studies=[1],
                corpus="MELD",
                language="en",
                text=r["Utterance"],
                audio=AudioRef(wav_path=audio_root / split / f"{key}.wav"),
                has_face=True,
                video_path=video,
                gold_raw=r["Emotion"],
                gold=None,
                speaker_id=r.get("Speaker") or None,
                gender=None,
                age=None,
            ))
    return samples
