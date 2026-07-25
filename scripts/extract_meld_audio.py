"""Extract 16 kHz mono WAV from each MELD utterance video (ffmpeg). Run once before ingest.

Iterates exactly the utterances the MELD loader yields (CSV as source of truth), so the
extracted wavs line up 1:1 with the Samples. Idempotent: skips wavs already extracted.

  PYTHONPATH=. .venv/bin/python scripts/extract_meld_audio.py

Output: datasets/MELD/audio/{split}/diaD_uttU.wav
"""
import subprocess
import sys
from pathlib import Path

from app.config.settings import Settings
from app.data.loaders.meld import load_meld

RAW = Path("datasets/MELD/MELD.Raw")
AUDIO = Path("datasets/MELD/audio")


def main():
    samples = load_meld(RAW, AUDIO)
    print(f"{len(samples)} utterances to extract -> {AUDIO}")
    done = fail = skip = 0
    failures = []
    for i, s in enumerate(samples):
        out = s.audio.wav_path
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(s.video_path),
             "-vn", "-ac", "1", "-ar", "16000", str(out)],
            capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            done += 1
        else:
            fail += 1
            failures.append((s.sample_id, r.stderr.strip()[:120]))
            out.unlink(missing_ok=True)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(samples)}  done={done} skip={skip} fail={fail}")
    print(f"DONE: extracted={done} skipped={skip} failed={fail}")
    if failures:
        print("failures (first 10):")
        for sid, err in failures[:10]:
            print(f"  {sid}: {err}")
    sys.exit(1 if fail and done == 0 else 0)


if __name__ == "__main__":
    main()
