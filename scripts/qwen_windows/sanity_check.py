r"""Quick sanity check for the Qwen2-Audio zero-shot setup — run BEFORE the full run.

Loads the model and runs it on a handful of clips from each source you point it at,
then PRINTS the sample_id, the model's raw answer, and the parsed canonical label so
you can eyeball that everything works (CUDA, model download, audio decode, prompt,
parser). It writes NOTHING — the real run is `run_qwen.py`.

Reuses run_qwen.py's logic (same prompt/parser/loaders), so there is nothing extra to
keep in sync. Put this file next to run_qwen.py.

Usage (PowerShell, from C:\qwen with .venv-qwen active):
  python sanity_check.py --cameo-dir data\CAMEO\data --nemo-dir data\nEMO\data ^
                         --ravdess-audio-dir data\RAVDESS\Audio_Speech_Actors_01-24 --n 3
"""
import argparse
import itertools

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from run_qwen import (
    audio_kwarg,
    build_prompt,
    classify,
    iter_cameo,
    iter_nemo,
    iter_ravdess,
    load_audio,
    parse_response,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameo-dir", default=None)
    ap.add_argument("--nemo-dir", default=None)
    ap.add_argument("--ravdess-audio-dir", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--n", type=int, default=3, help="clips to test per source")
    args = ap.parse_args()

    print("CUDA available:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA — the full run needs the GPU. Aborting sanity check.")
        return

    print(f"loading {args.model} (first run downloads ~16 GB) ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    prompt = build_prompt()
    print("prompt:", prompt)
    print(f"passing audio to the processor via '{audio_kwarg(processor)}=' "
          "(must NOT be silently ignored)")
    print("-" * 100)
    print(f"{'sample_id':45s} | {'raw_response':30s} | parsed")
    print("-" * 100)

    sources = []
    if args.cameo_dir:
        sources.append(("CAMEO", iter_cameo(args.cameo_dir)))
    if args.nemo_dir:
        sources.append(("nEMO", iter_nemo(args.nemo_dir)))
    if args.ravdess_audio_dir:
        sources.append(("RAVDESS", iter_ravdess(args.ravdess_audio_dir)))
    if not sources:
        print("no source given — pass at least one of --cameo-dir / --nemo-dir / --ravdess-audio-dir")
        return

    for label, source in sources:
        print(f"# source: {label}")
        for sid, corpus, audio_bytes, wav_path in itertools.islice(source, args.n):
            audio = load_audio(audio_bytes, wav_path)
            reply = classify(processor, model, audio, prompt)
            parsed = parse_response(reply) or "(abstain)"
            print(f"{sid:45s} | {reply.strip()[:30]:30s} | {parsed}")
    print("-" * 100)
    print("If the parsed column looks reasonable, run the full run_qwen.py.")


if __name__ == "__main__":
    main()
