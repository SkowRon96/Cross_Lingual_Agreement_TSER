"""Standalone Qwen2-Audio zero-shot emotion inference for the Windows + NVIDIA box.

Run on the GPU machine (NOT on the Mac). Produces qwen_predictions.parquet that is copied
back to the repo and imported via `python -m app.cli import-predictions`.

Usage (from the transferred data root):
  python run_qwen.py --cameo-dir CAMEO/data --ravdess-audio-dir RAVDESS/Audio_Speech_Actors_01-24 \
                     --out qwen_predictions.parquet

Resumable: re-run to continue; already-predicted sample_ids are skipped.
"""
import argparse
import glob
import io
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ---- prompt + parser: COPIED VERBATIM from app/models/speech/zeroshot/prompt.py ----
CANONICAL_WORDS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
_SYNONYMS = {
    "anger": "anger", "angry": "anger", "annoyed": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear", "afraid": "fear", "scared": "fear",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness", "joyful": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "calm": "neutral", "none": "neutral",
}
def build_prompt():
    return ("Listen to the speech and classify the speaker's emotion into exactly one of "
            f"these categories: {', '.join(CANONICAL_WORDS)}. Answer with only one word from that list.")
def parse_response(raw):
    if not raw:
        return ""
    text = raw.strip().lower()
    for key in sorted(_SYNONYMS, key=len, reverse=True):
        if key in text:
            return _SYNONYMS[key]
    return ""
# --------------------------------------------------------------------------------

RAVDESS_EMOTIONS = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
TARGET_SR = 16000


def corpus_from_cameo_path(path):
    return re.sub(r"-\d+-of-\d+\.parquet$", "", os.path.basename(path))


def iter_cameo(cameo_dir):
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(os.path.join(cameo_dir, "*.parquet"))):
        corpus = corpus_from_cameo_path(path)
        cols = pq.read_table(path, columns=["file_id", "audio"]).to_pydict()
        for i in range(len(cols["file_id"])):
            sid = f"{corpus}:{cols['file_id'][i]}"
            audio_bytes = cols["audio"][i]["bytes"]
            yield sid, corpus, audio_bytes, None


def iter_nemo(nemo_dir):
    # Study-1 nEMO (Polish speech): standalone parquet shards, distinct namespace
    # from CAMEO's Study-2 "nemo" split. sample_id/corpus match app/data/loaders/nemo.py.
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(os.path.join(nemo_dir, "*.parquet"))):
        cols = pq.read_table(path, columns=["file_id", "audio"]).to_pydict()
        for i in range(len(cols["file_id"])):
            sid = f"nEMO:{cols['file_id'][i]}"
            yield sid, "nEMO", cols["audio"][i]["bytes"], None


def iter_ravdess(audio_dir):
    for wav in sorted(glob.glob(os.path.join(audio_dir, "Actor_*", "03-*.wav"))):
        c = os.path.basename(wav).split(".")[0].split("-")
        key = f"{c[1]}-{c[2]}-{c[3]}-{c[4]}-{c[5]}-{c[6]}"
        yield f"RAVDESS:{key}", "RAVDESS", None, wav


def load_audio(audio_bytes, wav_path):
    if wav_path is not None:
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        # lightweight linear resample to 16 kHz (processor expects 16 kHz)
        n = int(round(len(data) * TARGET_SR / sr))
        data = np.interp(np.linspace(0, len(data), n, endpoint=False), np.arange(len(data)), data).astype("float32")
    return np.ascontiguousarray(data, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameo-dir", default=None)
    ap.add_argument("--nemo-dir", default=None)
    ap.add_argument("--ravdess-audio-dir", default=None)
    ap.add_argument("--out", default="qwen_predictions.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_parquet(args.out, columns=["sample_id"])["sample_id"].tolist())
        print(f"resuming: {len(done)} already done")

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    prompt = build_prompt()

    sources = []
    if args.cameo_dir:
        sources.append(iter_cameo(args.cameo_dir))
    if args.nemo_dir:
        sources.append(iter_nemo(args.nemo_dir))
    if args.ravdess_audio_dir:
        sources.append(iter_ravdess(args.ravdess_audio_dir))

    buffer, n = [], 0
    for source in sources:
        for sid, corpus, audio_bytes, wav_path in source:
            if sid in done:
                continue
            audio = load_audio(audio_bytes, wav_path)
            conversation = [{"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audios=[audio], sampling_rate=TARGET_SR,
                               return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            reply = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            buffer.append({"sample_id": sid, "corpus": corpus,
                           "raw_response": reply, "pred": parse_response(reply)})
            n += 1
            if len(buffer) >= 100:
                _flush(buffer, args.out)
                buffer = []
                print(f"\rpredicted {n}", end="")
    if buffer:
        _flush(buffer, args.out)
    print(f"\ndone: wrote/updated {args.out}")


def _flush(buffer, out):
    new = pd.DataFrame(buffer)
    if os.path.exists(out):
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True)
    new.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
