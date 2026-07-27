"""Compare face-frame aggregation strategies for the ViT face model on RAVDESS.

Current pipeline averages per-crop softmax over 8 frames (aggregate_face_scores = mean).
Averaging can dilute the peak expression (RAVDESS clips start/end neutral). We extract the
per-crop probability vectors ONCE, then compare 4 aggregations offline:
  mean       (current)   - mean of per-crop softmax
  max_pool               - element-wise max over crops
  max_conf               - the single crop whose top prob is highest
  vote                   - majority vote of per-crop argmax
Each aggregated native score vector is mapped to canonical via the model's own mapper and
scored against RAVDESS gold. Prints accuracy per strategy so we can decide whether to change
aggregate_face_scores.

  PYTHONPATH=. .venv/bin/python scripts/face_aggregation_experiment.py
"""
import numpy as np
import torch

from app.config.settings import Settings
from app.data.ravdess import load_ravdess
from app.data.video import extract_face_frames
from app.labels.mapping import map_label
from app.models.registry import build_model

s = Settings()
m = build_model("en_face_vit")          # ViTFaceModel: .processor .model .native_labels .mapper .device
samples = load_ravdess(s.ravdess_audio_dir, s.ravdess_video_dir)
print(f"RAVDESS samples: {len(samples)}")

# 1) extract per-crop probs once
probs_by_sid, gold_by_sid = {}, {}
no_face = 0
for i, smp in enumerate(samples):
    g = map_label(smp.gold_raw)
    if g is None:
        continue
    crops = extract_face_frames(smp.video_path, num_frames=m.num_frames)
    if not crops:
        no_face += 1
        continue
    inp = m.processor(images=crops, return_tensors="pt").to(m.device)
    with torch.no_grad():
        logits = m.model(**inp).logits
    probs_by_sid[smp.sample_id] = torch.softmax(logits, dim=-1).cpu().numpy()  # (n_crops, n_labels)
    gold_by_sid[smp.sample_id] = g
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(samples)} (no_face={no_face})")
print(f"usable: {len(probs_by_sid)} | no_face_detected: {no_face}")

n_labels = len(m.native_labels)


def agg(name, probs):
    if name == "mean":
        return probs.mean(axis=0)
    if name == "max_pool":
        return probs.max(axis=0)
    if name == "max_conf":
        return probs[np.argmax(probs.max(axis=1))]
    if name == "vote":
        v = np.zeros(n_labels)
        for j in np.argmax(probs, axis=1):
            v[j] += 1
        return v / v.sum()
    raise ValueError(name)


def score(name):
    correct = n = abst = 0
    for sid, probs in probs_by_sid.items():
        native = agg(name, probs)
        canon = m.mapper.map_scores(native.tolist())
        if not canon:
            abst += 1
            continue
        pred = max(canon, key=canon.get)
        n += 1
        correct += (pred == gold_by_sid[sid])
    return correct / n if n else 0.0, n, abst


print("\n=== RAVDESS face aggregation (canonical accuracy vs gold) ===")
for name in ["mean", "max_pool", "max_conf", "vote"]:
    acc, n, abst = score(name)
    tag = "  <- current" if name == "mean" else ""
    print(f"  {name:9s} acc={acc:.3f} (n={n}, abstain={abst}){tag}")
