"""Model-roster overview: id, source, modality, language, native emotions, AUTHOR-claimed accuracy.

The accuracy column is the figure the MODEL's author reports on THEIR OWN test set (sourced
from the HF model card, or — for the custom models — our documented notebook/_v2 held-out).
These numbers are NOT comparable across models (different datasets, class counts, single- vs
multi-label) and are NOT our measured accuracy on the study corpora. The `source` column says
where each number comes from. Custom-model 'author' = the original project (this study).

  PYTHONPATH=. .venv/bin/python scripts/models_overview.py
"""
from pathlib import Path

from app.config.settings import Settings

# model_id, source model, modality, language(s), native emotions, author-claimed accuracy, source
ROWS = [
    ("en_text_hartmann", "j-hartmann/distilroberta", "text", "en",
     "anger,disgust,fear,happiness,sadness,surprise,neutral",
     "0.66 acc (7-class)", "HF card: eval acc on 20% held-out of ~20k balanced"),
    ("mling_tabularis", "tabularisai/multilingual-emotion", "text", "multilingual (23)",
     "7 canonical (+ contempt,frustration,gratitude,love→DROP)",
     "macro-F1 0.839 / subset-acc 0.640", "HF card: 11-class multi-label, 11.5k test, τ=0.5"),
    ("xlmr_pooled10", "poltextlab/xlm-roberta pooled-emotions10", "text", "multilingual (EU)",
     "anger,fear,disgust,sadness,happiness,neutral (no surprise)",
     "0.82 acc / macro-F1 0.83", "poltextlab pooled-emotions10 card: 10-class (Neutral/Anger/Fear/"
     "Disgust/Sadness/Joy/Hope/Pride/Enthusiasm/None); v3 variant may differ slightly"),
    ("pl_text_roberta6 ⚠", "visegradmedia/Emotion_RoBERTa_polish6", "text", "pl",
     "anger,fear,disgust,sadness,happiness,neutral (no surprise)",
     "0.82 acc / macro-F1 0.85", "HF card (⚠ label order ASSUMED by us; secondary model)"),
    ("pl_text_lstm", "own LSTM (word2vec+BiLSTM)", "text", "pl",
     "7 canonical",
     "not published", "own model, no author-reported metric (see §9.1)"),
    ("speech_zeroshot", "Qwen2-Audio-7B-Instruct", "speech", "multilingual (audio)",
     "7 canonical (prompted)",
     "n/a (zero-shot)", "no SER fine-tuning — no author accuracy; clean/headline model"),
    ("speech_cnn_tf_ravdess", "own parallel CNN-Transformer (RAVDESS-8)", "speech", "en",
     "7 canonical (+ calm→neutral)",
     "0.71 (_v2 held-out; orig notebook 0.61)", "own: _v2 held-out 0.713 (was 0.613 pre-fix); LEAKY"),
    ("speech_cnn_tf_nemo", "own parallel CNN-Transformer (nEMO-6)", "speech", "pl",
     "anger,fear,happiness,sadness,surprise,neutral (no disgust)",
     "0.89 (_v2 held-out; orig notebook 0.85)", "own: _v2 held-out 0.887 (was 0.846); LEAKY"),
    ("en_speech_iemocap", "speechbrain wav2vec2 IEMOCAP", "speech", "en",
     "anger,happiness,sadness,neutral (no fear,disgust,surprise)",
     "0.787 acc (4-class)", "HF card: IEMOCAP test accuracy"),
    ("en_face_vit", "trpakov/vit-face-expression", "face", "language-agnostic",
     "7 canonical",
     "0.712 acc", "HF card: FER test-set accuracy (7-class)"),
]
COLS = ["model_id", "source", "modality", "language", "native_emotions",
        "author_accuracy", "accuracy_source"]

rows = [dict(zip(COLS, r)) for r in ROWS]
out = Settings().results_root / "tables"
out.mkdir(parents=True, exist_ok=True)

lines = ["# Models overview (author-claimed accuracy)", "",
         "> Accuracy = what each model's author reports on THEIR OWN test set — NOT comparable "
         "across models and NOT our accuracy on the study corpora. See `accuracy_source`.", "",
         "| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
for r in rows:
    lines.append("| " + " | ".join(r[c] for c in COLS) + " |")
md = "\n".join(lines) + "\n"
(out / "models_overview.md").write_text(md)
print(md)
print(f"-> {out}/models_overview.md")
