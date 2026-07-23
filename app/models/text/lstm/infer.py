"""Standalone Keras LSTM inference. Run with the TF venv (.venv-tf), NOT the main venv.

Reads {"texts": [...]} JSON on stdin, writes {"labels": [...], "probs": [[...]]} JSON on
stdout. Reconstructs the Keras Tokenizer from the saved word_index and reproduces the
training-time tokenization (default Tokenizer preprocessing, lowercased text, maxlen 30).

Usage: python infer.py <model_dir>     # model_dir has model.h5 + finalwordindex.pkl
Imports ONLY tensorflow/numpy/stdlib — never the `app` package (which pulls in torch).
"""
import json
import os
import pickle
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MAXLEN = 30
LABELS = ["neutral", "happiness", "sadness", "fear", "anger", "disgust", "surprise"]


def main() -> None:
    model_dir = sys.argv[1]
    payload = json.load(sys.stdin)
    texts = [str(t).lower() for t in payload["texts"]]

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import pad_sequences

    model = load_model(f"{model_dir}/model.h5")
    with open(f"{model_dir}/finalwordindex.pkl", "rb") as f:
        word_index = pickle.load(f)
    tokenizer = Tokenizer()          # default preprocessing (lower + punctuation filters), matches training
    tokenizer.word_index = word_index
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=MAXLEN)

    probs = model.predict(x, verbose=0)
    json.dump({"labels": LABELS, "probs": probs.tolist()}, sys.stdout)


if __name__ == "__main__":
    main()
