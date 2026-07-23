"""Audio feature pipeline for the custom CNN-Transformer speech model.

Faithfully reproduces the training pipeline: resample to 48 kHz, take a 3 s
window starting at 0.5 s (zero-padded), then a log-mel spectrogram (128 x 563).
"""
import librosa
import numpy as np

SAMPLE_RATE = 48000
OFFSET_S = 0.5
DURATION_S = 3
TARGET_LEN = SAMPLE_RATE * DURATION_S  # 144000


def to_model_signal(audio: np.ndarray, native_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")
    if native_sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SAMPLE_RATE)
    start = int(OFFSET_S * SAMPLE_RATE)
    window = audio[start:start + TARGET_LEN]
    signal = np.zeros(TARGET_LEN, dtype="float32")
    signal[:len(window)] = window
    return signal


def mel_spectrogram(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=1024, win_length=512, window="hamming",
        hop_length=256, n_mels=128, fmax=sr / 2,
    )
    return librosa.power_to_db(mel, ref=np.max).astype("float32")
