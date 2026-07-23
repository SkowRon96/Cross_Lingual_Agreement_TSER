import numpy as np
from app.models.speech.features import to_model_signal, mel_spectrogram, SAMPLE_RATE


def test_to_model_signal_length_and_resample():
    # 5s of 24kHz audio -> resampled to 48kHz, offset 0.5s + 3s window, padded to 144000
    native_sr = 24000
    audio = np.sin(2 * np.pi * 220 * np.arange(5 * native_sr) / native_sr).astype("float32")
    sig = to_model_signal(audio, native_sr)
    assert sig.dtype == np.float32
    assert sig.shape == (SAMPLE_RATE * 3,)   # 144000


def test_to_model_signal_pads_short_audio():
    native_sr = 48000
    audio = np.ones(int(0.7 * native_sr), dtype="float32")  # only 0.7s; after 0.5s offset -> 0.2s left
    sig = to_model_signal(audio, native_sr)
    assert sig.shape == (144000,)
    assert np.count_nonzero(sig) <= int(0.2 * native_sr) + 1  # mostly zero-padded


def test_mel_spectrogram_shape():
    sig = np.random.randn(144000).astype("float32")
    mel = mel_spectrogram(sig)
    assert mel.shape == (128, 563)
    assert mel.dtype == np.float32
