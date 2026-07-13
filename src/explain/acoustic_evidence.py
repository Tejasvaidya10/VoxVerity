"""
Acoustic descriptors computed per flagged clip so LLM rationales are grounded
in real numbers instead of placeholders. Beyond pitch/flatness (which high-end
deepfakes pass), we probe where vocoder artifacts hide: unvoiced segments
(breaths, silences) and the high band (>4 kHz shelf/smear).
"""

import numpy as np
import soundfile as sf
import librosa

HIGH_BAND_HZ = 4000.0


def compute_evidence(audio: np.ndarray, sr: int) -> dict:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    eps = 1e-10

    stft_mag = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    power = stft_mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft_mag)))
    rolloff = float(np.mean(
        librosa.feature.spectral_rolloff(S=stft_mag, sr=sr, roll_percent=0.95)))

    total_power = float(power.sum()) + eps
    high_band = float(power[freqs >= HIGH_BAND_HZ].sum()) / total_power

    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=60.0, fmax=500.0, sr=sr, frame_length=1024, hop_length=256)
    voiced_f0 = f0[np.asarray(voiced_flag, dtype=bool)] if f0 is not None else np.array([])
    voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
    pitch_variance = float(np.var(voiced_f0)) if voiced_f0.size >= 2 else 0.0

    frame_energy = power.sum(axis=0)
    n = min(len(frame_energy), len(voiced_flag)) if voiced_flag is not None else 0
    if n > 0:
        unvoiced = ~np.asarray(voiced_flag[:n], dtype=bool)
        unvoiced_ratio = float(frame_energy[:n][unvoiced].sum() / (frame_energy[:n].sum() + eps))
    else:
        unvoiced_ratio = 1.0

    return {
        "pitch_variance": pitch_variance,
        "spectral_flatness": flatness,
        "spectral_rolloff_95_hz": rolloff,
        "unvoiced_energy_ratio": unvoiced_ratio,
        "high_band_energy_fraction": high_band,
    }


def compute_evidence_from_file(path: str) -> dict:
    audio, sr = sf.read(path, dtype="float32")
    return compute_evidence(np.asarray(audio), sr)
