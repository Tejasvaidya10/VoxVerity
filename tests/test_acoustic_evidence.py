import numpy as np
from src.explain.acoustic_evidence import compute_evidence

SR = 16000
KEYS = {"pitch_variance", "spectral_flatness", "spectral_rolloff_95_hz",
        "unvoiced_energy_ratio", "high_band_energy_fraction"}


def _tone(freq=220.0, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds=2.0):
    rng = np.random.default_rng(0)
    return (0.5 * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def test_returns_all_keys_as_floats():
    ev = compute_evidence(_tone(), SR)
    assert set(ev) == KEYS
    assert all(isinstance(v, float) for v in ev.values())


def test_noise_flatter_and_brighter_than_tone():
    tone, noise = compute_evidence(_tone(), SR), compute_evidence(_noise(), SR)
    assert noise["spectral_flatness"] > tone["spectral_flatness"]
    assert noise["spectral_rolloff_95_hz"] > tone["spectral_rolloff_95_hz"]
    assert noise["high_band_energy_fraction"] > tone["high_band_energy_fraction"]


def test_steady_tone_has_low_pitch_variance():
    ev = compute_evidence(_tone(), SR)
    assert ev["pitch_variance"] < 100.0  # Hz^2; steady 220Hz tone barely wobbles


def test_no_nan_on_silence():
    ev = compute_evidence(np.zeros(SR, dtype=np.float32), SR)
    assert all(np.isfinite(v) for v in ev.values())
