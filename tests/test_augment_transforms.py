import numpy as np
import pytest
from src.preprocessing.augment import apply_noise, apply_bandlimit, apply_codec, TRANSFORMS
from src.explain.acoustic_evidence import compute_evidence

SR = 16000


def _tone(freq=440.0, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rng():
    return np.random.default_rng(0)


def test_transforms_registry():
    assert set(TRANSFORMS) == {"codec", "noise", "bandlimit"}


@pytest.mark.parametrize("name", ["codec", "noise", "bandlimit"])
def test_length_dtype_preserved(name):
    audio = _tone()
    out = TRANSFORMS[name](audio, SR, _rng())
    assert out.shape == audio.shape
    assert out.dtype == np.float32
    assert not np.array_equal(out, audio)  # it must actually do something


def test_noise_raises_flatness():
    audio = _tone()
    clean = compute_evidence(audio, SR)
    noised = compute_evidence(apply_noise(audio, SR, _rng()), SR)
    assert noised["spectral_flatness"] > clean["spectral_flatness"]


def test_bandlimit_kills_high_band():
    rng_sig = np.random.default_rng(1)
    noise = (0.3 * rng_sig.standard_normal(2 * SR)).astype(np.float32)
    out = apply_bandlimit(noise, SR, _rng())
    ev = compute_evidence(out, SR)
    assert ev["high_band_energy_fraction"] < 0.05  # cutoffs are all <= 6 kHz


def test_codec_roundtrip_changes_signal_but_keeps_speech_band():
    audio = _tone()
    out = apply_codec(audio, SR, _rng())
    # energy survives (not silence), waveform differs (lossy codec)
    assert np.abs(out).max() > 0.05
    assert np.abs(out - audio).mean() > 1e-5


def test_deterministic_given_same_rng_seed():
    audio = _tone()
    a = apply_noise(audio, SR, np.random.default_rng(7))
    b = apply_noise(audio, SR, np.random.default_rng(7))
    assert np.array_equal(a, b)
