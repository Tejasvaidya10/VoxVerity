"""Generate the WAV fixtures pinned by tests/test_preprocess_regression.py.

Run once from the repo root; the WAVs are committed so a library change
can't silently alter the test input:

    .venv/bin/python tests/fixtures/preprocess_regression/make_fixture.py

The "speech" is a harmonic complex with syllable-rate amplitude modulation,
checked here to trigger WebRTC VAD at aggressiveness 2.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.preprocessing.preprocess import vad_trim  # noqa: E402

SR = 16000
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SR)), dtype=np.float32)


def speech(seconds: float) -> np.ndarray:
    t = np.arange(int(round(seconds * SR))) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 25))
    x = x * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
    x = 0.25 * x / np.abs(x).max() + 0.003 * rng.standard_normal(len(t))
    return x.astype(np.float32)


CLIPS = {
    "pause": np.concatenate([silence(0.5), speech(1.5), silence(1.0), speech(2.0), silence(0.7)]),
    "short": np.concatenate([silence(0.3), speech(2.5), silence(0.3)]),
    "long": speech(9.3),
    "nospeech_2s": silence(2.0),                         # VAD fall-back, then zero-padded
    "nospeech_exact4s": np.zeros(64000, dtype=np.float32),  # only way to be exactly 4.0 s after trim
}


def main():
    for name, audio in CLIPS.items():
        trimmed = vad_trim(audio, SR, 2)
        if name.startswith("nospeech"):
            assert len(trimmed) == len(audio), f"{name}: expected VAD fall-back"
        else:
            assert len(trimmed) < len(audio), f"{name}: VAD found no speech; fixture invalid"
        if name == "pause":
            quiet = np.abs(trimmed) < 1e-2
            run = best = 0
            for q in quiet:
                run = run + 1 if q else 0
                best = max(best, run)
            assert best >= int(0.9 * SR), "pause: internal pause was not kept by edge trim"
        sf.write(OUT / f"{name}.wav", audio, SR, subtype="PCM_16")
        print(f"{name}.wav: {len(audio)} samples, trimmed to {len(trimmed)}")


if __name__ == "__main__":
    main()
