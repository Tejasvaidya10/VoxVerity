"""
Channel augmentation: decorrelate recording quality from labels.

Each transform simulates a real-world channel (lossy codec, noise floor,
band-limited link) and is applied label-independently, so channel quality
stops predicting bonafide-vs-spoof. See
docs/superpowers/specs/2026-07-17-channel-augmentation-design.md.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

FFMPEG = "ffmpeg"
CODECS = [("libmp3lame", "32k", ".mp3"), ("libmp3lame", "64k", ".mp3"),
          ("libmp3lame", "96k", ".mp3"), ("libopus", "16k", ".opus"),
          ("libopus", "32k", ".opus"), ("libopus", "48k", ".opus")]


def _match_length(out: np.ndarray, n: int) -> np.ndarray:
    if len(out) >= n:
        return out[:n]
    padded = np.zeros(n, dtype=np.float32)
    padded[:len(out)] = out
    return padded


def apply_codec(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    codec, bitrate, ext = CODECS[rng.integers(len(CODECS))]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        enc = Path(td) / f"enc{ext}"
        dec = Path(td) / "out.wav"
        sf.write(src, audio, sr, subtype="PCM_16")
        for cmd in (
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
             "-c:a", codec, "-b:a", bitrate, str(enc)],
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(enc),
             "-ar", str(sr), "-ac", "1", str(dec)],
        ):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed ({codec} {bitrate}): {proc.stderr.strip()}")
        out, out_sr = sf.read(dec, dtype="float32")
    assert out_sr == sr
    if out.ndim > 1:
        out = out.mean(axis=1)
    return _match_length(out.astype(np.float32), len(audio))


def apply_noise(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    snr_db = rng.uniform(5.0, 25.0)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    if rng.random() < 0.5:  # pink: 1/f spectral shaping
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1 / sr)
        spec[1:] /= np.sqrt(freqs[1:])
        noise = np.fft.irfft(spec, n=len(noise)).astype(np.float32)
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    out = audio + scale * noise
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def apply_bandlimit(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:  # telephone band-pass
        sos = butter(6, [300, 3400], btype="bandpass", fs=sr, output="sos")
    else:                    # low-pass with random cutoff
        cutoff = rng.uniform(3400.0, 6000.0)
        sos = butter(8, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


TRANSFORMS = {"codec": apply_codec, "noise": apply_noise, "bandlimit": apply_bandlimit}
