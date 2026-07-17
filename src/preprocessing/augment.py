"""
Channel augmentation: decorrelate recording quality from labels.

Each transform simulates a real-world channel (lossy codec, noise floor,
band-limited link) and is applied label-independently, so channel quality
stops predicting bonafide-vs-spoof. See
docs/superpowers/specs/2026-07-17-channel-augmentation-design.md.
"""

import argparse
import csv
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


def augment_manifest(manifest_path: Path, out_audio_dir: Path,
                     out_manifest_path: Path, seed: int = 42,
                     augment_prob: float = 0.5) -> None:
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    names = sorted(TRANSFORMS)
    out_rows, counts, failures = [], {"none": 0}, 0
    for idx, row in enumerate(rows):
        rng = np.random.default_rng([seed, idx])
        if rng.random() >= augment_prob:
            out_rows.append({**row, "augmentation": "none"})
            counts["none"] += 1
            continue
        name = names[rng.integers(len(names))]
        try:
            audio, sr = sf.read(row["path"], dtype="float32")
            out = TRANSFORMS[name](np.asarray(audio), sr, rng)
        except Exception as e:
            if name == "codec":  # spec: codec failures fall back to noise
                try:
                    name = "noise"
                    out = apply_noise(np.asarray(audio), sr, rng)
                except Exception:
                    failures += 1
                    out_rows.append({**row, "augmentation": "none"})
                    continue
            else:
                failures += 1
                print(f"Augment failed for {row['path']}: {e}")
                out_rows.append({**row, "augmentation": "none"})
                continue
        out_path = out_audio_dir / f"{Path(row['path']).stem}__aug_{name}.wav"
        sf.write(out_path, out, sr, subtype="PCM_16")
        out_rows.append({**row, "path": str(out_path), "augmentation": name})
        counts[name] = counts.get(name, 0) + 1

        if (idx + 1) % 2000 == 0:
            print(f"Augmented {idx + 1}/{len(rows)}", flush=True)

    fieldnames = list(rows[0].keys()) + ["augmentation"]
    with open(out_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out_manifest_path}; "
          f"counts={counts}; failures={failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-audio-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-prob", type=float, default=0.5)
    args = parser.parse_args()
    augment_manifest(args.manifest, args.out_audio_dir, args.out_manifest,
                     seed=args.seed, augment_prob=args.augment_prob)
