"""
Generate a tiny synthetic dataset using macOS `say` TTS, laid out exactly like
the three real raw datasets so manifest.py's builders run unchanged.

Bonafide = clean TTS render. Spoof = same render degraded (low-pass + noise +
bit-crush) so the classes are separable by construction. Engineering smoke
test only — no research validity.

Usage: .venv/bin/python scripts/make_smoke_data.py --out-dir data/raw_smoke
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

VOICES = ["Samantha", "Alex", "Daniel", "Karen"]  # 4 "speakers"
SENTENCES = [
    "The weather turned colder as the week went on.",
    "She placed the letter carefully on the wooden table.",
    "Bank transfers can take several days to complete.",
    "The committee will meet again on Thursday morning.",
    "He walked the long way home past the old harbor.",
    "Please confirm your appointment before noon tomorrow.",
    "The recipe calls for two cups of flour and one egg.",
    "Traffic on the bridge was heavier than usual today.",
    "Our flight departs from gate twenty two at nine.",
    "The museum opens late on the first Friday of the month.",
]
SR = 16000


def render_tts(text: str, voice: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        subprocess.run(["say", "-v", voice, "-o", tmp.name, text], check=True)
        audio, sr = sf.read(tmp.name, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def degrade(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sos = butter(6, 3200, btype="low", fs=SR, output="sos")  # kill the high band
    out = sosfilt(sos, audio).astype(np.float32)
    out = np.round(out * 128) / 128          # bit-crush -> quantization noise
    out += rng.standard_normal(len(out)).astype(np.float32) * 0.01
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def main(out_dir: Path):
    rng = np.random.default_rng(0)
    clips = []  # (voice, sent_idx, clean_audio)
    for voice in VOICES:
        for i, text in enumerate(SENTENCES):
            clips.append((voice, i, render_tts(text, voice)))
            print(f"Rendered {voice} #{i}")

    # asvspoof5-style (train domain): flac audio + protocol.txt
    a_audio = out_dir / "asvspoof5" / "audio"
    a_audio.mkdir(parents=True, exist_ok=True)
    proto_lines = []
    for voice, i, audio in clips:
        for kind in ("bonafide", "spoof"):
            fid = f"{voice}_{i}_{kind}"
            wav = audio if kind == "bonafide" else degrade(audio, rng)
            sf.write(a_audio / f"{fid}.flac", wav, SR)
            attack = "-" if kind == "bonafide" else "A01"
            proto_lines.append(f"{voice} {fid} - {attack} {kind}")
    (out_dir / "asvspoof5" / "protocol.txt").write_text("\n".join(proto_lines) + "\n")

    # p2v-style (eval): wav audio + metadata.csv — heavier degradation
    p_audio = out_dir / "p2v" / "audio"
    p_audio.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "p2v" / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "file_id", "label", "generation_method"])
        w.writeheader()
        for voice, i, audio in clips[:20]:
            for kind in ("real", "fake"):
                fid = f"p2v_{voice}_{i}_{kind}.wav"
                wav = audio if kind == "real" else degrade(degrade(audio, rng), rng)
                sf.write(p_audio / fid, wav, SR)
                w.writerow({"subject_id": voice, "file_id": fid,
                            "label": kind, "generation_method": "smoke"})

    # in_the_wild-style (eval): wav audio + meta.csv
    w_audio = out_dir / "in_the_wild" / "audio"
    w_audio.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "in_the_wild" / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader()
        for voice, i, audio in clips[20:]:
            for kind in ("bona-fide", "spoof"):
                fid = f"itw_{voice}_{i}_{kind.replace('-', '')}.wav"
                wav = audio if kind == "bona-fide" else degrade(audio, rng)
                sf.write(w_audio / fid, wav, SR)
                w.writerow({"file": fid, "speaker": voice, "label": kind})

    print(f"Smoke data written under {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/raw_smoke"))
    main(p.parse_args().out_dir)
