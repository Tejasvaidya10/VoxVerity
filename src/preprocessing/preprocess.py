"""
Resample, VAD-trim, and chunk raw audio into fixed-length windows.
Caches processed chunks to data/processed/<dataset>/<original_stem>_<chunk_idx>.wav
alongside an updated manifest that references the chunks instead of full files.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import webrtcvad


def load_and_resample(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def vad_trim(audio: np.ndarray, sr: int, aggressiveness: int) -> np.ndarray:
    """Drop leading/trailing silence using WebRTC VAD. Operates on 16kHz, 16-bit PCM."""
    assert sr == 16000, "webrtcvad requires 16kHz input"
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    pcm16 = (audio * 32768).astype(np.int16).tobytes()

    voiced_frames = []
    for start in range(0, len(pcm16) - frame_len * 2, frame_len * 2):
        frame = pcm16[start:start + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        is_voiced = vad.is_speech(frame, sr)
        voiced_frames.append(is_voiced)

    if not any(voiced_frames):
        return audio  # fall back to original if VAD finds nothing

    first = voiced_frames.index(True)
    last = len(voiced_frames) - 1 - voiced_frames[::-1].index(True)
    start_sample = first * frame_len
    end_sample = min((last + 1) * frame_len, len(audio))
    return audio[start_sample:end_sample]


def chunk_audio(audio: np.ndarray, sr: int, chunk_seconds: float,
                stride_seconds: float | None = None) -> list:
    """Sliding-window chunking. Overlap keeps boundary artifacts fully
    interior to at least one window (generative glitches often sit at
    seams a hard slice would bisect)."""
    chunk_len = int(sr * chunk_seconds)
    stride = int(sr * (stride_seconds if stride_seconds else chunk_seconds))
    if len(audio) <= chunk_len:
        # pad short clips to full chunk length
        padded = np.zeros(chunk_len, dtype=np.float32)
        padded[:len(audio)] = audio
        return [padded]
    starts = list(range(0, len(audio) - chunk_len + 1, stride))
    last_start = len(audio) - chunk_len
    if starts[-1] != last_start:
        starts.append(last_start)  # end-aligned window so the tail is covered
    return [audio[s:s + chunk_len] for s in starts]


def process_manifest(manifest_path: Path, out_audio_dir: Path, out_manifest_path: Path,
                      sample_rate: int, chunk_seconds: float, vad_aggressiveness: int,
                      chunk_stride_seconds: float | None = None) -> None:
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for i, row in enumerate(rows):
        try:
            audio = load_and_resample(row["path"], sample_rate)
            audio = vad_trim(audio, sample_rate, vad_aggressiveness)
            chunks = chunk_audio(audio, sample_rate, chunk_seconds, chunk_stride_seconds)
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
            continue

        stem = Path(row["path"]).stem
        for j, chunk in enumerate(chunks):
            out_path = out_audio_dir / f"{stem}_{j}.wav"
            sf.write(out_path, chunk, sample_rate)
            out_rows.append({**row, "path": str(out_path), "chunk_idx": j})

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(rows)} source files")

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} chunk rows to {out_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-audio-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--chunk-stride-seconds", type=float, default=None,
                        help="sliding-window stride; defaults to chunk length (no overlap)")
    parser.add_argument("--vad-aggressiveness", type=int, default=2)
    args = parser.parse_args()

    process_manifest(
        args.manifest, args.out_audio_dir, args.out_manifest,
        args.sample_rate, args.chunk_seconds, args.vad_aggressiveness,
        chunk_stride_seconds=args.chunk_stride_seconds,
    )
