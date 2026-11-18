"""
Resample, VAD-trim, and chunk raw audio into fixed-length windows.
Caches processed chunks to data/processed/<dataset>/<original_stem>_<chunk_idx>.wav
alongside an updated manifest that references the chunks instead of full files.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import yaml

from src.provenance import check_resume, sha256_file, write_sidecar


@dataclass(frozen=True)
class PreprocessSettings:
    """Defaults equal today's CLI behavior; configs/default.yaml is the source
    of truth when passed via --config."""
    sample_rate: int = 16000
    chunk_seconds: float = 4.0
    chunk_stride_seconds: float | None = None
    vad_aggressiveness: int = 2
    vad_mode: str = "edges"
    pad_mode: str = "zero"
    vad_hangover_ms: int = 90


def resolve_settings(config_path: Path | None, overrides: dict) -> PreprocessSettings:
    """All fields from the config's `preprocessing:` section (when given),
    then any non-None override wins."""
    settings = PreprocessSettings()
    if config_path is not None:
        with open(config_path) as f:
            section = yaml.safe_load(f)["preprocessing"]
        names = {fl.name for fl in fields(PreprocessSettings)}
        settings = PreprocessSettings(**{k: v for k, v in section.items() if k in names})
    return replace(settings, **{k: v for k, v in overrides.items() if v is not None})


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
                      chunk_stride_seconds: float | None = None, *,
                      pad_mode: str = "zero", vad_mode: str = "edges",
                      vad_hangover_ms: int = 90, config_path: Path | None = None,
                      allow_legacy_dir: bool = False) -> None:
    """Arguments are the effective settings; config_path is only recorded.
    Writes provenance sidecars next to the output manifest and inside
    out_audio_dir, and refuses to write into a folder produced with
    different settings or a different input manifest."""
    settings = PreprocessSettings(sample_rate, chunk_seconds, chunk_stride_seconds,
                                  vad_aggressiveness, vad_mode, pad_mode, vad_hangover_ms)
    if pad_mode != "zero" or vad_mode != "edges":
        raise NotImplementedError(f"pad_mode={pad_mode!r} / vad_mode={vad_mode!r}")

    input_sha = sha256_file(manifest_path)
    current = {"settings": asdict(settings), "input_manifest_sha256": input_sha}
    has_outputs = out_audio_dir.exists() and any(out_audio_dir.glob("*.wav"))
    check_resume(out_audio_dir / "_meta.json", current, ["settings", "input_manifest_sha256"],
                 has_outputs=has_outputs, allow_legacy=allow_legacy_dir)

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
            sf.write(out_path, chunk, sample_rate, subtype="PCM_16")
            out_rows.append({**row, "path": str(out_path), "chunk_idx": j})

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(rows)} source files")

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} chunk rows to {out_manifest_path}")

    meta = {**current, "stage": "preprocess",
            "config_path": str(config_path) if config_path is not None else None,
            "input_manifest": str(manifest_path),
            "output_manifest_sha256": sha256_file(out_manifest_path),
            "n_source_rows": len(rows), "n_chunks": len(out_rows)}
    write_sidecar(Path(str(out_manifest_path) + ".meta.json"), meta)
    write_sidecar(out_audio_dir / "_meta.json", meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-audio-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="take every preprocessing setting from this config's "
                             "`preprocessing:` section; explicit flags below override")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--chunk-stride-seconds", type=float, default=None,
                        help="sliding-window stride; without --config defaults to chunk "
                             "length (no overlap)")
    parser.add_argument("--vad-aggressiveness", type=int, default=None)
    parser.add_argument("--pad-mode", choices=["zero", "repeat"], default=None)
    parser.add_argument("--vad-mode", choices=["edges", "all"], default=None)
    parser.add_argument("--vad-hangover-ms", type=int, default=None)
    parser.add_argument("--allow-legacy-dir", action="store_true")
    args = parser.parse_args()

    s = resolve_settings(args.config, {
        "sample_rate": args.sample_rate, "chunk_seconds": args.chunk_seconds,
        "chunk_stride_seconds": args.chunk_stride_seconds,
        "vad_aggressiveness": args.vad_aggressiveness, "pad_mode": args.pad_mode,
        "vad_mode": args.vad_mode, "vad_hangover_ms": args.vad_hangover_ms,
    })
    process_manifest(
        args.manifest, args.out_audio_dir, args.out_manifest,
        s.sample_rate, s.chunk_seconds, s.vad_aggressiveness,
        chunk_stride_seconds=s.chunk_stride_seconds, pad_mode=s.pad_mode,
        vad_mode=s.vad_mode, vad_hangover_ms=s.vad_hangover_ms,
        config_path=args.config, allow_legacy_dir=args.allow_legacy_dir,
    )
