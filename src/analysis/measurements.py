"""
Phase 5 measurements: per-clip properties (duration, speech share, pauses,
SNR, bandwidth, pitch, level) and per-window properties (padding share,
speech share) for comparing In-the-Wild with ASVspoof5 and for plotting
detector errors against them.
"""

import csv
import math
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np

from src.preprocessing.preprocess import (VAD_FRAME_MS, PreprocessSettings, chunk_audio,
                                          load_and_resample, vad_keep_speech, vad_trim,
                                          voiced_flags)
from src.provenance import check_resume, sha256_file, write_sidecar

CLIP_FIELDS = ["duration_raw_s", "duration_trimmed_s", "duration_speech_only_s", "speech_ratio",
               "n_pauses", "longest_pause_s", "snr_db", "rolloff95_hz", "f0_median_hz",
               "rms_dbfs", "clip_frac"]
WINDOW_FIELDS = ["chunk_idx", "pad_fraction", "window_speech_ratio"]
ID_FIELDS = ["clip_id", "label", "speaker_id"]


def snr_db(audio: np.ndarray, sr: int) -> float:
    """Percentile SNR proxy: 90th vs 10th percentile of 30 ms frame power.
    Only used to rank clips (terciles), so calibration doesn't matter.
    Digital silence gives a very high value; that is recorded, not clipped."""
    n = int(sr * VAD_FRAME_MS / 1000)
    k = len(audio) // n
    if k == 0:
        return float("nan")
    power = (audio[:k * n].astype(np.float64).reshape(k, n) ** 2).mean(axis=1)
    p90 = np.percentile(power, 90)
    if p90 <= 1e-10:  # no signal at all: SNR undefined
        return float("nan")
    return float(10 * np.log10(p90 / max(np.percentile(power, 10), 1e-10)))


def pause_stats(flags: list, frame_ms: int = VAD_FRAME_MS, min_pause_ms: int = 300) -> tuple:
    """Internal pauses only (between the first and last voiced frame).
    Returns (number of non-voiced runs >= min_pause_ms, longest internal
    non-voiced run in seconds, of any length)."""
    voiced = np.flatnonzero(flags)
    if len(voiced) == 0:
        return 0, 0.0
    runs, run = [], 0
    for v in flags[voiced[0]:voiced[-1] + 1]:
        if v:
            if run:
                runs.append(run)
            run = 0
        else:
            run += 1
    n = sum(1 for r in runs if r * frame_ms >= min_pause_ms)
    return n, (max(runs) * frame_ms / 1000 if runs else 0.0)


def _voiced_median(values: np.ndarray, flags: list) -> float:
    m = min(len(values), len(flags))
    sel = values[:m][np.asarray(flags[:m], dtype=bool)]
    sel = sel[np.isfinite(sel)]
    return float(np.median(sel)) if sel.size else float("nan")


def measure_clip(path: str, settings: PreprocessSettings) -> dict:
    sr, aggr = settings.sample_rate, settings.vad_aggressiveness
    hop = int(sr * VAD_FRAME_MS / 1000)  # 480: STFT/yin frame i ~ VAD frame i
    raw = load_and_resample(path, sr)
    trimmed = vad_trim(raw, sr, aggr)
    speech_only = vad_keep_speech(raw, sr, aggr, settings.vad_hangover_ms)
    flags_trim = voiced_flags(trimmed, sr, aggr)
    flags_raw = voiced_flags(raw, sr, aggr)
    n_pauses, longest = pause_stats(flags_trim)

    if len(raw) >= 1024:
        S = np.abs(librosa.stft(raw, n_fft=1024, hop_length=hop))
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
        ro = _voiced_median(rolloff, flags_raw) if any(flags_raw) else float(np.median(rolloff))
        f0 = librosa.yin(raw, fmin=60.0, fmax=500.0, sr=sr, frame_length=1024, hop_length=hop)
        f0_med = _voiced_median(f0, flags_raw)
    else:
        ro, f0_med = float("nan"), float("nan")

    return {
        "duration_raw_s": len(raw) / sr,
        "duration_trimmed_s": len(trimmed) / sr,
        "duration_speech_only_s": len(speech_only) / sr,
        "speech_ratio": float(np.mean(flags_trim)) if flags_trim else 0.0,
        "n_pauses": n_pauses,
        "longest_pause_s": longest,
        "snr_db": snr_db(raw, sr),
        "rolloff95_hz": ro,
        "f0_median_hz": f0_med,
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(raw.astype(np.float64) ** 2)) + 1e-10)),
        "clip_frac": float(np.mean(np.abs(raw) >= 0.999)) if len(raw) else 0.0,
    }


def measure_windows(path: str, settings: PreprocessSettings) -> list:
    """One row per window of the default (edges, zero-pad) grid, matching
    the baseline chunk manifest by (clip_id, chunk_idx)."""
    sr, aggr = settings.sample_rate, settings.vad_aggressiveness
    trimmed = vad_trim(load_and_resample(path, sr), sr, aggr)
    chunk_len = int(sr * settings.chunk_seconds)
    windows = chunk_audio(trimmed, sr, settings.chunk_seconds, settings.chunk_stride_seconds)
    pad = (chunk_len - len(trimmed)) / chunk_len if len(trimmed) < chunk_len else 0.0
    rows = []
    for j, w in enumerate(windows):
        flags = voiced_flags(w, sr, aggr)
        rows.append({"chunk_idx": j, "pad_fraction": pad if j == 0 else 0.0,
                     "window_speech_ratio": float(np.mean(flags)) if flags else 0.0})
    return rows


def _measure_row(args):
    row, settings = args
    clip_id = Path(row["path"]).stem
    try:
        clip = {"clip_id": clip_id, "label": row["label"], "speaker_id": row["speaker_id"],
                **measure_clip(row["path"], settings)}
        wins = [{"clip_id": clip_id, **w} for w in measure_windows(row["path"], settings)]
        return clip, wins, None
    except Exception as e:  # unreadable file: report and continue
        return None, None, f"{row['path']}: {e}"


def characterize_manifest(manifest_path: Path, dataset: str, out_dir: Path,
                          settings: PreprocessSettings, workers: int = 6) -> None:
    """Writes <dataset>_clips.csv and <dataset>_windows.csv. Resumable: a
    clip's window rows are written before its clip row, so the clip row is
    the commit marker; on restart, half-written window rows are dropped and
    committed clips are skipped. Refuses to resume with different settings
    or a different input manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_path, wins_path = out_dir / f"{dataset}_clips.csv", out_dir / f"{dataset}_windows.csv"
    meta = {"stage": "measurements", "settings": asdict(settings),
            "input_manifest": str(manifest_path), "input_manifest_sha256": sha256_file(manifest_path)}
    check_resume(out_dir / f"{dataset}_measurements.meta.json", meta,
                 ["settings", "input_manifest_sha256"], has_outputs=clips_path.exists(),
                 allow_legacy=False)
    write_sidecar(out_dir / f"{dataset}_measurements.meta.json", meta)

    done = set()
    if clips_path.exists():
        done = {r["clip_id"] for r in csv.DictReader(open(clips_path))}
    if wins_path.exists():  # drop window rows of clips that never committed
        kept = [r for r in csv.DictReader(open(wins_path)) if r["clip_id"] in done]
        with open(wins_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clip_id"] + WINDOW_FIELDS)
            w.writeheader()
            w.writerows(kept)

    with open(manifest_path) as f:
        todo = [r for r in csv.DictReader(f) if Path(r["path"]).stem not in done]
    print(f"{dataset}: {len(done)} clips already measured, {len(todo)} to go", flush=True)

    new_clips = not clips_path.exists()
    new_wins = not wins_path.exists()
    with open(clips_path, "a", newline="") as cf, open(wins_path, "a", newline="") as wf:
        cw = csv.DictWriter(cf, fieldnames=ID_FIELDS + CLIP_FIELDS)
        ww = csv.DictWriter(wf, fieldnames=["clip_id"] + WINDOW_FIELDS)
        if new_clips:
            cw.writeheader()
        if new_wins:
            ww.writeheader()
        jobs = [(r, settings) for r in todo]
        pool = Pool(workers) if workers > 1 else None
        try:
            results = (pool.imap_unordered(_measure_row, jobs, chunksize=16) if pool
                       else map(_measure_row, jobs))
            for i, (clip, wins, err) in enumerate(results, 1):
                if err:
                    print(f"Skipping {err}", flush=True)
                    continue
                ww.writerows(wins)
                wf.flush()
                cw.writerow(clip)
                cf.flush()
                if i % 2000 == 0:
                    print(f"{dataset}: {i}/{len(todo)} measured", flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
    print(f"{dataset}: done", flush=True)
