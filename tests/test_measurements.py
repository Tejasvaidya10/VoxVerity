import csv
import math
from pathlib import Path

import numpy as np
import pytest

from src.analysis.measurements import (CLIP_FIELDS, characterize_manifest, measure_clip,
                                       measure_windows, pause_stats, snr_db)
from src.preprocessing.preprocess import PreprocessSettings
from src.provenance import ProvenanceMismatch

FIX = Path(__file__).parent / "fixtures" / "preprocess_regression"
S = PreprocessSettings(chunk_stride_seconds=2.0)  # July settings
SR = 16000


def test_pause_stats_internal_only():
    f = [False] * 5 + [True] * 10 + [False] * 12 + [True] * 5 + [False] * 20
    n, longest = pause_stats(f)
    assert n == 1 and longest == pytest.approx(12 * 0.03)


def test_pause_stats_short_gap_not_counted_but_longest_reported():
    f = [True] * 3 + [False] * 5 + [True] * 3
    n, longest = pause_stats(f)
    assert n == 0 and longest == pytest.approx(5 * 0.03)


def test_pause_stats_no_speech():
    assert pause_stats([False] * 40) == (0, 0.0)


def test_snr_lower_with_more_noise():
    rng = np.random.default_rng(0)
    t = np.arange(4 * SR) / SR
    sig = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    quiet = sig + 0.001 * rng.standard_normal(len(t)).astype(np.float32)
    noisy = sig + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
    assert snr_db(quiet, SR) > snr_db(noisy, SR) + 10


def test_measure_clip_fields_and_durations():
    m = measure_clip(str(FIX / "pause.wav"), S)
    assert list(m) == CLIP_FIELDS
    assert m["duration_raw_s"] == pytest.approx(91200 / SR)
    assert m["duration_trimmed_s"] == pytest.approx(74400 / SR)  # pinned by the regression test
    assert m["duration_speech_only_s"] < m["duration_trimmed_s"]
    assert m["n_pauses"] == 1 and m["longest_pause_s"] >= 0.8
    assert 0.5 < m["speech_ratio"] < 1.0
    assert 120 < m["f0_median_hz"] < 170  # fixture F0 = 140 +/- 20 Hz


def test_measure_clip_no_speech():
    m = measure_clip(str(FIX / "nospeech_2s.wav"), S)
    assert m["speech_ratio"] == 0.0 and m["n_pauses"] == 0
    assert math.isnan(m["f0_median_hz"])
    assert math.isnan(m["snr_db"])  # digital silence: SNR undefined, not -inf


def test_measure_windows_short_clip_padded():
    (w,) = measure_windows(str(FIX / "short.wav"), S)
    assert w["chunk_idx"] == 0
    assert w["pad_fraction"] == pytest.approx(1 - 41760 / 64000)
    assert w["window_speech_ratio"] < 1 - 0.3  # padding counts as non-speech


def test_measure_windows_matches_chunk_grid():
    rows = measure_windows(str(FIX / "long.wav"), S)
    assert [r["chunk_idx"] for r in rows] == [0, 1, 2, 3]
    assert all(r["pad_fraction"] == 0.0 for r in rows)


def _manifest(tmp_path, clips):
    m = tmp_path / "raw.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "speaker_id", "dataset_source", "attack_type"])
        w.writeheader()
        for c in clips:
            w.writerow({"path": str(FIX / f"{c}.wav"), "label": "bonafide", "speaker_id": f"s_{c}",
                        "dataset_source": "x", "attack_type": "-"})
    return m


def test_characterize_writes_and_resumes_after_crash(tmp_path):
    out = tmp_path / "meas"
    m = _manifest(tmp_path, ["short", "long"])
    characterize_manifest(m, "ds", out, S, workers=1)
    # simulate a crash after "long"'s window rows were written but before its clip row
    clips_path = out / "ds_clips.csv"
    lines = clips_path.read_text().splitlines(keepends=True)
    clips_path.write_text("".join(l for l in lines if not l.startswith("long,")))
    characterize_manifest(m, "ds", out, S, workers=1)
    clips = list(csv.DictReader(open(clips_path)))
    wins = list(csv.DictReader(open(out / "ds_windows.csv")))
    assert sorted(r["clip_id"] for r in clips) == ["long", "short"]
    assert sorted((r["clip_id"], r["chunk_idx"]) for r in wins) == [
        ("long", "0"), ("long", "1"), ("long", "2"), ("long", "3"), ("short", "0")]
    assert clips[0]["speaker_id"].startswith("s_") and clips[0]["label"] == "bonafide"


def test_characterize_refuses_different_settings(tmp_path):
    out = tmp_path / "meas"
    m = _manifest(tmp_path, ["short"])
    characterize_manifest(m, "ds", out, S, workers=1)
    with pytest.raises(ProvenanceMismatch, match="settings"):
        characterize_manifest(m, "ds", out, PreprocessSettings(chunk_stride_seconds=4.0), workers=1)


def test_characterize_parallel_matches_serial(tmp_path):
    m = _manifest(tmp_path, ["short", "long", "pause"])
    characterize_manifest(m, "ds", tmp_path / "serial", S, workers=1)
    characterize_manifest(m, "ds", tmp_path / "par", S, workers=2)
    key = lambda p: sorted(csv.DictReader(open(p)), key=lambda r: r["clip_id"])
    assert key(tmp_path / "serial" / "ds_clips.csv") == key(tmp_path / "par" / "ds_clips.csv")
