"""Regression test pinning today's default preprocessing output.

Characterization test: the expected values below were captured from the
code at commit cc2ae54 (the code that produced Phases 1-4), so any later
change to preprocessing must leave default output byte-identical.
Structure is checked before checksums so a failure says what changed.
Regenerate the fixture only deliberately (make_fixture.py).
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from src.preprocessing.preprocess import load_and_resample, process_manifest, vad_trim

SR = 16000
WINDOW = 64000  # 4.0 s
FIX = Path(__file__).parent / "fixtures" / "preprocess_regression"
CLIPS = ["pause", "short", "long", "nospeech_2s", "nospeech_exact4s"]
COLUMNS = ["path", "label", "speaker_id", "dataset_source", "attack_type", "chunk_idx"]

# Hand-derived from the fixture layout (make_fixture.py) and chunk_audio's
# 4 s / 2 s grid with an end-aligned tail window.
EXPECTED_WINDOWS = {"pause": 2, "short": 1, "long": 4, "nospeech_2s": 1, "nospeech_exact4s": 1}

# Trimmed lengths in samples. nospeech_* are hand-derived (VAD fall-back
# returns the input untouched); the rest were captured from cc2ae54.
EXPECTED_TRIMMED = {
    "pause": 74400,               # 155 VAD frames
    "short": 41760,               # 87 frames
    "long": 148320,               # 309 of 310 frames: the final frame is never examined
    "nospeech_2s": 32000,         # fall-back, hand-derived
    "nospeech_exact4s": 64000,    # fall-back, hand-derived; exactly one window, no padding
}

# SHA-256 of each window's int16 bytes, in manifest order, captured from cc2ae54.
EXPECTED_SHA256 = {
    "pause": ["0f5d1ebc3900fcf76046a17aed864445afcbeb8bb07e993fdbf08a9a378b8ffd",
              "ed843fe1b92e2fc4fe42de6bf80edd88167020172f7133ce5b55051980cd39b6"],
    "short": ["7f6c93d8c2ac655d81fec1fa4c43aff155e03daca7bc0dcd9cab4ed37e603aea"],
    "long": ["f7789f64c59ef6e8d600e1b14b3afb267bb784cf7a3a0f3e6ef45ac208ae068e",
             "35ef96a723025eabbc1a4f4698a5bfb5a2db87b8d135268dae24c45ffe3271bd",
             "7cd6efcf130219e1f8f8665215a6c4965b65a22a90c6f0e38101914aa5b01b43",
             "c3d6b6d138657edd256c406b0b42151b91afca4b178a9c348a445fee2d5cf804"],
    "nospeech_2s": ["eec19bc6af0b3b6dfb97a08782c65f4bb3c3203e789a015d2008b0d689ad08be"],
    "nospeech_exact4s": ["eec19bc6af0b3b6dfb97a08782c65f4bb3c3203e789a015d2008b0d689ad08be"],
}


def _input_manifest(tmp_path: Path) -> Path:
    rows = [{"path": str(FIX / f"{c}.wav"), "label": "spoof" if i % 2 else "bonafide",
             "speaker_id": f"spk{i}", "dataset_source": "fixture", "attack_type": f"A{i}"}
            for i, c in enumerate(CLIPS)]
    m = tmp_path / "in.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return m


def _run(tmp_path: Path) -> list:
    out_manifest = tmp_path / "out" / "chunks.csv"
    process_manifest(_input_manifest(tmp_path), tmp_path / "out" / "audio", out_manifest,
                     16000, 4.0, 2, chunk_stride_seconds=2.0)
    with open(out_manifest) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def test_vad_trim_lengths_pinned():
    for c in CLIPS:
        trimmed = vad_trim(load_and_resample(str(FIX / f"{c}.wav"), SR), SR, 2)
        assert len(trimmed) == EXPECTED_TRIMMED[c], f"{c}: trimmed length changed"


def test_internal_pause_kept():
    trimmed = vad_trim(load_and_resample(str(FIX / "pause.wav"), SR), SR, 2)
    run = best = 0
    for quiet in np.abs(trimmed) < 1e-2:
        run = run + 1 if quiet else 0
        best = max(best, run)
    assert best >= int(0.9 * SR), "edge-only trim should keep the 1.0 s internal pause"


def test_process_manifest_pinned(tmp_path):
    rows = _run(tmp_path)

    # 1) row order: clips in input order, chunk_idx 0..n-1, metadata copied
    expected_order = [(c, j) for c in CLIPS for j in range(EXPECTED_WINDOWS[c])]
    got_order = [(Path(r["path"]).stem.rsplit("_", 1)[0], int(r["chunk_idx"])) for r in rows]
    assert got_order == expected_order
    for r in rows:
        i = CLIPS.index(Path(r["path"]).stem.rsplit("_", 1)[0])
        assert (r["speaker_id"], r["attack_type"], r["dataset_source"]) == (f"spk{i}", f"A{i}", "fixture")

    windows = {c: [] for c in CLIPS}
    for r in rows:
        audio, sr = sf.read(r["path"], dtype="int16")
        assert sr == SR
        windows[Path(r["path"]).stem.rsplit("_", 1)[0]].append(audio)

    # 2) window count and length
    for c in CLIPS:
        assert len(windows[c]) == EXPECTED_WINDOWS[c], f"{c}: window count changed"
        assert all(len(w) == WINDOW for w in windows[c]), f"{c}: window length changed"

    # 3) single-window clips: everything after the trimmed audio is exact-zero padding
    for c in ("short", "nospeech_2s", "nospeech_exact4s"):
        (w,) = windows[c]
        assert not w[EXPECTED_TRIMMED[c]:].any(), f"{c}: padding is no longer exact zeros"
    assert EXPECTED_TRIMMED["nospeech_exact4s"] == WINDOW  # boundary case: zero-length padding

    # 4) LAST: exact bytes
    for c in CLIPS:
        got = [hashlib.sha256(w.tobytes()).hexdigest() for w in windows[c]]
        assert got == EXPECTED_SHA256[c], (
            f"{c}: structure unchanged but audio bytes differ: likely a library change "
            "(resampling/WAV I/O), or a real change to default preprocessing")
