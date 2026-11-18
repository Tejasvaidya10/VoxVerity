import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.analysis.phase5_run import check_gate, link_identical_embeddings, sample_clips
from src.provenance import read_sidecar, sha256_file, write_sidecar

MODEL = "facebook/wav2vec2-xls-r-300m"


def _write_manifest(path: Path, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_sample_clips_deterministic_and_sized(tmp_path):
    rows = [{"path": f"/a/D_{i:04d}.flac", "label": "spoof", "speaker_id": "s"} for i in range(100)]
    src = tmp_path / "dev.csv"
    _write_manifest(src, rows[::-1])  # input order must not matter
    sample_clips(src, 10, 42, tmp_path / "a.csv")
    sample_clips(src, 10, 42, tmp_path / "b.csv")
    a = list(csv.DictReader(open(tmp_path / "a.csv")))
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert len(a) == 10 and [r["path"] for r in a] == sorted(r["path"] for r in a)
    sample_clips(src, 500, 42, tmp_path / "all.csv")
    assert len(list(csv.DictReader(open(tmp_path / "all.csv")))) == 100


def _gate_files(tmp_path, eer_offset=0.0, drop=0):
    rng = np.random.default_rng(0)
    rows = [{"path": f"/p/{i}_0.wav", "clip_id": str(i), "chunk_idx": 0,
             "label": "spoof" if i % 2 else "bonafide", "speaker_id": "s", "score": float(rng.random())}
            for i in range(200)]
    scores = tmp_path / "in_the_wild_chunks_scores.csv"
    _write_manifest(scores, rows[drop:])
    from src.analysis.phase5_report import eer_pct
    df = pd.DataFrame(rows)
    ref = {"generalization": {"in_the_wild_chunks": {
        "eer_pct": eer_pct(df) + eer_offset, "n_bonafide": 100, "n_spoof": 100}}}
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps(ref))
    return scores, reference


def test_gate_passes_on_reproduction(tmp_path):
    ok, msg = check_gate(*_gate_files(tmp_path))
    assert ok and msg.startswith("GATE PASS")


def test_gate_fails_on_eer_drift(tmp_path):
    ok, msg = check_gate(*_gate_files(tmp_path, eer_offset=0.06))
    assert not ok and "EER" in msg


def test_gate_fails_on_window_count(tmp_path):
    ok, msg = check_gate(*_gate_files(tmp_path, drop=1))
    assert not ok and "windows" in msg


def _windows(dirp: Path, clip_audio: dict):
    dirp.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, audio in clip_audio.items():
        sf.write(dirp / f"{stem}.wav", audio, 16000, subtype="PCM_16")
        rows.append({"path": str(dirp / f"{stem}.wav"), "label": "spoof", "speaker_id": "s",
                     "dataset_source": "x", "attack_type": "-", "chunk_idx": stem.rsplit("_", 1)[1]})
    return rows


def test_linker_links_only_byte_identical_windows(tmp_path):
    same = np.full(64000, 0.1, np.float32)
    base_rows = _windows(tmp_path / "base_audio", {"c1_0": same, "c2_0": np.zeros(64000, np.float32)})
    var_rows = _windows(tmp_path / "var_audio", {"c1_0": same, "c2_0": np.full(64000, 0.2, np.float32)})
    bm, vm = tmp_path / "base_chunks.csv", tmp_path / "var_chunks.csv"
    _write_manifest(bm, base_rows)
    _write_manifest(vm, var_rows)
    write_sidecar(Path(str(vm) + ".meta.json"), {"settings": {"pad_mode": "repeat"}})
    bf, vf = tmp_path / "base_feats", tmp_path / "var_feats"
    bf.mkdir()
    for stem in ("c1_0", "c2_0"):
        np.save(bf / f"{stem}.npy", np.ones(4))
    n = link_identical_embeddings(vm, bm, bf, vf, MODEL, -1, "mean")
    assert n == 1
    assert os.path.islink(vf / "c1_0.npy") and not (vf / "c2_0.npy").exists()
    meta = read_sidecar(vf / "_meta.json")
    assert meta["preprocessing"] == {"pad_mode": "repeat"}
    assert meta["input_manifest_sha256"] == sha256_file(vm)
    assert link_identical_embeddings(vm, bm, bf, vf, MODEL, -1, "mean") == 0  # idempotent
