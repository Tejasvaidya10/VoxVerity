import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.eval.harness import collect_provenance, run_generalization_eval, write_scores
from src.models.detector import build_head
from src.provenance import ProvenanceMismatch, read_sidecar, sha256_file, write_sidecar

CONFIG = yaml.safe_load(open("configs/default.yaml"))
ZERO = {"sample_rate": 16000, "chunk_seconds": 4.0, "chunk_stride_seconds": 2.0,
        "vad_aggressiveness": 2, "vad_mode": "edges", "pad_mode": "zero", "vad_hangover_ms": 90}


def _dataset(tmp_path: Path, name: str, clips: list) -> tuple:
    """clips: [(clip_id, label, n_windows)] -> (manifest, features dir) with fake embeddings."""
    feats = tmp_path / f"feats_{name}"
    feats.mkdir()
    rng = np.random.default_rng(len(name))
    rows = []
    for clip, label, n in clips:
        for j in range(n):
            stem = f"{clip}_{j}"
            np.save(feats / f"{stem}.npy", rng.standard_normal(1024).astype(np.float32))
            rows.append({"path": f"/proc/{name}/{stem}.wav", "label": label, "speaker_id": f"spk_{clip}",
                         "dataset_source": name, "attack_type": "-", "chunk_idx": j})
    m = tmp_path / f"{name}_chunks.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return m, feats


def _model():
    torch.manual_seed(0)
    return build_head(CONFIG["model"]).eval()


def test_features_dirs_used_per_manifest(tmp_path):
    ma, fa = _dataset(tmp_path, "a", [("T_1", "spoof", 2), ("T_2", "bonafide", 1)])
    mb, fb = _dataset(tmp_path, "b", [("7", "spoof", 1), ("8", "bonafide", 3)])
    res = run_generalization_eval(CONFIG, _model(), [ma, mb], None, "cpu", features_dirs=[fa, fb])
    assert len(res["a_chunks"]["per_file_scores"]) == 3
    assert len(res["b_chunks"]["per_file_scores"]) == 4


def test_scores_csv_columns_order_and_meta(tmp_path):
    m, f = _dataset(tmp_path, "a", [("T_1", "spoof", 2), ("T_2", "bonafide", 1)])
    res = run_generalization_eval(CONFIG, _model(), [m], f, "cpu")
    prov = {"preprocessing": ZERO, "input_manifest_sha256": "abc", "features": None}
    out = write_scores(m, res["a_chunks"]["per_file_scores"], tmp_path / "scores", prov)
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0].keys()) == ["path", "clip_id", "chunk_idx", "label", "speaker_id", "score"]
    assert [(r["clip_id"], r["chunk_idx"], r["label"]) for r in rows] == [
        ("T_1", "0", "spoof"), ("T_1", "1", "spoof"), ("T_2", "0", "bonafide")]
    assert 0.0 <= float(rows[0]["score"]) <= 1.0
    assert read_sidecar(Path(str(out) + ".meta.json"))["preprocessing"] == ZERO


def test_provenance_collects_settings_and_checksum(tmp_path):
    m, f = _dataset(tmp_path, "a", [("T_1", "spoof", 1)])
    write_sidecar(Path(str(m) + ".meta.json"), {"settings": ZERO})
    write_sidecar(f / "_meta.json", {"preprocessing": ZERO, "input_manifest_sha256": sha256_file(m)})
    prov = collect_provenance(m, f)
    assert prov["preprocessing"] == ZERO
    assert prov["input_manifest_sha256"] == sha256_file(m)
    assert prov["features"]["preprocessing"] == ZERO


def test_mismatched_preprocessing_raises(tmp_path):
    m, f = _dataset(tmp_path, "a", [("T_1", "spoof", 1)])
    write_sidecar(Path(str(m) + ".meta.json"), {"settings": ZERO})
    write_sidecar(f / "_meta.json", {"preprocessing": {**ZERO, "pad_mode": "repeat"},
                                     "input_manifest_sha256": sha256_file(m)})
    with pytest.raises(ProvenanceMismatch, match="preprocessing"):
        collect_provenance(m, f)


def test_features_from_other_manifest_raise(tmp_path):
    m, f = _dataset(tmp_path, "a", [("T_1", "spoof", 1)])
    write_sidecar(Path(str(m) + ".meta.json"), {"settings": ZERO})
    write_sidecar(f / "_meta.json", {"preprocessing": ZERO, "input_manifest_sha256": "other"})
    with pytest.raises(ProvenanceMismatch, match="manifest"):
        collect_provenance(m, f)


def test_legacy_manifest_warns_not_fails(tmp_path, capsys):
    m, f = _dataset(tmp_path, "a", [("T_1", "spoof", 1)])
    prov = collect_provenance(m, f)
    assert prov["preprocessing"] == "unrecorded (pre-Phase 5)" and prov["features"] is None
    assert "WARNING" in capsys.readouterr().out
