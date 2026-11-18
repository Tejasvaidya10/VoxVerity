from pathlib import Path

import numpy as np
import pytest

from src.features.extract_features import features_meta, prepare_features_dir
from src.provenance import ProvenanceMismatch, read_sidecar, sha256_file, write_sidecar

SETTINGS_ZERO = {"sample_rate": 16000, "chunk_seconds": 4.0, "chunk_stride_seconds": 2.0,
                 "vad_aggressiveness": 2, "vad_mode": "edges", "pad_mode": "zero",
                 "vad_hangover_ms": 90}
MODEL = "facebook/wav2vec2-xls-r-300m"


def _manifest(tmp_path: Path, settings: dict | None) -> Path:
    m = tmp_path / "chunks.csv"
    m.write_text("path,label\nx_0.wav,bonafide\n")
    if settings is not None:
        write_sidecar(Path(str(m) + ".meta.json"), {"stage": "preprocess", "settings": settings})
    return m


def test_meta_copies_preprocessing_settings(tmp_path):
    m = _manifest(tmp_path, SETTINGS_ZERO)
    meta = features_meta(m, MODEL, -1, "mean")
    assert meta == {"stage": "features", "preprocessing": SETTINGS_ZERO,
                    "input_manifest_sha256": sha256_file(m), "model_name": MODEL,
                    "layer": -1, "pooling": "mean"}


def test_meta_marks_legacy_manifest(tmp_path):
    m = _manifest(tmp_path, None)
    assert features_meta(m, MODEL, -1, "mean")["preprocessing"] == "unrecorded (pre-Phase 5)"


def test_refuses_dir_from_other_settings(tmp_path):
    out = tmp_path / "feats"
    prepare_features_dir(out, features_meta(_manifest(tmp_path, SETTINGS_ZERO), MODEL, -1, "mean"),
                         allow_legacy=False)
    (out / "x_0.npy").write_bytes(b"")
    repeat = {**SETTINGS_ZERO, "pad_mode": "repeat"}
    (tmp_path / "chunks.csv.meta.json").unlink()
    with pytest.raises(ProvenanceMismatch, match="preprocessing"):
        prepare_features_dir(out, features_meta(_manifest(tmp_path, repeat), MODEL, -1, "mean"),
                             allow_legacy=False)


def test_refuses_legacy_npy_dir_unless_allowed(tmp_path):
    out = tmp_path / "feats"
    out.mkdir()
    np.save(out / "x_0.npy", np.zeros(3))
    meta = features_meta(_manifest(tmp_path, SETTINGS_ZERO), MODEL, -1, "mean")
    with pytest.raises(ProvenanceMismatch, match="no provenance"):
        prepare_features_dir(out, meta, allow_legacy=False)
    prepare_features_dir(out, meta, allow_legacy=True)
    assert read_sidecar(out / "_meta.json")["preprocessing"] == SETTINGS_ZERO


def test_identical_meta_resumes(tmp_path):
    out = tmp_path / "feats"
    meta = features_meta(_manifest(tmp_path, SETTINGS_ZERO), MODEL, -1, "mean")
    prepare_features_dir(out, meta, allow_legacy=False)
    np.save(out / "x_0.npy", np.zeros(3))
    prepare_features_dir(out, meta, allow_legacy=False)  # no error
