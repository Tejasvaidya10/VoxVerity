import csv
from pathlib import Path

import pytest

from src.preprocessing.preprocess import process_manifest, resolve_settings
from src.provenance import ProvenanceMismatch, read_sidecar, sha256_file

CONFIG = Path("configs/default.yaml")
SHORT = Path(__file__).parent / "fixtures" / "preprocess_regression" / "short.wav"


def test_config_defaults_match_today():
    s = resolve_settings(CONFIG, {})
    assert (s.pad_mode, s.vad_mode, s.vad_hangover_ms) == ("zero", "edges", 90)
    assert (s.sample_rate, s.chunk_seconds, s.chunk_stride_seconds, s.vad_aggressiveness) == (16000, 4.0, 2.0, 2)


def test_no_config_keeps_cli_defaults():
    s = resolve_settings(None, {})
    assert s.chunk_stride_seconds is None  # today's CLI default: no overlap
    assert (s.pad_mode, s.vad_mode) == ("zero", "edges")


def test_explicit_override_wins_and_none_is_ignored():
    s = resolve_settings(CONFIG, {"pad_mode": "repeat", "chunk_seconds": None})
    assert s.pad_mode == "repeat"
    assert s.chunk_seconds == 4.0


def _manifest(tmp_path: Path) -> Path:
    m = tmp_path / "in.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "speaker_id", "dataset_source", "attack_type"])
        w.writeheader()
        w.writerow({"path": str(SHORT), "label": "bonafide", "speaker_id": "s",
                    "dataset_source": "x", "attack_type": "-"})
    return m


def _run(tmp_path: Path, **kw):
    m = _manifest(tmp_path)
    out_audio, out_manifest = tmp_path / "audio", tmp_path / "chunks.csv"
    process_manifest(m, out_audio, out_manifest, 16000, 4.0, 2, chunk_stride_seconds=2.0, **kw)
    return m, out_audio, out_manifest


def test_sidecars_written_with_settings_and_checksums(tmp_path):
    m, out_audio, out_manifest = _run(tmp_path, config_path=CONFIG)
    side = read_sidecar(Path(str(out_manifest) + ".meta.json"))
    dir_side = read_sidecar(out_audio / "_meta.json")
    assert side["stage"] == "preprocess"
    assert side["settings"] == {"sample_rate": 16000, "chunk_seconds": 4.0, "chunk_stride_seconds": 2.0,
                                "vad_aggressiveness": 2, "vad_mode": "edges", "pad_mode": "zero",
                                "vad_hangover_ms": 90}
    assert side["config_path"] == str(CONFIG)
    assert side["input_manifest_sha256"] == sha256_file(m)
    assert side["output_manifest_sha256"] == sha256_file(out_manifest)
    assert (side["n_source_rows"], side["n_chunks"]) == (1, 1)
    assert dir_side["settings"] == side["settings"]


def test_refuses_resume_with_different_settings(tmp_path):
    _run(tmp_path)
    with pytest.raises(ProvenanceMismatch, match="settings"):
        _run(tmp_path, vad_hangover_ms=120)


def test_resume_with_identical_settings_ok(tmp_path):
    _, _, out_manifest = _run(tmp_path)
    first = out_manifest.read_bytes()
    _run(tmp_path)
    assert out_manifest.read_bytes() == first


def test_refuses_legacy_dir_unless_allowed(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "stray_0.wav").write_bytes(b"")
    with pytest.raises(ProvenanceMismatch, match="no provenance"):
        _run(tmp_path)
    _run(tmp_path, allow_legacy_dir=True)


def test_non_default_modes_not_yet_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        _run(tmp_path, pad_mode="repeat")
