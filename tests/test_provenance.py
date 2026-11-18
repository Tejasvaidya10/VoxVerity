import pytest

from src.provenance import (ProvenanceMismatch, check_resume, read_sidecar,
                            sha256_file, write_sidecar)


def test_sha256_file_known_value(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sidecar_round_trip_adds_stamp(tmp_path):
    p = tmp_path / "m.csv.meta.json"
    write_sidecar(p, {"settings": {"pad_mode": "zero"}})
    meta = read_sidecar(p)
    assert meta["settings"] == {"pad_mode": "zero"}
    assert "created_at" in meta and "git_commit" in meta


def test_read_missing_sidecar_is_none(tmp_path):
    assert read_sidecar(tmp_path / "nope.json") is None


def test_resume_into_empty_dir_ok(tmp_path):
    check_resume(tmp_path / "_meta.json", {"settings": 1}, ["settings"],
                 has_outputs=False, allow_legacy=False)


def test_legacy_dir_with_outputs_refused_unless_allowed(tmp_path):
    with pytest.raises(ProvenanceMismatch, match="no provenance"):
        check_resume(tmp_path / "_meta.json", {"settings": 1}, ["settings"],
                     has_outputs=True, allow_legacy=False)
    check_resume(tmp_path / "_meta.json", {"settings": 1}, ["settings"],
                 has_outputs=True, allow_legacy=True)


def test_matching_sidecar_resumes_despite_different_stamp(tmp_path):
    meta_path = tmp_path / "_meta.json"
    write_sidecar(meta_path, {"settings": {"pad_mode": "zero"}, "input_manifest_sha256": "aa"})
    check_resume(meta_path, {"settings": {"pad_mode": "zero"}, "input_manifest_sha256": "aa",
                             "created_at": "later"},
                 ["settings", "input_manifest_sha256"], has_outputs=True, allow_legacy=False)


def test_differing_sidecar_refused_naming_key(tmp_path):
    meta_path = tmp_path / "_meta.json"
    write_sidecar(meta_path, {"settings": {"pad_mode": "zero"}, "input_manifest_sha256": "aa"})
    with pytest.raises(ProvenanceMismatch, match="settings"):
        check_resume(meta_path, {"settings": {"pad_mode": "repeat"}, "input_manifest_sha256": "aa"},
                     ["settings", "input_manifest_sha256"], has_outputs=True, allow_legacy=False)
