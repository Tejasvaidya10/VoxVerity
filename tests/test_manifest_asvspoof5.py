import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.manifest import build_asvspoof5_manifest

FIXTURE = Path("tests/fixtures/asvspoof5_protocol_sample.txt")
FILE_IDS = ["T_0000000000", "T_0000000001", "T_0000000002",
            "T_0000000011", "T_0000000019"]


def _setup_raw(tmp_path, audio_subdir):
    raw = tmp_path / "asvspoof5"
    audio = raw / audio_subdir
    audio.mkdir(parents=True)
    proto = raw / "protocol.txt"
    proto.write_text(FIXTURE.read_text())
    for file_id in FILE_IDS:
        sf.write(audio / f"{file_id}.flac", np.zeros(160, dtype=np.float32), 16000)
    return raw, proto


def test_real_protocol_lines_parse(tmp_path):
    raw, proto = _setup_raw(tmp_path, "flac_T")
    out = tmp_path / "m.csv"
    build_asvspoof5_manifest(raw, out, protocol_path=proto, audio_subdir="flac_T")
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 5
    labels = {r["path"].split("/")[-1]: r["label"] for r in rows}
    assert labels["T_0000000000.flac"] == "spoof"
    assert labels["T_0000000011.flac"] == "bonafide"
    by_id = {Path(r["path"]).stem: r for r in rows}
    assert by_id["T_0000000000"]["speaker_id"] == "T_4850"
    assert by_id["T_0000000000"]["attack_type"] == "A05"


def test_legacy_smoke_format_still_parses(tmp_path):
    raw = tmp_path / "asvspoof5"
    (raw / "audio").mkdir(parents=True)
    (raw / "protocol.txt").write_text(
        "spk1 clip_a - A01 spoof\nspk2 clip_b - - bonafide\n")
    for fid in ("clip_a", "clip_b"):
        sf.write(raw / "audio" / f"{fid}.flac", np.zeros(160, dtype=np.float32), 16000)
    out = tmp_path / "m.csv"
    build_asvspoof5_manifest(raw, out)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert {r["label"] for r in rows} == {"spoof", "bonafide"}
