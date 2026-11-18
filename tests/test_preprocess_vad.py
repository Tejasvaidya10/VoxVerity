import csv
from pathlib import Path

import numpy as np

import src.preprocessing.preprocess as pp
from src.preprocessing.preprocess import (load_and_resample, process_manifest, vad_keep_speech,
                                          vad_trim, voiced_flags)
from src.provenance import read_sidecar

SR = 16000
FRAME = 480  # 30 ms
FIX = Path(__file__).parent / "fixtures" / "preprocess_regression"


def test_voiced_flags_skips_final_frame():
    # 10 whole frames -> only 9 examined (today's loop never looks at the last one)
    assert len(voiced_flags(np.zeros(10 * FRAME, np.float32), SR, 2)) == 9


def _longest_quiet_run(x: np.ndarray) -> int:
    run = best = 0
    for quiet in np.abs(x) < 1e-2:
        run = run + 1 if quiet else 0
        best = max(best, run)
    return best


def test_keep_speech_removes_internal_pause():
    a = load_and_resample(str(FIX / "pause.wav"), SR)
    edges, allm = vad_trim(a, SR, 2), vad_keep_speech(a, SR, 2, 90)
    assert _longest_quiet_run(edges) >= int(0.9 * SR)   # edge trim keeps the pause
    # only 2 x 90 ms hangover (plus WebRTC's own ~0.1 s) of the pause may remain
    assert _longest_quiet_run(allm) < int(0.4 * SR)


def test_keep_speech_no_voiced_falls_back():
    z = np.zeros(2 * SR, np.float32)
    assert np.array_equal(vad_keep_speech(z, SR, 2, 90), z)


def test_hangover_keeps_three_frames_each_side_clipped_to_bounds(monkeypatch):
    # frames: voiced at 1 and 10..11 of 15 examined; hangover 90 ms = 3 frames
    flags = [False] * 15
    flags[1] = flags[10] = flags[11] = True
    monkeypatch.setattr(pp, "voiced_flags", lambda audio, sr, aggr: flags)
    a = np.arange(16 * FRAME, dtype=np.float32)
    out = vad_keep_speech(a, SR, 2, 90)
    kept_frames = [0, 1, 2, 3, 4] + [7, 8, 9, 10, 11, 12, 13, 14]  # 1±3 clipped at 0; 10..11 ±3
    expected = np.concatenate([a[i * FRAME:(i + 1) * FRAME] for i in kept_frames])
    assert np.array_equal(out, expected)


def test_kept_length_is_frame_multiple():
    a = load_and_resample(str(FIX / "pause.wav"), SR)
    assert len(vad_keep_speech(a, SR, 2, 90)) % FRAME == 0


def test_process_manifest_vad_all_shortens_and_records(tmp_path):
    m = tmp_path / "in.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "speaker_id", "dataset_source", "attack_type"])
        w.writeheader()
        w.writerow({"path": str(FIX / "pause.wav"), "label": "bonafide", "speaker_id": "s",
                    "dataset_source": "x", "attack_type": "-"})
    out_manifest = tmp_path / "chunks.csv"
    process_manifest(m, tmp_path / "audio", out_manifest, 16000, 4.0, 2,
                     chunk_stride_seconds=2.0, vad_mode="all")
    rows = list(csv.DictReader(open(out_manifest)))
    kept = vad_keep_speech(load_and_resample(str(FIX / "pause.wav"), SR), SR, 2, 90)
    import soundfile as sf
    w0, w1 = (sf.read(r["path"], dtype="float32")[0] for r in rows)  # 4.11 s -> 2 windows
    assert np.allclose(w0, kept[:4 * SR], atol=1 / 32768) and np.allclose(w1, kept[-4 * SR:], atol=1 / 32768)
    assert _longest_quiet_run(np.concatenate([w0, w1])) < int(0.4 * SR)
    assert read_sidecar(Path(str(out_manifest) + ".meta.json"))["settings"]["vad_mode"] == "all"
