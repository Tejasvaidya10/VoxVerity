import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.analysis.phase5_report import build_listening_pack, run_analysis
from src.provenance import write_sidecar

ZERO = {"sample_rate": 16000, "chunk_seconds": 4.0, "chunk_stride_seconds": 2.0,
        "vad_aggressiveness": 2, "vad_mode": "edges", "pad_mode": "zero", "vad_hangover_ms": 90}


def _wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(1600, np.float32), 16000, subtype="PCM_16")


def test_listening_pack_picks_most_confident_errors(tmp_path):
    clip_scores = pd.DataFrame({"clip_id": [f"r{i}" for i in range(30)] + ["f0", "f1"],
                                "label": ["bonafide"] * 30 + ["spoof"] * 2,
                                "speaker_id": "s", "score": list(np.linspace(0.0, 0.99, 30)) + [0.1, 0.9],
                                "n_windows": 1})
    windows = clip_scores.assign(chunk_idx=0)[["clip_id", "chunk_idx", "label", "score"]]
    clips = pd.DataFrame({"clip_id": clip_scores["clip_id"], "snr_db": 20.0})
    raw = {}
    for c in clip_scores["clip_id"]:
        _wav(tmp_path / "raw" / f"{c}.wav")
        raw[c] = str(tmp_path / "raw" / f"{c}.wav")
    out = build_listening_pack(clip_scores, windows, clips, 0.5, raw, tmp_path / "listen", k=25)
    reals = out[out["direction"] == "real_flagged_fake"]
    expected = clip_scores[(clip_scores.label == "bonafide") & (clip_scores.score >= 0.5)] \
        .sort_values("score", ascending=False)["clip_id"].tolist()[:25]
    assert reals["clip_id"].tolist() == expected            # 15 qualify, ranked by score desc
    assert out[out["direction"] == "fake_passed_real"]["clip_id"].tolist() == ["f0"]
    assert (tmp_path / "listen" / "real_flagged_fake" / f"01_{expected[0]}.wav").exists()
    assert (tmp_path / "listen" / "listening.csv").exists()
    assert "## real_flagged_fake 01" in (tmp_path / "listen" / "listening_notes.md").read_text()


def _scores(path: Path, rows, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "clip_id", "chunk_idx", "label", "speaker_id", "score"])
        w.writeheader()
        w.writerows(rows)
    write_sidecar(Path(str(path) + ".meta.json"),
                  {"preprocessing": settings, "input_manifest_sha256": "x", "features": None})


def _tree(tmp_path: Path):
    rng = np.random.default_rng(0)
    itw_rows, win_meas, clip_meas, raw_rows = [], [], [], []
    for i in range(60):
        label = "spoof" if i % 2 else "bonafide"
        n = 1 if i % 3 == 0 else 3
        for j in range(n):
            itw_rows.append({"path": f"/p/{i}_{j}.wav", "clip_id": str(i), "chunk_idx": j, "label": label,
                             "speaker_id": f"spk{i % 5}", "score": float(rng.random())})
            win_meas.append({"clip_id": str(i), "chunk_idx": j, "pad_fraction": 0.3 if n == 1 else 0.0,
                             "window_speech_ratio": float(rng.random())})
        clip_meas.append({"clip_id": str(i), "label": label, "speaker_id": f"spk{i % 5}",
                          "duration_raw_s": 9.0, "duration_trimmed_s": 3.0 if n == 1 else 8.0,
                          "duration_speech_only_s": 2.5 if n == 1 else 6.0,
                          "speech_ratio": float(rng.random()), "n_pauses": 1, "longest_pause_s": 0.5,
                          "snr_db": float(rng.uniform(5, 50)), "rolloff95_hz": 5000.0,
                          "f0_median_hz": 150.0, "rms_dbfs": -20.0, "clip_frac": 0.0})
        _wav(tmp_path / "raw" / f"{i}.wav")
        raw_rows.append({"path": str(tmp_path / "raw" / f"{i}.wav"), "label": label,
                         "speaker_id": f"spk{i % 5}", "dataset_source": "in_the_wild", "attack_type": "-"})
    dev_rows = [{"path": f"/d/D_{i}_0.wav", "clip_id": f"D_{i}", "chunk_idx": 0,
                 "label": "spoof" if i % 2 else "bonafide", "speaker_id": "d", "score": float(rng.random())}
                for i in range(40)]
    sc = tmp_path / "scores"
    _scores(sc / "baseline" / "in_the_wild_chunks_scores.csv", itw_rows, ZERO)
    _scores(sc / "baseline" / "asvspoof5_dev_slice_chunks_scores.csv", dev_rows, ZERO)
    rep = {**ZERO, "pad_mode": "repeat"}
    _scores(sc / "pad-repeat" / "in_the_wild__pad-repeat_chunks_scores.csv",
            [{**r, "score": min(1.0, r["score"] * 1.1)} for r in itw_rows], rep)
    _scores(sc / "pad-repeat" / "asvspoof5_dev_slice__pad-repeat_chunks_scores.csv", dev_rows, rep)
    meas = tmp_path / "meas"
    meas.mkdir()
    pd.DataFrame(clip_meas).to_csv(meas / "in_the_wild_clips.csv", index=False)
    pd.DataFrame(win_meas).to_csv(meas / "in_the_wild_windows.csv", index=False)
    for ds in ("asvspoof5_train", "asvspoof5_dev"):
        pd.DataFrame(clip_meas).assign(clip_id=lambda d: ds + d["clip_id"]).to_csv(meas / f"{ds}_clips.csv", index=False)
    raw_manifest = tmp_path / "in_the_wild.csv"
    pd.DataFrame(raw_rows).to_csv(raw_manifest, index=False)
    return sc, meas, raw_manifest


def test_summary_records_variant_settings_and_predictions(tmp_path):
    sc, meas, raw_manifest = _tree(tmp_path)
    summary = run_analysis(sc, meas, raw_manifest, tmp_path / "out", n_boot=20)
    on_disk = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert on_disk["variants"]["pad-repeat"]["settings"]["pad_mode"] == "repeat"
    assert on_disk["variants"]["baseline"]["settings"]["pad_mode"] == "zero"
    assert set(on_disk["predictions"]) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert on_disk["variants"]["baseline"]["itw"]["padded_share"] == summary["variants"]["baseline"]["itw"]["padded_share"]
    assert abs(on_disk["variants"]["baseline"]["itw"]["padded_share"] - 20 / 60) < 1e-9
    assert len(on_disk["variants"]["pad-repeat"]["itw"]["delta_ci"]) == 2
    assert "itw_eer_pct" in on_disk["clip_mean"]
    assert any((tmp_path / "out" / "plots").glob("*.png"))
    assert (tmp_path / "out" / "listening" / "listening.csv").exists()


def test_duplicate_measurement_rows_rejected(tmp_path):
    import pytest
    sc, meas, raw_manifest = _tree(tmp_path)
    p = meas / "in_the_wild_clips.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[1]]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        run_analysis(sc, meas, raw_manifest, tmp_path / "out", n_boot=5)


def test_unmeasured_windows_rejected(tmp_path):
    import pytest
    sc, meas, raw_manifest = _tree(tmp_path)
    p = meas / "in_the_wild_windows.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="unmeasured"):
        run_analysis(sc, meas, raw_manifest, tmp_path / "out", n_boot=5)


def test_missing_score_sidecar_rejected(tmp_path):
    import pytest
    sc, meas, raw_manifest = _tree(tmp_path)
    (sc / "pad-repeat" / "in_the_wild__pad-repeat_chunks_scores.csv.meta.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        run_analysis(sc, meas, raw_manifest, tmp_path / "out", n_boot=5)
