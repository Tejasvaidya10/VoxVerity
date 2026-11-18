import numpy as np
import pandas as pd
import pytest

from src.analysis.phase5_stats import (Phase5Inputs, bootstrap_clips, errors_at,
                                       evaluate_predictions, paired_bootstrap)
from src.eval.metrics import compute_eer, fast_eer

T = 0.5  # threshold used throughout the synthetic tests


def test_fast_eer_equals_compute_eer_with_ties():
    rng = np.random.default_rng(0)
    b = np.r_[rng.random(500), np.ones(50)]
    s = np.r_[rng.random(400), np.ones(80), np.zeros(20)]
    assert fast_eer(b, s) == compute_eer(b, s)
    b2, s2 = rng.random(3000) ** 3, rng.random(2500) ** 0.5
    assert fast_eer(b2, s2) == compute_eer(b2, s2)


def test_errors_at_both_directions():
    df = pd.DataFrame({"label": ["spoof", "spoof", "bonafide", "bonafide"],
                       "score": [0.4, 0.6, 0.6, 0.4]})
    assert errors_at(df, T).tolist() == [True, False, True, False]


def test_bootstrap_resamples_clips_not_windows():
    # one clip with 100 wrong windows, 99 clips with one right window each
    df = pd.DataFrame({"clip_id": ["big"] * 100 + [f"c{i}" for i in range(99)],
                       "err": [1.0] * 100 + [0.0] * 99})
    lo, hi = bootstrap_clips(df, lambda d: d["err"].mean(), n=500)
    assert hi - lo > 0.3  # window-level resampling would give a width of ~0.07


def test_paired_bootstrap_uses_same_clips():
    a = pd.DataFrame({"clip_id": [f"c{i}" for i in range(50)], "v": np.arange(50.0)})
    b = a.assign(v=a["v"] + 2.0)
    lo, hi = paired_bootstrap(a, b, lambda d: d["v"].mean(), n=200)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)  # constant shift: zero width


def _inputs(padded_err=0.6, unpadded_err=0.1, pad_share=0.4, **kw):
    """200 ITW clips; short clips (one padded window) vs long (3 unpadded windows)."""
    rng = np.random.default_rng(1)
    clips, wins = [], []
    for i in range(200):
        label = "spoof" if i % 2 else "bonafide"
        short = i < 200 * pad_share
        snr = float(rng.uniform(0, 60))
        clips.append({"clip_id": f"c{i}", "label": label, "duration_trimmed_s": 2.0 if short else 8.0,
                      "speech_ratio": 0.6, "snr_db": snr})
        for j in range(1 if short else 3):
            wrong = rng.random() < (padded_err if short else unpadded_err)
            score = (0.2 if label == "spoof" else 0.8) if wrong else (0.8 if label == "spoof" else 0.2)
            wins.append({"clip_id": f"c{i}", "chunk_idx": j, "label": label, "score": score,
                         "pad_fraction": 0.5 if short else 0.0,
                         "window_speech_ratio": 0.5 if short else 0.9})
    train = pd.DataFrame({"clip_id": [f"t{i}" for i in range(100)],
                          "duration_trimmed_s": [8.0] * 98 + [2.0] * 2, "speech_ratio": [0.8] * 100})
    base = dict(itw_clips=pd.DataFrame(clips), train_clips=train, itw_windows=pd.DataFrame(wins),
                threshold=T, baseline_eers={"itw": 11.30, "dev": 1.18},
                variant_eers={"pad-repeat": {"itw": 10.20, "dev": 1.48},
                              "vad-all": {"itw": 11.00, "dev": 1.20}},
                clip_level_itw_eer=9.90, n_boot=200)
    base.update(kw)
    return Phase5Inputs(**base)


def test_p1_holds_when_padded_windows_fail_more():
    r = evaluate_predictions(_inputs())["P1"]
    assert r["holds"] and r["share_itw_le_4s"] == pytest.approx(0.4) and r["share_train_le_4s"] == 0.02


def test_p1_fails_without_error_gap():
    assert not evaluate_predictions(_inputs(padded_err=0.1))["P1"]["holds"]


def test_empty_bin_reports_not_holds():
    r = evaluate_predictions(_inputs(pad_share=0.0))["P1"]
    assert r["holds"] is False and "no padded windows" in r["reason"]


def test_p2_rule():
    assert evaluate_predictions(_inputs())["P2"]["holds"]  # -1.10 itw, +0.30 dev
    worse_dev = _inputs(variant_eers={"pad-repeat": {"itw": 10.20, "dev": 1.78},
                                      "vad-all": {"itw": 11.0, "dev": 1.2}})
    assert not evaluate_predictions(worse_dev)["P2"]["holds"]  # dev +0.60


def test_p3_rule():
    assert evaluate_predictions(_inputs())["P3"]["holds"]  # 9.90 <= 11.30 - 1
    assert not evaluate_predictions(_inputs(clip_level_itw_eer=10.50))["P3"]["holds"]


def test_p4_holds_when_low_speech_windows_fail_more():
    r = evaluate_predictions(_inputs())["P4"]
    assert r["holds"] and r["median_speech_ratio_itw"] < r["median_speech_ratio_train"]


def test_p5_not_holding_when_snr_unrelated():
    assert not evaluate_predictions(_inputs())["P5"]["holds"]  # SNR random in the builder


def test_p6_rule():
    assert not evaluate_predictions(_inputs())["P6"]["holds"]  # pad-repeat moved 1.10
    flat = _inputs(variant_eers={"pad-repeat": {"itw": 10.9, "dev": 1.2}, "vad-all": {"itw": 11.5, "dev": 1.2}},
                   clip_level_itw_eer=10.8)
    assert evaluate_predictions(flat)["P6"]["holds"]
