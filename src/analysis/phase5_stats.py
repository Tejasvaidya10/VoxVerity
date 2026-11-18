"""
Phase 5 statistics and the pre-registered prediction rules (design spec
section 3). Rules are judged on point estimates; where a rule requires it,
the 95% bootstrap interval of the difference must exclude 0. Bootstraps
resample CLIPS, not windows (windows from one clip are not independent).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

GAP_POINTS = 1.0        # an intervention "moves" EER if it changes by >= 1.0 point
DEV_TOLERANCE = 0.5     # dev-control EER may rise by at most 0.5 point
P1_ITW_SHARE = 0.23
P1_TRAIN_SHARE = 0.05
SEED = 42


def errors_at(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Misclassified at the fixed threshold (score = spoof probability)."""
    spoof = df["label"].to_numpy() == "spoof"
    score = df["score"].to_numpy()
    return np.where(spoof, score < threshold, score >= threshold)


def _clip_index(clip_ids: pd.Series, categories=None):
    codes = pd.Categorical(clip_ids, categories=categories).codes
    n_clips = len(categories) if categories is not None else codes.max() + 1
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes[codes >= 0], minlength=n_clips)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return order, starts, counts


def _rows_for(sample: np.ndarray, order, starts, counts) -> np.ndarray:
    lens = counts[sample]
    offsets = np.repeat(starts[sample] - np.r_[0, np.cumsum(lens)[:-1]], lens)
    return order[np.arange(lens.sum()) + offsets]


def bootstrap_clips(df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float],
                    n: int = 1000, seed: int = SEED) -> tuple:
    cats = pd.unique(df["clip_id"])
    order, starts, counts = _clip_index(df["clip_id"], cats)
    rng = np.random.default_rng(seed)
    vals = [stat_fn(df.iloc[_rows_for(rng.integers(0, len(cats), len(cats)), order, starts, counts)])
            for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_bootstrap(dfa: pd.DataFrame, dfb: pd.DataFrame,
                     stat_fn: Callable[[pd.DataFrame], float],
                     n: int = 1000, seed: int = SEED) -> tuple:
    """CI of stat_fn(b) - stat_fn(a), drawing the same clips for both."""
    cats = np.array(sorted(set(dfa["clip_id"]) & set(dfb["clip_id"])))
    ia = _clip_index(dfa["clip_id"], cats)
    ib = _clip_index(dfb["clip_id"], cats)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.integers(0, len(cats), len(cats))
        vals.append(stat_fn(dfb.iloc[_rows_for(sample, *ib)]) - stat_fn(dfa.iloc[_rows_for(sample, *ia)]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


@dataclass
class Phase5Inputs:
    itw_clips: pd.DataFrame       # clip_id, label, duration_trimmed_s, speech_ratio, snr_db, ...
    train_clips: pd.DataFrame     # clip_id, duration_trimmed_s, speech_ratio, ...
    itw_windows: pd.DataFrame     # baseline window scores + window measures: clip_id, chunk_idx,
                                  # label, score, pad_fraction, window_speech_ratio
    threshold: float              # baseline ITW window-level EER threshold
    baseline_eers: dict           # {"itw": pct, "dev": pct} window-level baseline
    variant_eers: dict = field(default_factory=dict)  # {"pad-repeat": {"itw": pct, "dev": pct}, ...}
    clip_level_itw_eer: float = float("nan")          # pct, clip-mean scoring of baseline
    n_boot: int = 1000


def _rate_diff(d: pd.DataFrame, a_mask: str, b_mask: str) -> float:
    a, b = d[a_mask].to_numpy(), d[b_mask].to_numpy()
    if not a.any() or not b.any():
        return float("nan")
    err = d["err"].to_numpy()
    return float(err[a].mean() - err[b].mean())


def _verdict(holds: bool, reason: str, **nums) -> dict:
    return {"holds": bool(holds), "reason": reason, **nums}


def _p1(x: Phase5Inputs, w: pd.DataFrame) -> dict:
    share_itw = float((x.itw_clips["duration_trimmed_s"] <= 4.0).mean())
    share_train = float((x.train_clips["duration_trimmed_s"] <= 4.0).mean())
    w = w.assign(padded=w["pad_fraction"] > 0, unpadded=~(w["pad_fraction"] > 0))
    nums = {"share_itw_le_4s": share_itw, "share_train_le_4s": share_train}
    if not w["padded"].any():
        return _verdict(False, "no padded windows", **nums)
    if not w["unpadded"].any():
        return _verdict(False, "no unpadded windows", **nums)
    diff = _rate_diff(w, "padded", "unpadded")
    lo, hi = bootstrap_clips(w, lambda d: _rate_diff(d, "padded", "unpadded"), x.n_boot)
    holds = share_itw >= P1_ITW_SHARE and share_train < P1_TRAIN_SHARE and diff > 0 and lo > 0
    return _verdict(holds, f"padded minus unpadded error rate {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                    **nums, error_rate_diff=diff, ci=[lo, hi])


def _p2(x: Phase5Inputs) -> dict:
    v = x.variant_eers.get("pad-repeat")
    if v is None:
        return _verdict(False, "pad-repeat variant missing")
    d_itw, d_dev = v["itw"] - x.baseline_eers["itw"], v["dev"] - x.baseline_eers["dev"]
    return _verdict(d_itw <= -GAP_POINTS and d_dev <= DEV_TOLERANCE,
                    f"ITW {d_itw:+.2f} pts, dev {d_dev:+.2f} pts", delta_itw=d_itw, delta_dev=d_dev)


def _p3(x: Phase5Inputs) -> dict:
    d = x.clip_level_itw_eer - x.baseline_eers["itw"]
    if np.isnan(d):
        return _verdict(False, "clip-level EER missing")
    return _verdict(d <= -GAP_POINTS, f"clip-level minus window-level ITW EER {d:+.2f} pts", delta_itw=d)


def _p4(x: Phase5Inputs, w: pd.DataFrame) -> dict:
    med_itw = float(x.itw_clips["speech_ratio"].median())
    med_train = float(x.train_clips["speech_ratio"].median())
    wsr = w["window_speech_ratio"].to_numpy()
    n = len(wsr)
    rng = np.random.default_rng(SEED)  # ties broken in a fixed random order
    pos = np.empty(n, dtype=int)
    pos[np.lexsort((rng.random(n), wsr))] = np.arange(n)
    w = w.assign(low=pos < n / 4, high=pos >= 3 * n / 4)
    nums = {"median_speech_ratio_itw": med_itw, "median_speech_ratio_train": med_train}
    if n < 4:
        return _verdict(False, "too few windows for quartiles", **nums)
    diff = _rate_diff(w, "low", "high")
    lo, hi = bootstrap_clips(w, lambda d: _rate_diff(d, "low", "high"), x.n_boot)
    holds = med_itw < med_train and diff > 0 and lo > 0
    return _verdict(holds, f"lowest minus highest speech-ratio quartile error {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                    **nums, error_rate_diff=diff, ci=[lo, hi])


def _p5(x: Phase5Inputs, w: pd.DataFrame) -> dict:
    clips = x.itw_clips.dropna(subset=["snr_db"])
    flags = {}
    for label in ("spoof", "bonafide"):
        c = clips[clips["label"] == label]
        if len(c) < 3:
            return _verdict(False, f"too few {label} clips with an SNR")
        q1, q2 = np.percentile(c["snr_db"], [100 / 3, 200 / 3])
        flags.update({cid: "low" for cid in c.loc[c["snr_db"] <= q1, "clip_id"]})
        flags.update({cid: "high" for cid in c.loc[c["snr_db"] >= q2, "clip_id"]})
    terc = w["clip_id"].map(flags)
    spoof = w["label"] == "spoof"
    w = w.assign(fake_low=spoof & (terc == "low"), fake_high=spoof & (terc == "high"),
                 real_low=~spoof & (terc == "low"), real_high=~spoof & (terc == "high"))
    miss = _rate_diff(w, "fake_low", "fake_high")   # misses: low-SNR fakes minus high-SNR fakes
    fa = _rate_diff(w, "real_high", "real_low")     # false alarms: high-SNR reals minus low-SNR reals
    if np.isnan(miss) or np.isnan(fa):
        return _verdict(False, "an SNR tercile has no windows")
    mlo, mhi = bootstrap_clips(w, lambda d: _rate_diff(d, "fake_low", "fake_high"), x.n_boot)
    flo, fhi = bootstrap_clips(w, lambda d: _rate_diff(d, "real_high", "real_low"), x.n_boot)
    holds = miss > 0 and mlo > 0 and fa > 0 and flo > 0
    return _verdict(holds, f"miss low-high SNR {miss:+.3f} [{mlo:+.3f}, {mhi:+.3f}]; "
                           f"false alarm high-low SNR {fa:+.3f} [{flo:+.3f}, {fhi:+.3f}]",
                    miss_diff=miss, miss_ci=[mlo, mhi], fa_diff=fa, fa_ci=[flo, fhi])


def _p6(x: Phase5Inputs) -> dict:
    deltas = {k: v["itw"] - x.baseline_eers["itw"] for k, v in x.variant_eers.items()}
    if not np.isnan(x.clip_level_itw_eer):
        deltas["clip-mean"] = x.clip_level_itw_eer - x.baseline_eers["itw"]
    if not deltas:
        return _verdict(False, "no interventions to compare")
    moved = {k: d for k, d in deltas.items() if abs(d) >= GAP_POINTS}
    reason = ("no intervention moved ITW EER by >= 1 pt (remaining gap most likely unseen attacks; inference)"
              if not moved else f"moved by >= 1 pt: {', '.join(f'{k} {d:+.2f}' for k, d in moved.items())}")
    return _verdict(not moved, reason, deltas=deltas)


def evaluate_predictions(inputs: Phase5Inputs) -> dict:
    w = inputs.itw_windows.assign(err=errors_at(inputs.itw_windows, inputs.threshold))
    return {"P1": _p1(inputs, w), "P2": _p2(inputs), "P3": _p3(inputs),
            "P4": _p4(inputs, w), "P5": _p5(inputs, w), "P6": _p6(inputs)}
