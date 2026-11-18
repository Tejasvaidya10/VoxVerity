"""
Phase 5 analysis: per-variant EERs with paired bootstrap CIs (each with the
preprocessing settings recorded in its score sidecar), the pre-registered
prediction verdicts, comparison and error plots, per-speaker error rates,
and the listening pack.
"""

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.phase5_stats import (Phase5Inputs, bootstrap_clips, errors_at,  # noqa: E402
                                       evaluate_predictions, paired_bootstrap)
from src.eval.clip_scores import aggregate_clips, read_scores  # noqa: E402
from src.eval.metrics import fast_eer  # noqa: E402
from src.provenance import read_sidecar  # noqa: E402

CLIP_PLOT_FIELDS = ["duration_trimmed_s", "speech_ratio", "longest_pause_s", "snr_db",
                    "rolloff95_hz", "f0_median_hz", "rms_dbfs"]
WINDOW_PLOT_FIELDS = ["pad_fraction", "window_speech_ratio"]
DIRECTIONS = {"bonafide": "real flagged as fake (false alarm)",
              "spoof": "fake passed as real (miss)"}


def eer_pct(df: pd.DataFrame) -> float:
    spoof = df["label"].to_numpy() == "spoof"
    score = df["score"].to_numpy()
    if spoof.all() or not spoof.any():
        return float("nan")
    return 100 * fast_eer(score[~spoof], score[spoof])[0]


def _find(variant_dir: Path, prefix: str) -> Path | None:
    hits = sorted(variant_dir.glob(f"{prefix}*_scores.csv"))
    if len(hits) > 1:
        raise ValueError(f"more than one {prefix}* score file in {variant_dir}: {hits}")
    return hits[0] if hits else None


def load_variants(scores_root: Path) -> dict:
    variants = {}
    for d in sorted(p for p in Path(scores_root).iterdir() if p.is_dir()):
        itw = _find(d, "in_the_wild")
        if itw is None:
            continue
        dev = _find(d, "asvspoof5_dev")
        for f in (itw, dev):
            if f is not None and read_sidecar(Path(str(f) + ".meta.json")) is None:
                raise ValueError(f"no provenance sidecar for {f}; every Phase 5 result must record its settings")
        side = read_sidecar(Path(str(itw) + ".meta.json"))
        variants[d.name] = {"itw": read_scores(itw), "dev": read_scores(dev) if dev else None,
                            "settings": side["preprocessing"]}
    if "baseline" not in variants:
        raise ValueError(f"no baseline variant under {scores_root}")
    return variants


def build_listening_pack(clip_scores: pd.DataFrame, windows: pd.DataFrame, clips: pd.DataFrame,
                         threshold: float, raw_paths: dict, out_dir: Path, k: int = 25) -> pd.DataFrame:
    out_dir = Path(out_dir)
    meas = clips.drop(columns=[c for c in ("label", "speaker_id") if c in clips.columns]).set_index("clip_id")
    rows = []
    for direction, label, high_first in (("real_flagged_fake", "bonafide", True),
                                         ("fake_passed_real", "spoof", False)):
        wrong = clip_scores["score"] >= threshold if high_first else clip_scores["score"] < threshold
        sel = clip_scores[(clip_scores["label"] == label) & wrong] \
            .sort_values("score", ascending=not high_first).head(k)
        (out_dir / direction).mkdir(parents=True, exist_ok=True)
        for rank, r in enumerate(sel.itertuples(index=False), 1):
            src = Path(raw_paths[r.clip_id])
            dst = out_dir / direction / f"{rank:02d}_{r.clip_id}{src.suffix}"
            shutil.copy2(src, dst)
            w = windows[windows["clip_id"] == r.clip_id]
            worst = w.loc[w["score"].idxmax() if high_first else w["score"].idxmin()]
            extra = meas.loc[r.clip_id].to_dict() if r.clip_id in meas.index else {}
            rows.append({"rank": rank, "direction": direction, "clip_id": r.clip_id,
                         "speaker_id": r.speaker_id, "label": r.label, "clip_score": r.score,
                         "n_windows": r.n_windows, "worst_window": int(worst["chunk_idx"]),
                         "worst_window_score": float(worst["score"]), "file": str(dst), **extra})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "listening.csv", index=False)
    notes = ["# Listening notes", "",
             "For each clip, what do you hear? Noise, reverb, music, long pauses or silence, codec "
             "artifacts, phone/radio bandwidth, accent, robotic prosody, anything odd.", ""]
    for r in rows:
        notes += [f"## {r['direction']} {r['rank']:02d}: clip {r['clip_id']} ({r['speaker_id']}), "
                  f"score {r['clip_score']:.3f}", "", "- ", ""]
    (out_dir / "listening_notes.md").write_text("\n".join(notes))
    return df


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _ecdf_plots(datasets: dict, plots_dir: Path) -> None:
    for field in CLIP_PLOT_FIELDS:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, df in datasets.items():
            if df is None or field not in df:
                continue
            v = np.sort(df[field].dropna().to_numpy())
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=f"{name} (n={len(v)})")
        ax.set_xlabel(field)
        ax.set_ylabel("share of clips at or below")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"ecdf_{field}.png", dpi=120)
        plt.close(fig)


def _errors_by_decile(w: pd.DataFrame, field: str, plots_dir: Path, n_boot: int) -> None:
    if w[field].nunique(dropna=True) < 2:
        return
    w = w.assign(bin=pd.qcut(w[field], 10, duplicates="drop"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, title in DIRECTIONS.items():
        sub = w[w["label"] == label]
        xs, ys, lo, hi = [], [], [], []
        for b, g in sub.groupby("bin", observed=True):
            if g.empty:
                continue
            rate = g["err"].mean()
            ci = bootstrap_clips(g, lambda d: d["err"].mean(), n=n_boot)
            xs.append(b.mid)
            ys.append(rate)
            lo.append(rate - ci[0])
            hi.append(ci[1] - rate)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=title)
    ax.set_xlabel(f"{field} (decile midpoints)")
    ax.set_ylabel("error rate at fixed threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"errors_by_{field}.png", dpi=120)
    plt.close(fig)


def _speakers(w: pd.DataFrame, clips: pd.DataFrame, out_dir: Path, plots_dir: Path) -> None:
    per = (w.groupby(["speaker_id", "label"])["err"].agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "error_rate", "size": "n_windows"}))
    med = clips.groupby("speaker_id")[[c for c in CLIP_PLOT_FIELDS if c in clips]].median().reset_index()
    per.merge(med, on="speaker_id", how="left").to_csv(out_dir / "speakers.csv", index=False)
    for label, title in DIRECTIONS.items():
        sub = per[per["label"] == label]
        big = sub[sub["n_windows"] >= 50]
        top = (big if len(big) else sub).sort_values("error_rate", ascending=False).head(20)
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top["speaker_id"].astype(str)[::-1], top["error_rate"][::-1])
        ax.set_xlabel(f"error rate: {title}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"speakers_{label}.png", dpi=120)
        plt.close(fig)


def run_analysis(scores_root: Path, measurements_dir: Path, raw_manifest_itw: Path,
                 out_dir: Path, n_boot: int = 1000) -> dict:
    out_dir, meas_dir = Path(out_dir), Path(measurements_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variants = load_variants(scores_root)
    base = variants["baseline"]
    base_itw, base_dev = base["itw"], base["dev"]
    spoof = base_itw["label"].to_numpy() == "spoof"
    threshold = fast_eer(base_itw["score"].to_numpy()[~spoof], base_itw["score"].to_numpy()[spoof])[1]

    def read(name):
        p = meas_dir / name
        return pd.read_csv(p, dtype={"clip_id": str, "speaker_id": str}) if p.exists() else None

    itw_clips, train_clips, dev_clips = read("in_the_wild_clips.csv"), read("asvspoof5_train_clips.csv"), \
        read("asvspoof5_dev_clips.csv")
    itw_wmeas = read("in_the_wild_windows.csv")
    if itw_clips is None or train_clips is None or itw_wmeas is None:
        raise ValueError(f"measurements missing under {meas_dir}")
    for name, df, keys in (("in_the_wild_clips", itw_clips, ["clip_id"]),
                           ("asvspoof5_train_clips", train_clips, ["clip_id"]),
                           ("asvspoof5_dev_clips", dev_clips, ["clip_id"]),
                           ("in_the_wild_windows", itw_wmeas, ["clip_id", "chunk_idx"]),
                           ("baseline ITW scores", base_itw, ["clip_id", "chunk_idx"])):
        if df is not None and df.duplicated(subset=keys).any():
            raise ValueError(f"duplicate rows in {name} ({int(df.duplicated(subset=keys).sum())})")
    itw_windows = base_itw.merge(itw_wmeas, on=["clip_id", "chunk_idx"], how="left")
    unmeasured = int(itw_windows["pad_fraction"].isna().sum())
    if len(itw_windows) != len(base_itw) or unmeasured:
        raise ValueError(f"{unmeasured} unmeasured baseline ITW windows; rerun the measurements")
    itw_windows = itw_windows.assign(err=errors_at(itw_windows, threshold))

    summary = {"threshold": threshold, "variants": {}, "unmeasured_windows": unmeasured}
    for name, v in variants.items():
        s = v["settings"]
        speech_only = isinstance(s, dict) and s.get("vad_mode") == "all"
        dur = itw_clips["duration_speech_only_s" if speech_only else "duration_trimmed_s"]
        entry = {"settings": s,
                 "itw": {"eer_pct": eer_pct(v["itw"]), "n_windows": len(v["itw"]),
                         "padded_share": float((dur < 4.0).mean())},
                 "dev": {"eer_pct": eer_pct(v["dev"]) if v["dev"] is not None else None}}
        if name != "baseline":
            entry["itw"]["delta_pts"] = entry["itw"]["eer_pct"] - eer_pct(base_itw)
            entry["itw"]["delta_ci"] = list(paired_bootstrap(base_itw, v["itw"], eer_pct, n_boot))
            if v["dev"] is not None and base_dev is not None:
                entry["dev"]["delta_pts"] = entry["dev"]["eer_pct"] - eer_pct(base_dev)
                entry["dev"]["delta_ci"] = list(paired_bootstrap(base_dev, v["dev"], eer_pct, n_boot))
        summary["variants"][name] = entry

    clip_itw = aggregate_clips(base_itw, "mean")
    summary["clip_mean"] = {"itw_eer_pct": eer_pct(clip_itw),
                            "delta_pts": eer_pct(clip_itw) - eer_pct(base_itw),
                            "delta_ci": list(paired_bootstrap(base_itw, clip_itw, eer_pct, n_boot)),
                            "dev_eer_pct": eer_pct(aggregate_clips(base_dev, "mean")) if base_dev is not None else None}
    summary["exploratory"] = {f"clip_{how}_itw_eer_pct": eer_pct(aggregate_clips(base_itw, how))
                              for how in ("median", "max")}

    inputs = Phase5Inputs(
        itw_clips=itw_clips, train_clips=train_clips, itw_windows=itw_windows.drop(columns=["err"]),
        threshold=threshold,
        baseline_eers={"itw": eer_pct(base_itw), "dev": eer_pct(base_dev) if base_dev is not None else float("nan")},
        variant_eers={n: {"itw": e["itw"]["eer_pct"], "dev": e["dev"]["eer_pct"]}
                      for n, e in summary["variants"].items()
                      if n != "baseline" and e["dev"]["eer_pct"] is not None},
        clip_level_itw_eer=summary["clip_mean"]["itw_eer_pct"], n_boot=n_boot)
    summary["predictions"] = evaluate_predictions(inputs)

    _ecdf_plots({"In-the-Wild": itw_clips, "ASVspoof5 train": train_clips, "ASVspoof5 dev": dev_clips}, plots_dir)
    joined = itw_windows.merge(itw_clips[["clip_id"] + [c for c in CLIP_PLOT_FIELDS if c in itw_clips]],
                               on="clip_id", how="left")
    for field in WINDOW_PLOT_FIELDS + CLIP_PLOT_FIELDS:
        _errors_by_decile(joined, field, plots_dir, n_boot=min(n_boot, 200))
    _speakers(itw_windows, itw_clips, out_dir, plots_dir)

    raw = {Path(p).stem: p for p in pd.read_csv(raw_manifest_itw)["path"]}
    build_listening_pack(clip_itw, base_itw, itw_clips, threshold, raw, out_dir / "listening")

    summary = _clean(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
