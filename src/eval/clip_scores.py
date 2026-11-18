"""
Clip-level scoring: the detector scores 4 s windows, so a long clip counts
many times in a window-level EER. These helpers map windows back to their
source clip and aggregate window scores into one score per clip.
"""

from pathlib import Path

import pandas as pd

SCORE_COLUMNS = ["path", "clip_id", "chunk_idx", "label", "speaker_id", "score"]


def clip_id_from_chunk_path(path: str) -> str:
    """Windows are written as <clip stem>_<j>.wav; clip stems may contain
    underscores (ASVspoof5 T_0000000000), so split on the last one only."""
    return Path(path).stem.rsplit("_", 1)[0]


def read_scores(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={"path": str, "clip_id": str, "label": str,
                                        "speaker_id": str, "chunk_idx": int, "score": float})


def aggregate_clips(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if how not in ("mean", "median", "max"):
        raise ValueError(f"unknown aggregation {how!r}")
    mixed = df.groupby("clip_id")["label"].nunique()
    if (mixed > 1).any():
        raise ValueError(f"mixed labels within clip(s): {list(mixed[mixed > 1].index[:5])}")
    out = df.groupby("clip_id", sort=False).agg(
        label=("label", "first"), speaker_id=("speaker_id", "first"),
        score=("score", how), n_windows=("score", "size")).reset_index()
    return out[["clip_id", "label", "speaker_id", "score", "n_windows"]]
