import pandas as pd
import pytest

from src.eval.clip_scores import aggregate_clips, clip_id_from_chunk_path, read_scores


def test_clip_id_last_underscore_only():
    assert clip_id_from_chunk_path("x/T_0000000000_3.wav") == "T_0000000000"
    assert clip_id_from_chunk_path("data/processed/in_the_wild/1234_0.wav") == "1234"


def _df(rows):
    return pd.DataFrame(rows, columns=["path", "clip_id", "chunk_idx", "label", "speaker_id", "score"])


def test_mean_aggregation_one_row_per_clip():
    df = _df([["a_0.wav", "a", 0, "spoof", "s1", 0.2],
              ["a_1.wav", "a", 1, "spoof", "s1", 0.4],
              ["b_0.wav", "b", 0, "bonafide", "s2", 0.9]])
    out = aggregate_clips(df).set_index("clip_id")
    assert out.loc["a", "score"] == pytest.approx(0.3)
    assert out.loc["a", "n_windows"] == 2
    assert (out.loc["b", "label"], out.loc["b", "speaker_id"], out.loc["b", "score"]) == ("bonafide", "s2", 0.9)


def test_max_and_median_aggregation():
    df = _df([["a_0.wav", "a", 0, "spoof", "s", 0.1],
              ["a_1.wav", "a", 1, "spoof", "s", 0.2],
              ["a_2.wav", "a", 2, "spoof", "s", 0.9]])
    assert aggregate_clips(df, "max")["score"].iloc[0] == pytest.approx(0.9)
    assert aggregate_clips(df, "median")["score"].iloc[0] == pytest.approx(0.2)


def test_mixed_labels_within_clip_raise():
    df = _df([["a_0.wav", "a", 0, "spoof", "s", 0.2], ["a_1.wav", "a", 1, "bonafide", "s", 0.4]])
    with pytest.raises(ValueError, match="mixed labels"):
        aggregate_clips(df)


def test_read_scores_types(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("path,clip_id,chunk_idx,label,speaker_id,score\nT_1_0.wav,T_1,0,spoof,s,0.25\n")
    df = read_scores(p)
    assert df["clip_id"].iloc[0] == "T_1" and df["score"].iloc[0] == 0.25
    assert df["chunk_idx"].dtype.kind == "i"
