import numpy as np
from src.preprocessing.preprocess import chunk_audio

SR = 16000


def test_short_audio_padded_to_one_chunk():
    audio = np.ones(SR, dtype=np.float32)  # 1s
    chunks = chunk_audio(audio, SR, chunk_seconds=4.0, stride_seconds=2.0)
    assert len(chunks) == 1
    assert len(chunks[0]) == 4 * SR


def test_no_overlap_when_stride_omitted():
    audio = np.ones(8 * SR, dtype=np.float32)
    chunks = chunk_audio(audio, SR, chunk_seconds=4.0)
    assert len(chunks) == 2


def test_overlap_stride():
    audio = np.ones(8 * SR, dtype=np.float32)  # 8s, 4s window, 2s stride -> starts 0,2,4
    chunks = chunk_audio(audio, SR, chunk_seconds=4.0, stride_seconds=2.0)
    assert len(chunks) == 3
    assert all(len(c) == 4 * SR for c in chunks)


def test_tail_covered_by_end_aligned_window():
    audio = np.arange(9 * SR, dtype=np.float32)  # 9s: starts 0,2,4 + end-aligned 5
    chunks = chunk_audio(audio, SR, chunk_seconds=4.0, stride_seconds=2.0)
    assert len(chunks) == 4
    assert chunks[-1][-1] == audio[-1]  # last window reaches the final sample


def test_repeat_pad_tiles_from_start():
    a = np.arange(1, SR + 1, dtype=np.float32)  # 1 s ramp, no zeros
    (c,) = chunk_audio(a, SR, chunk_seconds=4.0, stride_seconds=2.0, pad_mode="repeat")
    assert len(c) == 4 * SR
    assert np.array_equal(c[:SR], a) and np.array_equal(c[SR:2 * SR], a)
    assert c[-1] == a[-1]  # 4 whole repeats fill the window exactly


def test_repeat_pad_truncates_partial_repeat():
    a = np.arange(1, int(1.5 * SR) + 1, dtype=np.float32)  # 1.5 s -> 2 full + 1/3 repeat
    (c,) = chunk_audio(a, SR, chunk_seconds=4.0, stride_seconds=2.0, pad_mode="repeat")
    assert len(c) == 4 * SR
    assert np.array_equal(c[int(3.0 * SR):], a[:SR])


def test_repeat_pad_leaves_long_clips_alone():
    a = np.arange(9 * SR, dtype=np.float32)
    zero = chunk_audio(a, SR, 4.0, 2.0)
    rep = chunk_audio(a, SR, 4.0, 2.0, pad_mode="repeat")
    assert len(zero) == len(rep) == 4
    assert all(np.array_equal(x, y) for x, y in zip(zero, rep))


def test_repeat_pad_empty_input_falls_back_to_zeros():
    (c,) = chunk_audio(np.zeros(0, dtype=np.float32), SR, 4.0, 2.0, pad_mode="repeat")
    assert len(c) == 4 * SR and not c.any()


def test_unknown_pad_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="pad_mode"):
        chunk_audio(np.ones(SR, dtype=np.float32), SR, 4.0, 2.0, pad_mode="mirror")
