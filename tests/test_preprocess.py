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
