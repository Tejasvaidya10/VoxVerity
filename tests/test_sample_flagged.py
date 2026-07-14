from src.eval.harness import sample_flagged


def _results():
    return {
        "in_domain": {"per_file_scores": {f"/a/{i}.wav": 0.99 - i * 1e-4 for i in range(40)}},
        "out_of_domain": {"per_file_scores": {f"/b/{i}.wav": 0.80 - i * 1e-4 for i in range(40)}},
        "unflagged": {"per_file_scores": {f"/c/{i}.wav": 0.1 for i in range(10)}},
    }


def test_sample_is_stratified_not_dominated_by_top_scores():
    sample = sample_flagged(_results(), sample_size=10)
    datasets = {d for d, _, _ in sample}
    assert datasets == {"in_domain", "out_of_domain"}  # unflagged has no clips over 0.5
    counts = {d: sum(1 for x in sample if x[0] == d) for d in datasets}
    assert counts["out_of_domain"] == 5  # would be 0 under a global top-N sort
    assert len(sample) == 10


def test_highest_confidence_first_within_dataset():
    sample = sample_flagged(_results(), sample_size=4)
    ood = [s for d, _, s in sample if d == "out_of_domain"]
    assert ood == sorted(ood, reverse=True)


def test_backfills_when_one_pool_is_small():
    results = {
        "big": {"per_file_scores": {f"/a/{i}.wav": 0.9 for i in range(20)}},
        "small": {"per_file_scores": {"/b/0.wav": 0.7}},
    }
    sample = sample_flagged(results, sample_size=10)
    assert len(sample) == 10  # small pool has 1; the rest backfills from big
    assert sum(1 for d, _, _ in sample if d == "small") == 1


def test_empty_when_nothing_flagged():
    assert sample_flagged({"x": {"per_file_scores": {"/a.wav": 0.2}}}, 10) == []
