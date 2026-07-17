import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location(
    "failure_analysis", Path("scripts/failure_analysis.py"))
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def test_error_breakdown_directions_and_speakers():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.05])
    labels = np.array(["bonafide", "bonafide", "spoof", "spoof", "bonafide", "spoof"])
    speakers = np.array(["a", "a", "b", "b", "c", "c"])
    out = fa.error_breakdown(scores, labels, speakers, threshold=0.5)
    assert abs(out["fa_rate"] - 2 / 3) < 1e-9   # bonafide 0.9 and 0.9 flagged
    assert abs(out["fr_rate"] - 2 / 3) < 1e-9   # spoof 0.2 and 0.05 passed
    fa_speakers = {s for s, bad, tot in out["worst_fa_speakers"] if bad > 0}
    assert fa_speakers == {"a", "c"}
