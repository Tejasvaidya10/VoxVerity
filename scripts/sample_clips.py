"""Deterministic clip sample (Phase 5 dev-control slice)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.phase5_run import sample_clips  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--n", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sample_clips(a.manifest, a.n, a.seed, a.out)
