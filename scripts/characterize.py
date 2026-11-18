"""Measure every clip of a raw manifest for Phase 5 (see src/analysis/measurements.py).

    .venv/bin/python scripts/characterize.py --manifest data/manifests/in_the_wild.csv \
        --dataset in_the_wild --out-dir reports/phase5/measurements --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.measurements import characterize_manifest  # noqa: E402
from src.preprocessing.preprocess import resolve_settings  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("reports/phase5/measurements"))
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    characterize_manifest(a.manifest, a.dataset, a.out_dir, resolve_settings(a.config, {}), a.workers)
