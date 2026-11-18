"""Phase 5 analysis: summary.json, plots, speakers.csv, listening pack.

    .venv/bin/python scripts/phase5_analysis.py --scores-root reports/phase5/scores \
        --measurements reports/phase5/measurements \
        --raw-manifest data/manifests/in_the_wild.csv --out reports/phase5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.phase5_report import run_analysis  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores-root", type=Path, default=Path("reports/phase5/scores"))
    p.add_argument("--measurements", type=Path, default=Path("reports/phase5/measurements"))
    p.add_argument("--raw-manifest", type=Path, default=Path("data/manifests/in_the_wild.csv"))
    p.add_argument("--out", type=Path, default=Path("reports/phase5"))
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    s = run_analysis(a.scores_root, a.measurements, a.raw_manifest, a.out, n_boot=a.n_boot)
    for k, v in s["predictions"].items():
        print(f"{k}: {'HOLDS' if v['holds'] else 'does not hold'} | {v['reason']}")
    print(json.dumps({n: e["itw"] for n, e in s["variants"].items()}, indent=2))
