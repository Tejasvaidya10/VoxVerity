"""Phase 5 reproduction gate: exit 0 only if the baseline reproduces Phase 3."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.phase5_run import check_gate  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("reports/eval_report_aug.json"))
    a = p.parse_args()
    ok, msg = check_gate(a.scores, a.reference)
    print(msg)
    sys.exit(0 if ok else 1)
