"""Reuse baseline embeddings for variant windows whose audio is byte-identical."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.phase5_run import link_identical_embeddings  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant-manifest", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--baseline-features", type=Path, required=True)
    p.add_argument("--variant-features", type=Path, required=True)
    p.add_argument("--model-name", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--pooling", default="mean")
    a = p.parse_args()
    link_identical_embeddings(a.variant_manifest, a.baseline_manifest, a.baseline_features,
                              a.variant_features, a.model_name, a.layer, a.pooling)
