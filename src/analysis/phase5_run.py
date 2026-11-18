"""
Phase 5 run helpers: deterministic dev-control slice, reuse of baseline
embeddings for byte-identical variant windows, and the reproduction gate.
"""

import csv
import filecmp
import json
import os
import random
from pathlib import Path

from src.eval.clip_scores import clip_id_from_chunk_path, read_scores
from src.features.extract_features import features_meta, prepare_features_dir

GATE_TOLERANCE_PTS = 0.05


def sample_clips(manifest_path: Path, n: int, seed: int, out_path: Path) -> None:
    """Seeded sample of n clips (rows sorted by path first, so input order
    doesn't matter); written sorted by path."""
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, sorted(reader, key=lambda r: r["path"])
    picked = sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda r: r["path"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(picked)
    print(f"Sampled {len(picked)} of {len(rows)} clips -> {out_path}")


def link_identical_embeddings(variant_manifest: Path, baseline_manifest: Path,
                              baseline_features: Path, variant_features: Path,
                              model_name: str, layer: int, pooling: str) -> int:
    """Record the variant features folder's provenance first (so the
    extractor later resumes into it), then symlink each baseline embedding
    whose window audio is byte-identical to the variant's window. Returns
    the number of new links."""
    variant_features = Path(variant_features)
    prepare_features_dir(variant_features,
                         features_meta(variant_manifest, model_name, layer, pooling),
                         allow_legacy=False)
    with open(baseline_manifest) as f:
        base = {(clip_id_from_chunk_path(r["path"]), str(r["chunk_idx"])): r["path"]
                for r in csv.DictReader(f)}
    linked = 0
    with open(variant_manifest) as f:
        for r in csv.DictReader(f):
            key = (clip_id_from_chunk_path(r["path"]), str(r["chunk_idx"]))
            stem = Path(r["path"]).stem
            dst = variant_features / f"{stem}.npy"
            src_wav = base.get(key)
            if src_wav is None or dst.exists():
                continue
            src_npy = Path(baseline_features) / f"{Path(src_wav).stem}.npy"
            if src_npy.exists() and filecmp.cmp(src_wav, r["path"], shallow=False):
                os.symlink(src_npy.resolve(), dst)
                linked += 1
    print(f"Linked {linked} byte-identical windows into {variant_features}")
    return linked


def check_gate(scores_csv: Path, reference_report: Path,
               dataset_key: str = "in_the_wild_chunks") -> tuple:
    """The restored baseline must reproduce the Phase 3 In-the-Wild result:
    same real/fake window counts and EER within 0.05 points."""
    from src.analysis.phase5_report import eer_pct
    df = read_scores(scores_csv)
    ref = json.loads(Path(reference_report).read_text())["generalization"][dataset_key]
    n_b, n_s = int((df["label"] == "bonafide").sum()), int((df["label"] == "spoof").sum())
    eer = eer_pct(df)
    detail = (f"EER {eer:.4f}% vs reference {ref['eer_pct']:.4f}%; windows {n_b} real + {n_s} fake "
              f"vs {ref['n_bonafide']} + {ref['n_spoof']}")
    if (n_b, n_s) != (ref["n_bonafide"], ref["n_spoof"]):
        return False, f"GATE FAIL (windows differ): {detail}"
    if abs(eer - ref["eer_pct"]) > GATE_TOLERANCE_PTS:
        return False, f"GATE FAIL (EER differs by more than {GATE_TOLERANCE_PTS} pts): {detail}"
    return True, f"GATE PASS: {detail}"
