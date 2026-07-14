"""
Main evaluation harness. Runs the trained detector on ASVspoof5 test (in-domain)
and then on P2V / In-the-Wild (out-of-domain), reporting the EER gap between
them — this gap is the paper's central finding: models look great in-domain
and collapse on real-world conditions. Also samples flagged clips and runs
them through the rationale agent + judge for explanation quality.

Usage:
    python harness.py --config configs/default.yaml \
        --checkpoint checkpoints/detector.pt \
        --manifests data/manifests/asvspoof5_test.csv data/manifests/p2v.csv data/manifests/in_the_wild.csv \
        --features-dir data/features \
        --out-report reports/eval_report.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.detector import build_head
from src.eval.metrics import summarize_scores
from src.explain.acoustic_evidence import compute_evidence_from_file
from src.explain.rationale_agent import DetectionEvidence, generate_rationale, score_rationale


def build_evidence(path: str, spoof_probability: float) -> DetectionEvidence:
    feats = compute_evidence_from_file(path)
    return DetectionEvidence(
        file_id=Path(path).stem,
        spoof_probability=spoof_probability,
        high_attention_time_ranges=[],  # LinearHead has no attention map
        **feats,
    )


def load_model(config: dict, checkpoint_path: Path, device: str):
    model = build_head(config["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def score_manifest(model, manifest_path: Path, features_dir: Path, device: str) -> dict:
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    bonafide_scores, spoof_scores = [], []
    per_file_scores = {}

    with torch.no_grad():
        for row in rows:
            stem = Path(row["path"]).stem
            emb_path = features_dir / f"{stem}.npy"
            if not emb_path.exists():
                continue
            emb = torch.from_numpy(np.load(emb_path).astype(np.float32)).unsqueeze(0).to(device)
            logit = model(emb).item()
            score = 1 / (1 + np.exp(-logit))  # sigmoid -> spoof probability

            per_file_scores[row["path"]] = score
            if row["label"] == "spoof":
                spoof_scores.append(score)
            else:
                bonafide_scores.append(score)

    metrics = summarize_scores(np.array(bonafide_scores), np.array(spoof_scores))
    return {"metrics": metrics, "per_file_scores": per_file_scores}


def run_generalization_eval(config: dict, model, manifests: list, features_dir: Path, device: str) -> dict:
    results = {}
    for manifest_path in manifests:
        dataset_name = Path(manifest_path).stem
        print(f"Scoring {dataset_name}...")
        results[dataset_name] = score_manifest(model, Path(manifest_path), features_dir, device)
        print(f"  EER: {results[dataset_name]['metrics']['eer_pct']:.2f}%")
    return results


def run_rationale_eval(config: dict, generalization_results: dict, sample_size: int) -> list:
    import anthropic
    client = anthropic.Anthropic()

    # Sample flagged (high spoof-probability) clips, stratified across datasets.
    # A global top-N sort would hand every slot to the in-domain set (its scores
    # are the most confident), leaving the out-of-domain failures — the ones this
    # project exists to explain — unexplained.
    sample = sample_flagged(generalization_results, sample_size)

    judged_results = []
    for dataset_name, path, score in sample:
        try:
            evidence = build_evidence(path, score)
        except Exception as e:
            print(f"Skipping rationale for {path}: {e}")
            continue
        model = config["explain"]["anthropic_model"]
        rationale = generate_rationale(client, evidence, model=model,
                                       max_tokens=config["explain"]["max_tokens"])
        judged = score_rationale(client, evidence, rationale, model=model)
        judged_results.append({
            "dataset": dataset_name, "path": path, "score": score,
            "rationale": rationale, "judge": judged,
        })
    return judged_results


def sample_flagged(generalization_results: dict, sample_size: int) -> list:
    """Pick flagged clips evenly across datasets, highest-confidence first within
    each. Returns [(dataset_name, path, score), ...]."""
    by_dataset = {}
    for dataset_name, result in generalization_results.items():
        flagged = [(dataset_name, path, score)
                   for path, score in result["per_file_scores"].items() if score > 0.5]
        flagged.sort(key=lambda x: -x[2])
        if flagged:
            by_dataset[dataset_name] = flagged

    if not by_dataset:
        return []

    per_dataset = max(1, sample_size // len(by_dataset))
    sample = []
    for flagged in by_dataset.values():
        sample.extend(flagged[:per_dataset])

    # backfill any remaining slots from the largest pools
    if len(sample) < sample_size:
        taken = {id(x) for x in sample}
        leftovers = [x for flagged in by_dataset.values() for x in flagged[per_dataset:]]
        leftovers.sort(key=lambda x: -x[2])
        sample.extend(leftovers[:sample_size - len(sample)])
    return sample[:sample_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifests", nargs="+", required=True,
                         help="First manifest should be the in-domain (ASVspoof5) test set")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, default=Path("reports/eval_report.json"))
    parser.add_argument("--skip-rationales", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, args.checkpoint, device)

    generalization_results = run_generalization_eval(config, model, args.manifests, args.features_dir, device)

    report = {
        "generalization": {
            name: r["metrics"] for name, r in generalization_results.items()
        },
    }

    # report the gap between in-domain (first manifest) and each out-of-domain set
    in_domain_name = Path(args.manifests[0]).stem
    in_domain_eer = report["generalization"][in_domain_name]["eer_pct"]
    report["generalization_gap_pct"] = {
        name: r["eer_pct"] - in_domain_eer
        for name, r in report["generalization"].items()
        if name != in_domain_name
    }

    if not args.skip_rationales:
        report["rationale_eval"] = run_rationale_eval(
            config, generalization_results, config["eval"]["rationale_sample_size"]
        )

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {args.out_report}")
    print(json.dumps(report["generalization"], indent=2))
    print("Generalization gap (out-of-domain EER minus in-domain EER):")
    print(json.dumps(report["generalization_gap_pct"], indent=2))


if __name__ == "__main__":
    main()
