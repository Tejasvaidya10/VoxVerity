"""Per-direction and per-speaker error breakdown for a trained checkpoint."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def error_breakdown(scores, labels, speakers, threshold: float) -> dict:
    scores, labels, speakers = map(np.asarray, (scores, labels, speakers))
    bona, spoof = labels == "bonafide", labels == "spoof"
    fa_rate = float((scores[bona] >= threshold).mean()) if bona.any() else 0.0
    fr_rate = float((scores[spoof] < threshold).mean()) if spoof.any() else 0.0

    def per_speaker(mask, err_cond):
        d = defaultdict(lambda: [0, 0])
        for s, sp, m, e in zip(scores, speakers, mask, err_cond):
            if m:
                d[sp][1] += 1
                if e:
                    d[sp][0] += 1
        ranked = sorted(d.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1)))
        return [(sp, bad, tot) for sp, (bad, tot) in ranked]

    return {
        "fa_rate": fa_rate,
        "fr_rate": fr_rate,
        "worst_fa_speakers": per_speaker(bona, scores >= threshold),
        "worst_fr_speakers": per_speaker(spoof, scores < threshold),
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    import yaml
    from src.models.detector import build_head

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--finetuned-checkpoint", type=Path, default=None)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--min-count", type=int, default=50)
    args = p.parse_args()

    if bool(args.checkpoint) == bool(args.finetuned_checkpoint):
        p.error("provide exactly one of --checkpoint / --finetuned-checkpoint")

    config = yaml.safe_load(open(args.config))
    rows = list(csv.DictReader(open(args.manifest)))

    if args.finetuned_checkpoint:
        from src.models.finetune import load_finetuned, score_manifest_audio
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = load_finetuned(args.finetuned_checkpoint, config["model"], device)
        per_file = score_manifest_audio(model, rows, device=device)
        scores = [per_file[r["path"]] for r in rows if r["path"] in per_file]
        labels = [r["label"] for r in rows if r["path"] in per_file]
        speakers = [r["speaker_id"] for r in rows if r["path"] in per_file]
    else:
        if not args.features_dir:
            p.error("--features-dir is required with --checkpoint")
        model = build_head(config["model"])
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model.eval()
        scores, labels, speakers = [], [], []
        with torch.no_grad():
            for r in rows:
                f = args.features_dir / f"{Path(r['path']).stem}.npy"
                if not f.exists():
                    continue
                emb = torch.from_numpy(np.load(f).astype(np.float32)).unsqueeze(0)
                scores.append(1 / (1 + np.exp(-model(emb).item())))
                labels.append(r["label"])
                speakers.append(r["speaker_id"])

    out = error_breakdown(scores, labels, speakers, args.threshold)
    print(f"n={len(scores)} threshold={args.threshold}")
    print(f"false alarms (real->fake): {out['fa_rate']*100:.1f}%")
    print(f"misses (fake->real):       {out['fr_rate']*100:.1f}%")
    for key, title in (("worst_fa_speakers", "FALSE ALARMS"), ("worst_fr_speakers", "MISSES")):
        print(f"\nworst speakers, {title}:")
        shown = 0
        for sp, bad, tot in out[key]:
            if tot >= args.min_count and bad > 0:
                print(f"  {sp}: {bad}/{tot} ({bad/tot*100:.0f}%)")
                shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
