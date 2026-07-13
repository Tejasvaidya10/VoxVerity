"""Split a manifest CSV into train/val by speaker (no speaker overlap)."""

import argparse
import csv
import random
from pathlib import Path


def split_manifest(rows: list, val_fraction: float, seed: int) -> tuple:
    speakers = sorted({r["speaker_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val = max(1, round(len(speakers) * val_fraction))
    val_speakers = set(speakers[:n_val])
    train = [r for r in rows if r["speaker_id"] not in val_speakers]
    val = [r for r in rows if r["speaker_id"] in val_speakers]
    return train, val


def _write(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-train", type=Path, required=True)
    p.add_argument("--out-val", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    train, val = split_manifest(rows, args.val_fraction, args.seed)
    _write(train, args.out_train)
    _write(val, args.out_val)
