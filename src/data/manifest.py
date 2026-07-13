"""
Build a unified manifest CSV from raw dataset directories.

Expected raw layout (you create this after downloading each dataset):

data/raw/asvspoof5/
    audio/*.flac
    protocol.txt          # ASVspoof-style: speaker_id file_id - attack_type label

data/raw/p2v/
    audio/*.wav
    metadata.csv           # subject_id, file_id, label, generation_method

data/raw/in_the_wild/
    audio/*.wav
    meta.csv                # file, speaker, label

Output: data/manifests/<dataset>.csv with columns:
    path, label, speaker_id, dataset_source, split
label is "bonafide" or "spoof"
"""

import csv
from pathlib import Path
import argparse


def build_asvspoof5_manifest(raw_dir: Path, out_path: Path,
                             protocol_path: Path | None = None,
                             audio_subdir: str = "audio") -> None:
    """Handles both the official ASVspoof5 TSV (10 space-separated columns:
    speaker, file_id, gender, -, -, -, codec, attack, label, -) and the
    simple 5-column layout used by the smoke dataset."""
    protocol_path = protocol_path or raw_dir / "protocol.txt"
    audio_dir = raw_dir / audio_subdir
    rows = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:      # official ASVspoof5 format
                speaker_id, file_id = parts[0], parts[1]
                attack_type, label = parts[7], parts[8]
            elif len(parts) >= 5:    # legacy/smoke format
                speaker_id, file_id, _, attack_type, label = parts[:5]
            else:
                continue
            audio_path = audio_dir / f"{file_id}.flac"
            if not audio_path.exists():
                continue
            rows.append({
                "path": str(audio_path),
                "label": "bonafide" if label.lower() == "bonafide" else "spoof",
                "speaker_id": speaker_id,
                "dataset_source": "asvspoof5",
                "attack_type": attack_type,
            })
    _write_csv(rows, out_path)


def build_p2v_manifest(raw_dir: Path, out_path: Path) -> None:
    meta_path = raw_dir / "metadata.csv"
    audio_dir = raw_dir / "audio"
    rows = []
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            audio_path = audio_dir / r["file_id"]
            if not audio_path.exists():
                continue
            rows.append({
                "path": str(audio_path),
                "label": "bonafide" if r["label"].lower() in ("real", "bonafide") else "spoof",
                "speaker_id": r["subject_id"],
                "dataset_source": "p2v",
                "attack_type": r.get("generation_method", "unknown"),
            })
    _write_csv(rows, out_path)


def build_in_the_wild_manifest(raw_dir: Path, out_path: Path) -> None:
    meta_path = raw_dir / "meta.csv"
    audio_dir = raw_dir / "audio"
    rows = []
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            audio_path = audio_dir / r["file"]
            if not audio_path.exists():
                continue
            rows.append({
                "path": str(audio_path),
                "label": "bonafide" if r["label"].lower() in ("bona-fide", "bonafide", "real") else "spoof",
                "speaker_id": r["speaker"],
                "dataset_source": "in_the_wild",
                "attack_type": "unknown",
            })
    _write_csv(rows, out_path)


def _write_csv(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"WARNING: no rows written to {out_path} — check raw data paths")
        return
    fieldnames = ["path", "label", "speaker_id", "dataset_source", "attack_type"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_spoof = sum(1 for r in rows if r["label"] == "spoof")
    print(f"Wrote {len(rows)} rows to {out_path} ({n_spoof} spoof, {len(rows) - n_spoof} bonafide)")


BUILDERS = {
    "asvspoof5": build_asvspoof5_manifest,
    "p2v": build_p2v_manifest,
    "in_the_wild": build_in_the_wild_manifest,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=BUILDERS.keys(), required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--out-name", default=None,
                        help="output CSV filename (default: <dataset>.csv)")
    parser.add_argument("--protocol", type=Path, default=None,
                        help="asvspoof5 only: explicit protocol file (one per partition)")
    parser.add_argument("--audio-subdir", default="audio",
                        help="asvspoof5 only: audio dir inside raw dir (e.g. flac_T)")
    args = parser.parse_args()

    out_path = args.out_dir / (args.out_name or f"{args.dataset}.csv")
    if args.dataset == "asvspoof5":
        build_asvspoof5_manifest(args.raw_dir / args.dataset, out_path,
                                 protocol_path=args.protocol,
                                 audio_subdir=args.audio_subdir)
    else:
        BUILDERS[args.dataset](args.raw_dir / args.dataset, out_path)
