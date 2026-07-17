#!/bin/bash
# Detached driver: augment the real train manifest, link clean embeddings,
# extract embeddings for augmented chunks. Resumable at every stage.
cd /Users/tejas/Documents/VoxVerity || exit 1
set -e

echo "=== AUGMENT START $(date) ==="
.venv/bin/python -m src.preprocessing.augment \
  --manifest data/manifests/asvspoof5_train_part_chunks.csv \
  --out-audio-dir data/processed/asvspoof5_train_aug \
  --out-manifest data/manifests/asvspoof5_train_aug_chunks.csv --seed 42

echo "=== LINK CLEAN EMBEDDINGS $(date) ==="
.venv/bin/python - <<'EOF'
import csv, os
from pathlib import Path
rows = list(csv.DictReader(open("data/manifests/asvspoof5_train_aug_chunks.csv")))
src_dir = Path("data/features/asvspoof5_train_part").resolve()
dst = Path("data/features/asvspoof5_train_aug")
dst.mkdir(parents=True, exist_ok=True)
linked = 0
for r in rows:
    if r["augmentation"] == "none":
        stem = Path(r["path"]).stem
        target = dst / f"{stem}.npy"
        if not target.exists():
            os.symlink(src_dir / f"{stem}.npy", target)
            linked += 1
print("linked", linked, flush=True)
EOF

echo "=== EXTRACT START $(date) ==="
.venv/bin/python -m src.features.extract_features \
  --manifest data/manifests/asvspoof5_train_aug_chunks.csv \
  --out-dir data/features/asvspoof5_train_aug \
  --model-name facebook/wav2vec2-xls-r-300m --device mps --batch-size 16
echo "=== ALL DONE $(date) ==="
