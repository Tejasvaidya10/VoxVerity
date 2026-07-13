#!/bin/bash
# Detached driver for the long embedding-extraction job.
# Resumable: each dataset's extractor skips chunks whose .npy already exists.
cd /Users/tejas/Documents/VoxVerity || exit 1

for d in asvspoof5_train_part asvspoof5_dev_part in_the_wild; do
  echo "=== START $d $(date) ==="
  .venv/bin/python -m src.features.extract_features \
    --manifest "data/manifests/${d}_chunks.csv" \
    --out-dir "data/features/$d" \
    --model-name facebook/wav2vec2-xls-r-300m \
    --device mps --batch-size 16
  echo "=== DONE $d $(date) ==="
done

echo "ALL EMBEDDING EXTRACTION COMPLETE $(date)"
