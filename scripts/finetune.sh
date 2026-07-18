#!/bin/bash
cd /Users/tejas/Documents/VoxVerity || exit 1
echo "=== FINETUNE START $(date) ==="
.venv/bin/python -m src.models.finetune --config configs/finetune.yaml \
  --train-manifest data/manifests/train_aug.csv \
  --val-manifest data/manifests/val_aug.csv \
  --checkpoint-out checkpoints/detector_ft.pt --device mps
echo "=== FINETUNE DONE $(date) ==="
