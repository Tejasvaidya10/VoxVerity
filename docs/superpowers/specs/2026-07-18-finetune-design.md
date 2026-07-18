# Phase 4: SSL fine-tuning with attention pooling — design

**Date:** 2026-07-18
**Goal:** Engineer a better detector (not another controlled experiment):
partially fine-tune a wav2vec2 front-end end-to-end with MHFA attention
pooling, trained on the Phase-3 augmented data, and evaluate against the same
pre-registered bar.

## Decisions (from brainstorming)

1. **Compute: M4 laptop, partial fine-tune.** Full fine-tuning of xls-r-300m
   on MPS is a multi-day job at the edge of 16 GB; cloud was declined.
2. **Model: `facebook/wav2vec2-base`** (95M, 12 layers, 768-dim). Freeze the
   CNN feature extractor and transformer layers 0–5; fine-tune layers 6–11.
3. **Pooling: MHFA attention head** (`MHFAHead` in `src/models/detector.py`,
   currently unused) over frame-level hidden states, replacing mean pooling —
   trainable end-to-end with the top layers.
4. **Data: Phase-3 augmented manifest** (`data/manifests/train_aug.csv` /
   `val_aug.csv`, same seed-42 speaker split). Augmentation strictly helped;
   audio is read directly from disk (no cached embeddings — that shortcut
   does not survive fine-tuning).
5. **Success bar (carried forward):** In-the-Wild EER < 10% AND dev < 3%,
   against the Phase-3 baseline (11.30 / 1.18).
6. **Stated limitation (already in README):** backbone swap + tuning + pooling
   change land together; this phase optimizes performance, not attribution.

## Components

### 1. `src/models/finetune.py` (new)

End-to-end trainer:

- `FinetuneModel(nn.Module)`: wraps `Wav2Vec2Model` + `MHFAHead`. Forward:
  raw waveform batch → frozen CNN/bottom layers → tuned top layers → last
  hidden state (batch, time, 768) → MHFA → logit. `freeze(model, n_frozen_layers=6)`
  helper sets `requires_grad=False` on `feature_extractor`, `feature_projection`,
  positional conv, and `encoder.layers[:6]`.
- `AudioChunkDataset`: reads manifest rows, returns `(waveform float32[64000],
  label)`. Chunks are fixed 4 s so batches need no padding logic (defensive
  pad/trim to 64000 anyway).
- Optimizer: AdamW with two param groups — head at `training.learning_rate`
  (1e-4), unfrozen backbone at `learning_rate × ssl_learning_rate_scale`
  (0.05 → 5e-6).
- Loop: `max_epochs: 2`; validation every 5,000 steps (val loss on the
  augmented val split); best checkpoint saved each time val improves →
  training is interruption-tolerant. Progress printed every 200 steps with
  running loss and chunks/sec (flush=True, for the log monitor).
- Checkpoint format: `{"backbone": model_name, "state_dict": …}` saved to
  `checkpoints/detector_ft.pt` (state dict of the whole FinetuneModel).
- Device: MPS; batch size from new config key.

### 2. `configs/finetune.yaml` (new)

Copy of `default.yaml` with a `finetune:` section:
`model_name: facebook/wav2vec2-base`, `n_frozen_layers: 6`, `batch_size: 8`,
`max_epochs: 2`, `val_every_steps: 5000`, `num_workers: 2` (dataloader),
plus the MHFA dims (`embedding_dim: 768`, `num_attention_heads: 32`,
`compression_dim: 128`).

### 3. Scoring path for fine-tuned checkpoints

`src/eval/harness.py` gains `--finetuned-checkpoint` mode: when passed, it
loads `FinetuneModel`, scores manifests by running audio through the full
model (batched, `torch.no_grad`, MPS) instead of loading cached `.npy`
embeddings. Everything else (EER, gap, stratified rationales, report format)
unchanged. Scoring throughput ≈ forward-only wav2vec2-base (~40–60 chunks/s
batched) → dev+ITW+val ≈ 1.5–2 h, run detached.

### 4. Runs (all detached via `nohup` scripts, monitored)

- `scripts/finetune.sh`: the training run (~4–6 h/epoch × 2, overnight).
- Evaluation run → `reports/eval_report_ft.json` (baseline reports untouched).
- Failure analysis A/B via existing `scripts/failure_analysis.py` — it loads
  MLP-head checkpoints, so add the same `--finetuned-checkpoint` mode there
  (shared loading helper in `finetune.py`: `load_finetuned(path, config, device)`).

### 5. Reporting

README Phase 4 section updated with the results table (Phase-3 augmented
baseline vs fine-tuned), the bar verdict, threshold calibration note, and
failure-analysis comparison. Memory updated with the outcome.

## Testing

- Unit: `FinetuneModel` forward on random batch returns (batch,) logits;
  `freeze` leaves exactly the top 6 layers + MHFA trainable (count param
  groups); `AudioChunkDataset` returns fixed-length tensors and correct
  labels; two-group optimizer has the right LRs.
- Integration (smoke): 1 short fine-tune run on the smoke augmented manifest
  (`max_epochs: 1`, `val_every_steps: 50`, batch 8) — converging loss,
  checkpoint written, harness `--finetuned-checkpoint` scores the smoke eval
  sets end-to-end.
- No test touches the real data until the smoke path is green.

## Out of scope

- Fine-tuning xls-r-300m; cloud training; hyperparameter sweeps.
- Re-running Phase 1–3 artifacts (read-only).
- Training on In-the-Wild (still forbidden).
