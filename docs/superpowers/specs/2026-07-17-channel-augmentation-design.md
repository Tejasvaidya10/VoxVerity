# Channel-augmentation experiment — design

**Date:** 2026-07-17
**Goal:** Test the failure-analysis hypothesis that the detector uses recording
quality as a proxy for authenticity, by decorrelating channel quality from
labels in the training data and measuring the effect on the generalization gap.

## Hypothesis under test

From the In-the-Wild failure analysis (README): missed fakes cluster on
degraded/archival source audio (JFK 36%, Alan Watts 35% miss rates), false
alarms cluster on clean produced audio (Norm MacDonald, John Cleese ~60%), and
the operating threshold collapses out of domain (0.66 → 0.0029). If the model
keys on channel quality, making quality uninformative at training time should
shrink the In-the-Wild EER; if the gap persists, the residual is genuinely
harder domain mismatch.

## Decisions (from brainstorming)

1. **Strategy: randomized per-chunk.** Each training chunk independently stays
   clean (50%) or receives one randomly chosen channel transform (50%),
   applied identically to both classes. Same dataset size as baseline → clean
   A/B comparison.
2. **Success criterion:** In-the-Wild EER drops below ~10% (from 13.34%)
   while ASVspoof5-dev EER stays under ~3% (from 1.40%). Any large ITW
   improvement with dev intact confirms the hypothesis.
3. **Architecture: offline augmentation stage** between preprocessing and
   feature extraction (Approach A). On-the-fly augmentation is impossible
   (training consumes cached embeddings); embedding-space noise would not
   test the hypothesis (codec artifacts are nonlinear in embedding space).

## Components

### 1. `src/preprocessing/augment.py` (new)

- `augment_manifest(manifest_path, out_audio_dir, out_manifest_path, seed,
  augment_prob=0.5)` — reads a chunk manifest; per row, seeded RNG decides
  clean vs augmented. Clean rows pass through with their **original path**
  (so their cached embeddings are reused). Augmented rows are written to
  `out_audio_dir` as 16 kHz PCM_16 wav with a distinct stem suffix
  (`<stem>__aug_<transform>`), and the output manifest row records
  `augmentation: <name>` (clean rows: `none`).
- Transform families, one chosen uniformly per augmented chunk
  (parameters drawn per chunk from the listed ranges):
  - `codec`: ffmpeg round-trip, format/bitrate uniformly from
    MP3 {32, 64, 96} kbps and Opus {16, 32, 48} kbps, decoded back to
    16 kHz mono wav. ffmpeg invoked via subprocess; a chunk falls back to
    `noise` if ffmpeg fails (logged, counted).
  - `noise`: additive white or pink noise, SNR uniform in [5, 25] dB.
  - `bandlimit`: low-pass with cutoff uniform in [3400, 6000] Hz, or
    telephone band-pass 300–3400 Hz (50/50), via scipy sos filters.
- Deterministic: same seed → same assignment and parameters (RNG derived
  from `(seed, row_index)`).
- CLI mirrors the other pipeline scripts:
  `python -m src.preprocessing.augment --manifest … --out-audio-dir … --out-manifest … --seed 42`.

### 2. Extraction reuse

Run the existing `extract_features` on the augmented manifest with a new
out-dir `data/features/asvspoof5_train_aug`. Pre-populate it by symlinking the
clean chunks' existing `.npy` files (their stems are unchanged), so only the
~50% augmented chunks are extracted (~1.5 h detached via a variant of
`scripts/extract_all.sh`).

### 3. Retraining and evaluation

- Split the augmented manifest with the **same seed** (42) as the baseline so
  the speaker split is identical.
- Train with `configs/default.yaml` hyperparameters, unchanged, to
  `checkpoints/detector_aug.pt`. Baseline `checkpoints/detector.pt` and
  `reports/eval_report.json` are never overwritten.
- Evaluate with the existing harness (dev + In-the-Wild + val manifests,
  untouched eval embeddings) to `reports/eval_report_aug.json`, including the
  stratified rationales.
- Rerun the per-speaker failure analysis (promote the ad-hoc script to
  `scripts/failure_analysis.py`) on both checkpoints for a before/after
  comparison of the archival-audio miss pattern and threshold collapse.

### 4. Reporting

README gets an "Experiment: channel augmentation" subsection with the A/B
table (baseline vs augmented: val, dev, ITW EER + gap), the outcome relative
to the pre-registered success bar, and the updated failure-analysis read.

## Testing

- Unit tests for each transform using synthetic signals, measured with the
  existing `acoustic_evidence` functions: band-limited white noise has
  near-zero high-band energy fraction; noise-augmented tone has higher
  spectral flatness than the clean tone; codec round-trip preserves length
  (±1 frame tolerance, padded/trimmed to exact) and sample rate.
- Determinism test: two runs with the same seed produce identical manifests.
- Label-independence test: augmentation assignment rate is ~50% for both
  classes on a synthetic manifest.
- Integration: augment the smoke dataset's chunk manifest and run
  extraction+training on it before touching the real data.

## Out of scope

- Training on In-the-Wild or any eval-set data (would invalidate the
  benchmark).
- Fine-tuning the SSL front-end, MHFA head, additional train tars —
  deliberately deferred until this experiment's result is known.
- Intensity sweeps (single 50% setting only, per the chosen success
  criterion).
