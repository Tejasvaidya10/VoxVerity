# VoxVerity

Why audio deepfake detectors that ace benchmarks collapse on real-world audio — and a pre-registered attempt to fix it.

**[Read the full research report](REPORT.md)** · **[Model card](MODEL_CARD.md)** · **[Raw results](reports/)**

## Highlights

- **The gap, measured, not assumed**: a spoofed-speech detector trained on ASVspoof5 is evaluated cold on an independent real-world dataset (In-the-Wild) — no fine-tuning, no cherry-picking.
- **Failure made legible**: an LLM-based explainability layer turns per-speaker failure analysis into a concrete mechanism — the detector substantially uses **recording channel quality as a proxy for authenticity**.
- **Pre-registered hypothesis test**: channel-quality augmentation is proposed *before* running it, then tested — a real, partial fix (13.34% to 11.30% EER).
- **A negative result, reported honestly**: fine-tuning the SSL front-end to close the residual gap *regresses* performance (24.40% EER) — and that failure is itself the finding: the frozen, multilingually-pretrained front-end was a primary source of robustness.
- **Four sequential phases, each with a preserved report**: measure, diagnose, test the diagnosis, engineer for performance, with `reports/eval_report*.json` kept for every phase.

## Abstract

Academic audio deepfake detectors routinely report near-perfect accuracy on
benchmark datasets and are known to degrade sharply on real-world audio.
This study treats that gap as the research question. We train a
spoofed-speech detector on ASVspoof5, measure its collapse on an
independent real-world dataset (In-the-Wild), and use an LLM-based
explainability layer to make the failure mode legible. Per-speaker failure
analysis shows the detector substantially uses **recording channel quality
as a proxy for authenticity**. A pre-registered hypothesis test — training
with channel-quality augmentation — produces a real but partial fix. A
follow-up attempt to close the remaining gap by fine-tuning the SSL
front-end produces a **decisive negative result**, itself informative: it
shows the frozen, multilingually-pretrained front-end was a primary source
of the model's robustness. Full narrative, methodology, and discussion in
[REPORT.md](REPORT.md).

## Headline results

| Phase | What it measures | In-the-Wild EER |
|---|---|---|
| 1 — Baseline | The generalization gap | 13.34% |
| 3 — Channel augmentation | Testing the identified shortcut | **11.30%** (best) |
| 4 — Fine-tuning | Attempt to close the residual gap | 24.40% (regression) |

Lower is better. EER = equal error rate. See [REPORT.md](REPORT.md) for the
full per-phase methodology, the pre-registered success criteria, and the
per-speaker failure analyses behind each number. Recommended model
artifact: `checkpoints/detector_aug.pt` — see [MODEL_CARD.md](MODEL_CARD.md).

## Table of Contents

- [Project structure](#project-structure)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Reproducing](#reproducing)
- [Notes for anyone extending this](#notes-for-anyone-extending-this)
- [License](#license)

## Project structure

This was run as four sequential phases — research first, engineering
second:

1. **Measure** — quantify the benchmark-to-real-world generalization gap.
2. **Diagnose** — per-speaker, per-direction failure analysis to find the
   mechanism.
3. **Test the diagnosis** — a pre-registered augmentation experiment.
4. **Engineer for performance** — fine-tune the front-end; report the
   result honestly, including that it failed.

Each phase's full report is preserved: `reports/eval_report.json` (1),
`reports/eval_report_aug.json` (3), `reports/eval_report_ft.json` (4).

## Repository layout

```
configs/               # default.yaml, smoke.yaml, finetune.yaml — paths & hyperparameters
src/data/
  manifest.py           # build train/val/test manifests from raw data dirs
src/preprocessing/
  preprocess.py          # resample, VAD, sliding-window chunking
  augment.py              # channel augmentation (codec / noise / band-limit)
src/features/
  extract_features.py    # cache wav2vec2 SSL embeddings
src/models/
  detector.py             # LinearHead / MHFAHead classifier heads
  train.py                 # train head on cached embeddings
  finetune.py               # end-to-end SSL front-end fine-tuning
src/explain/
  acoustic_evidence.py     # real acoustic descriptors for the LLM layer
  rationale_agent.py        # LLM rationale generation + faithfulness judge
src/eval/
  metrics.py                # EER and related metrics
  harness.py                  # cross-dataset eval + rationale scoring
scripts/                # detached long-running drivers, split/failure-analysis tools
```

## Setup

```bash
uv venv --python 3.12 .venv          # webrtcvad has no working build on 3.14
uv pip install -p .venv/bin/python -r requirements.txt
```

Run everything from the repo root with `.venv/bin/python -m src.<pkg>.<module>`.
The LLM layer reads `ANTHROPIC_API_KEY` from a gitignored `.env`
(`set -a && source .env && set +a`).

## Data

- **ASVspoof5** — openly downloadable, no registration:
  https://zenodo.org/records/14498691 (ODC-By). Protocols + `flac_T_aa.tar`
  (train) + `flac_D_aa.tar` (dev) is enough to reproduce the numbers above; the
  manifest builder skips files not on disk, so partial tars are a valid subset.
- **In-the-Wild** — https://huggingface.co/datasets/mueller91/In-The-Wild
- **P2V** — *not publicly released yet* ("upon acceptance", arXiv 2508.10949);
  dropped from scope. The eval harness accepts any number of eval manifests, so
  it slots back in if it ships.

Place each under `data/raw/<dataset_name>/`; see `src/data/manifest.py`.
Datasets are not redistributed in this repository and remain subject to
their own licenses.

## Reproducing

```bash
# 1. manifests (asvspoof5 protocols are 10-col TSVs; label is col 9)
.venv/bin/python -m src.data.manifest --dataset asvspoof5 --raw-dir data/raw \
  --protocol data/raw/asvspoof5/ASVspoof5.train.tsv --audio-subdir flac_T \
  --out-name asvspoof5_train_part.csv
# 2. preprocess -> 3. embeddings (long; run detached via scripts/extract_all.sh)
# 4. split by speaker, train
.venv/bin/python scripts/split_manifest.py --manifest data/manifests/asvspoof5_train_part_chunks.csv \
  --out-train data/manifests/train.csv --out-val data/manifests/val.csv --val-fraction 0.1
.venv/bin/python -m src.models.train --config configs/default.yaml \
  --train-manifest data/manifests/train.csv --val-manifest data/manifests/val.csv \
  --features-dir data/features/asvspoof5_train_part --checkpoint-out checkpoints/detector.pt
# 5. evaluate (first manifest is the in-domain reference for the gap)
.venv/bin/python -m src.eval.harness --config configs/default.yaml \
  --checkpoint checkpoints/detector.pt \
  --manifests data/manifests/asvspoof5_dev_part_chunks.csv data/manifests/in_the_wild_chunks.csv \
  --features-dir data/features/all --out-report reports/eval_report.json
```

Phase 3 (recommended model) repeats this with `src/preprocessing/augment.py`
run first on the training manifest. Phase 4 uses `src/models/finetune.py`
and `--finetuned-checkpoint` on the harness instead of cached embeddings —
see `docs/` for the full experiment designs.

`configs/smoke.yaml` + `scripts/make_smoke_data.py` run the whole pipeline on a
synthetic TTS dataset in minutes — useful for verifying the plumbing without
touching the real data.

## Notes for anyone extending this

- Train and validate only on ASVspoof5. In-the-Wild is eval-only — mixing it
  into training defeats the point of the generalization test.
- `configs/*.yaml` centralizes all paths and hyperparameters so nothing is
  hardcoded across scripts.
- **Do not repeat the Phase 4 configuration** (small backbone, fine-tuned).
  REPORT.md §9 lays out what to try instead: fine-tune the large frozen
  backbone itself, under regularization, on adequate compute.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights and datasets have their
own licenses/terms — see [MODEL_CARD.md](MODEL_CARD.md) and § Data above.
Citation: [CITATION.cff](CITATION.cff).
