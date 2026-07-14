# Audio deepfake detection with explainable evaluation

Research project: train a spoofed-speech detector on ASVspoof5, stress-test
generalization on P2V and In-the-Wild, and add an LLM explainability layer
that turns detector evidence into a human-readable rationale.

## Why this scope

Most academic detectors report near-perfect AUC on ASVspoof but collapse on
real-world audio because they overfit to the artifact fingerprints of the
generators they were trained on. This project treats that generalization gap
as the actual research question, not an afterthought, and layers an
explainability/evaluation harness on top since that mirrors the RAG
evaluation work this project is modeled after.

## Directory structure

```
audio-deepfake-detection/
  configs/default.yaml        # paths, hyperparameters
  src/data/
    manifest.py                # build train/val/test manifests from raw data dirs
  src/preprocessing/
    preprocess.py               # resample, VAD, chunking
  src/features/
    extract_features.py         # wav2vec2 SSL embeddings
  src/models/
    detector.py                 # MHFA back-end classifier head
    train.py                    # training loop
  src/explain/
    rationale_agent.py          # LLM-generated explanation from detector evidence
  src/eval/
    metrics.py                  # EER and related metrics
    harness.py                  # cross-dataset eval + rationale quality scoring
```

## Results

Detector: frozen `wav2vec2-xls-r-300m` embeddings (mean-pooled, 4s windows /
2s stride) → MLP head. Trained on ASVspoof5 train partition only (attacks
A01–A08, 175k chunks). EER = equal error rate, lower is better.

| Eval set | What it tests | EER |
|---|---|---|
| Train-val split (A01–A08) | Same attacks, unseen speakers | **0.04%** |
| ASVspoof5 dev (A09–A16) | **Unseen attacks**, same corpus | **1.40%** |
| In-the-Wild | Real-world audio, unseen everything | **13.34%** |

**Generalization gap: 11.94 points** (In-the-Wild minus in-domain dev).

The interesting part is *where* the degradation comes from. The detector barely
flinches at synthesis methods it has never seen (0.04% → 1.40% across a fully
disjoint attack set), then falls apart when recording conditions change
(1.40% → 13.34%). So the failure is not simply memorized generator fingerprints
— something does transfer across generators. What breaks the model is **acoustic
domain shift**: real-world compression, channel effects, and background noise.
That points at codec/noise augmentation or domain-adversarial training as the
next lever, rather than merely training on more attack types.

Explainability layer: 50 flagged clips (stratified across all three eval sets)
given LLM-written rationales grounded in real acoustic evidence, each scored for
faithfulness by an LLM judge. Mean judge score **4.54 / 5**.

Full report: `reports/eval_report.json`.

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

`configs/smoke.yaml` + `scripts/make_smoke_data.py` run the whole pipeline on a
synthetic TTS dataset in minutes — useful for verifying the plumbing without
touching the real data.

## Pipeline order

1. `src/data/manifest.py` — scan raw audio + protocol files, build a CSV
   manifest (path, label, speaker_id, dataset_source)
2. `src/preprocessing/preprocess.py` — resample to 16kHz, VAD trim, chunk to
   fixed-length windows, cache to disk
3. `src/features/extract_features.py` — run wav2vec2, cache pooled embeddings
4. `src/models/train.py` — train the MHFA classifier head on ASVspoof5 only
5. `src/eval/harness.py` — evaluate on held-out ASVspoof5 test, then on P2V
   and In-the-Wild to measure the generalization gap
6. `src/explain/rationale_agent.py` — generate rationales for flagged samples,
   scored for faithfulness by `src/eval/harness.py`

## Notes

- Train and validate only on ASVspoof5. In-the-Wild is eval-only — mixing it
  into training defeats the point of the generalization test.
- `configs/default.yaml` centralizes all paths and hyperparameters so nothing
  is hardcoded across scripts.
- The current detector deliberately takes the simple path: the SSL front-end is
  **frozen** (not fine-tuned) and embeddings are **mean-pooled** over time, which
  dilutes short localized artifacts. `MHFAHead` in `src/models/detector.py` is
  wired for frame-level features when you want to go further; fine-tuning the
  front-end and training on all 5 train tars (we used 1) are the other two
  obvious levers.
