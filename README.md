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

## Setup

```bash
pip install torch torchaudio transformers soundfile librosa scikit-learn \
    webrtcvad anthropic pandas numpy --break-system-packages
```

## Data you need to download yourself (not included, requires registration)

- ASVspoof5: https://www.asvspoof.org/ (train/dev/eval protocol files + audio)
- P2V (Perturbed Public Voices): https://arxiv.org/abs/2508.10949 (check for
  release repo/access instructions)
- In-the-Wild: https://deepfake-total.com/in_the_wild

Place each under `data/raw/<dataset_name>/` following the structure documented
in `src/data/manifest.py`.

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

- Train and validate only on ASVspoof5. P2V and In-the-Wild are eval-only —
  mixing them into training defeats the point of the generalization test.
- `configs/default.yaml` centralizes all paths and hyperparameters so nothing
  is hardcoded across scripts.
