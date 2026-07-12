# End-to-end working pipeline — design

**Date:** 2026-07-12
**Goal:** Take the existing VoxVerity skeleton (audio deepfake detection with
explainable evaluation) from "code that has never run" to a verified,
end-to-end working pipeline, then run it on real data.

## Decisions made during brainstorming

1. **Session goal:** working end-to-end pipeline first, real research run
   second. Both in scope.
2. **LLM explainability layer:** real Anthropic API calls (not mocked),
   verified on a handful of samples.
3. **Data:**
   - ASVspoof5 is now fully open on Zenodo (ODC-By license, no registration).
     Downloading a subset: protocols + 1 train tar (7.5 GB) + 1 dev tar
     (6.6 GB). `manifest.py` skips missing files, so partial tars form a valid
     subset; more tars can be added later without rework.
   - In-the-Wild (8.2 GB zip from Hugging Face) is the out-of-domain eval set.
   - **P2V is not publicly released yet** ("upon acceptance" per the paper).
     Dropped from the plan for now; the harness accepts any number of eval
     manifests, so it slots in when released.
4. **Smoke test before real data:** a tiny synthetic dataset generated with
   macOS `say` TTS — clean renders as "bonafide", degraded renders (low-pass
   + noise + pitch flattening) as "spoof" — laid out in the exact raw-data
   formats `manifest.py` expects for all three datasets. Proves every stage
   end-to-end with zero downloads while the real data downloads.

## Environment

- Python 3.12 venv at `.venv` (created with `uv`; system Python 3.14 lacks
  wheels for `webrtcvad`), all of `requirements.txt` installed.
- Apple M4, 16 GB RAM, ~200 GB free disk. Heavy step is wav2vec2 embedding
  extraction on MPS; classifier-head training on cached embeddings is cheap.

## Code changes required (existing skeleton is otherwise sound)

1. **Real acoustic evidence** — `src/eval/harness.py` currently passes
   placeholder `-1.0` values for `pitch_variance` and `spectral_flatness` to
   the rationale agent. Add a small feature module (librosa: pitch via
   `pyin`/`yin`, spectral flatness) computed on flagged clips' audio and pass
   real values through, so rationales are grounded in actual evidence.
   `high_attention_time_ranges` stays empty while `LinearHead` is the
   classifier (documented, not silently fake).
2. **Import hygiene** — replace the `sys.path.append` hacks and bare
   `from detector import ...` imports with a proper `src` package
   (`__init__.py` files, `python -m src.models.train` style entry points) so
   every script runs from the repo root.
3. **Anthropic model ID** — verify/update `claude-sonnet-4-6` in
   `configs/default.yaml` and `rationale_agent.py` against current model IDs.
4. **Smoke config** — `configs/smoke.yaml`: `facebook/wav2vec2-base` (768-dim,
   ~360 MB) instead of xls-r-300m (1024-dim, 1.2 GB), few epochs, small
   rationale sample. `configs/default.yaml` stays as the real-run config.
5. **Split handling** — trained on ASVspoof5 train partition, validated on a
   held-out portion, dev partition used as the in-domain test set. Real
   ASVspoof5 protocol filenames/columns verified against the downloaded
   protocol files before wiring (README's assumed format may not match).

## Pipeline stages (unchanged from README)

manifest → preprocess (resample 16 kHz, VAD trim, 4 s chunks) → wav2vec2
embeddings (mean-pooled, cached .npy) → train classifier head (ASVspoof5
only) → eval harness (in-domain EER vs out-of-domain EER = generalization
gap) → rationale agent + LLM judge on flagged clips.

## Verification

1. **Smoke run:** all six stages on the synthetic dataset; expect near-zero
   EER in-domain (classes are separable by construction) and a written
   `reports/eval_report.json` including real rationale + judge outputs.
2. **Real run:** same commands pointed at ASVspoof5 subset + In-the-Wild via
   `configs/default.yaml`; deliverable is the eval report showing in-domain
   EER, out-of-domain EER, the gap, and judged rationales.

## Out of scope

- P2V (not released), fine-tuning the wav2vec2 front-end, MHFA on frame-level
  features (LinearHead on pooled embeddings is the baseline; MHFA stays in
  `detector.py` for later), any training-set augmentation.
