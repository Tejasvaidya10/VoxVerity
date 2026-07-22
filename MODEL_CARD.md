# Model Card: VoxVerity Audio Deepfake Detector

Format follows Mitchell et al., *Model Cards for Model Reporting* (2019).
See [REPORT.md](REPORT.md) for the full study and [README.md](README.md)
for reproduction instructions.

## Model details

- **Developed by:** VoxVerity project (independent research study).
- **Model type:** Binary audio classifier (bonafide speech vs. synthetic/
  spoofed speech), operating on 4-second audio chunks.
- **Architecture:** Frozen `facebook/wav2vec2-xls-r-300m` self-supervised
  embeddings (mean-pooled over time) → 2-layer MLP classification head
  (`LinearHead` in `src/models/detector.py`). Only the head is trained;
  the SSL front-end is never updated.
- **Recommended artifact:** `checkpoints/detector_aug.pt` — trained with
  channel augmentation (Phase 3). This is the best-performing model
  produced by the study and the one recommended for any further use.
- **Other artifacts in this repo** (kept for reproducibility, not
  recommended for use):
  - `checkpoints/detector.pt` — Phase 1, no augmentation. Superseded by
    `detector_aug.pt` on every metric.
  - `checkpoints/detector_ft.pt` — Phase 4, fine-tuned `wav2vec2-base` +
    MHFA attention pooling. **Known to underperform substantially**
    (In-the-Wild EER 24.40% vs. 11.30% for the recommended model) due to
    overfitting to training-set synthesis attacks. Retained only as a
    negative-result artifact for reproducibility; see REPORT.md §6.
- **License:** MIT (code). Model weights inherit the license of their
  training data — see Training Data below.
- **Contact / repository:** https://github.com/Tejasvaidya10/VoxVerity

## Intended use

- **Intended use:** Research and benchmarking of audio deepfake detection
  generalization; a baseline for further experimentation (augmentation
  strategies, front-end fine-tuning, pooling mechanisms); educational
  demonstration of the benchmark-to-real-world generalization gap in
  anti-spoofing systems.
- **Out-of-scope uses:** This model is **not validated for production
  deployment, forensic determinations, legal evidence, or any
  high-stakes decision about the authenticity of a specific audio clip.**
  Its real-world equal error rate (11.30%, best artifact) means roughly
  1 in 9 decisions at its operating threshold is wrong, and its decision
  threshold is not calibrated across recording conditions (see
  Quantitative Analyses below) — a raw score should not be read as a
  calibrated probability outside the ASVspoof5 domain.

## Training data

- **ASVspoof5** training partition (attacks A01–A08), Zenodo record
  14498691, Open Data Commons Attribution License (ODC-By). ~175,000
  training chunks after speaker-disjoint splitting, VAD trimming, and
  4-second/2-second-stride chunking.
- Phase 3 / recommended artifact additionally applies channel
  augmentation (codec re-encoding, additive noise, band-limiting) to 50%
  of training chunks, independent of label — see REPORT.md §5 for the
  transform specification.
- The model is never trained or validated on In-the-Wild or any
  evaluation-only dataset.

## Evaluation data

- **ASVspoof5 development partition** (attacks A09–A16 — disjoint from
  training attacks): in-domain, unseen-attack evaluation.
- **In-the-Wild** (Müller et al.; independent real-world deepfake audio
  of public figures): out-of-domain evaluation, used exclusively for
  testing, never training.

## Metrics

Equal error rate (EER) — the operating point where false-acceptance rate
equals false-rejection rate — is the primary metric, standard in the
anti-spoofing literature (lower is better). Also reported: per-direction
error rates (false-alarm vs. miss) and per-speaker error breakdowns.

## Quantitative analyses (recommended artifact: `detector_aug.pt`)

| Eval condition | EER |
|---|---|
| Held-out split of training distribution | 0.16% |
| ASVspoof5 dev (unseen attacks) | 1.18% |
| **In-the-Wild (real-world, out-of-domain)** | **11.30%** |

At the In-the-Wild operating threshold, false-alarm and false-negative
rates are both ~11.3%. The decision threshold itself shifts substantially
between in-domain and out-of-domain evaluation (approximately 0.81
in-domain vs. 0.017 on In-the-Wild) — **raw scores are not
cross-domain-calibrated**; only the ranking (hence EER) should be trusted
across domains, not the absolute score.

## Ethical considerations

- **Dual use.** A model for detecting synthetic speech is inherently
  dual-use: understanding its failure modes (this study's explicit focus)
  also documents how a synthetic-speech generator might evade detection
  by adjusting recording/channel characteristics. This is disclosed
  intentionally, in keeping with the study's transparency goals, and
  because the underlying dynamic (clean-vs-degraded channel confounding
  authenticity judgments) is already documented in the anti-spoofing
  literature this project builds on.
- **Real-world stakes of false results.** A false negative (fake scored as
  real) in a deployed detector could enable fraud, impersonation, or
  disinformation to go unflagged. A false positive (real audio scored as
  fake) could wrongly discredit genuine speech or evidence. Given the
  measured 11.3% error rate in realistic conditions, this model must not
  be the sole basis for any consequential determination.
- **Training data provenance.** In-the-Wild evaluation data includes
  audio of named public figures (used here strictly for evaluation, per
  the dataset's original release terms); this model card does not confer
  any rights to redistribute that audio.

## Caveats and recommendations

- Use `detector_aug.pt`, not the other two checkpoints in this repo.
- Do not interpret the raw sigmoid score as a calibrated probability
  outside the ASVspoof5 domain; recalibrate for any new deployment
  context.
- The 11.30% In-the-Wild EER is the best result this architecture family
  has achieved at this scale in the current study, not a guarantee of
  further improvement — see REPORT.md §9 for concrete next steps
  (fine-tuning the large frozen backbone under regularization, stronger
  augmentation, cross-domain calibration) before any deployment
  consideration.
- This model has not been evaluated on P2V (not publicly available at
  time of study) or any dataset beyond ASVspoof5 and In-the-Wild; its
  generalization to other real-world distributions (different languages,
  microphones, or synthesis methods released after this study) is
  unknown.
