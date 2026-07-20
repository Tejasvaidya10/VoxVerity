# VoxVerity: Measuring, Explaining, and Attempting to Close the Audio Deepfake Generalization Gap

**A research study in four phases.** Code and reproduction instructions: [README.md](README.md). Model artifact details: [MODEL_CARD.md](MODEL_CARD.md).

## Abstract

Academic audio deepfake detectors routinely report near-perfect accuracy on
benchmark datasets (ASVspoof) and are known, anecdotally, to degrade sharply
on real-world audio. This study treats that degradation as the primary
research question rather than an afterthought. We train a spoofed-speech
detector on ASVspoof5, measure its collapse on an independent real-world
dataset (In-the-Wild), and use an LLM-based explainability layer — detector
evidence in, a faithfulness-judged natural-language rationale out — to make
the failure mode legible rather than just numeric. Per-speaker failure
analysis reveals the detector substantially uses **recording channel
quality as a proxy for authenticity**. We pre-register a hypothesis test —
decorrelating channel quality from labels via training-time augmentation —
and find it produces a real but partial improvement (In-the-Wild EER 13.34%
→ 11.30%; hypothesis confirmed in mechanism, success threshold not met). We
then attempt to close the remaining gap by fine-tuning the SSL front-end
end-to-end; this produces a **decisive negative result** (In-the-Wild EER
regresses to 24.40%), which we show is itself informative: it indicates the
frozen, multilingually-pretrained front-end was a primary source of
robustness that a smaller, narrowly-pretrained, fine-tuned front-end
destroys. The best-performing artifact from this study is the
augmentation-trained detector (`checkpoints/detector_aug.pt`).

## 1. Introduction

Spoofed-speech detectors trained and evaluated on ASVspoof-family benchmarks
report equal error rates (EER) in the low single digits or lower. Deployed
against real-world audio — different microphones, codecs, background noise,
and speakers than the benchmark ever contained — performance is known to
degrade, but the *mechanism* of that degradation is rarely characterized in
detail: is it that real-world audio contains synthesis methods the detector
has never seen, or something else entirely? This study is designed around
that question specifically, using explainability tooling (an LLM rationale
generator and judge) as a first-class part of the evaluation harness rather
than a post-hoc add-on.

## 2. Methodology

### 2.1 Data

- **ASVspoof5** (train + development partitions): the training/in-domain
  benchmark. Openly downloadable from Zenodo (record 14498691, ODC-By
  license). Train partition contains synthesis attacks A01–A08; the
  development partition contains a fully disjoint attack set, A09–A16 —
  making the dev partition a genuine unseen-attack test, not merely
  unseen-speaker.
- **In-the-Wild**: an independent, real-world audio deepfake dataset
  (speeches, interviews, archival recordings of public figures), used
  exclusively for evaluation. Never seen during training at any phase.
- A third dataset, P2V (Perturbed Public Voices), was scoped into the
  original design but is not yet publicly released as of this study; the
  evaluation harness accepts an arbitrary number of held-out manifests, so
  it can be added without code changes if it ships.

### 2.2 Detector architecture (Phases 1–3)

Frozen `wav2vec2-xls-r-300m` self-supervised embeddings (mean-pooled over
time, 4-second windows with 2-second stride), followed by a small trainable
MLP classification head. The front-end is never updated; only the head is
trained. This is a deliberately conservative baseline architecture — its
purpose is to isolate the generalization question from architecture-search
questions.

### 2.3 Evaluation protocol

Equal error rate (EER) — the operating point where false-acceptance and
false-rejection rates are equal — is the primary metric, standard in the
anti-spoofing literature. Training and validation splits are speaker-disjoint
(no speaker appears in both). Three eval conditions are reported
consistently across phases: a held-out split of the training distribution
(same attacks, unseen speakers), the ASVspoof5 development partition
(unseen attacks, same corpus/recording conditions), and In-the-Wild (unseen
everything). The gap between the development-partition EER and the
In-the-Wild EER is the study's central dependent variable.

### 2.4 Explainability layer

For a sample of flagged clips (stratified across evaluation sets so
out-of-domain failures are represented, not just the most confident
in-domain flags), five acoustic descriptors are computed directly from the
audio: pitch variance, spectral flatness, 95th-percentile spectral
roll-off, unvoiced-segment energy ratio, and high-frequency (>4kHz) energy
fraction. These, plus the detector's score, are given to an LLM
(`claude-sonnet-4-6`) which writes a 2–3 sentence rationale grounded
strictly in the provided evidence. A second LLM call judges each rationale
for faithfulness (1–5) against the same evidence. This turns individual
detector decisions into inspectable, gradeable natural-language artifacts
and was used throughout the study, not only for reporting.

## 3. Phase 1 — Measuring the generalization gap

| Eval condition | EER |
|---|---|
| Held-out split, same attacks (A01–A08) | 0.04% |
| ASVspoof5 dev, unseen attacks (A09–A16) | 1.40% |
| In-the-Wild, unseen everything | 13.34% |

**Finding:** the gap decomposes into two very different steps. Generalizing
to unseen *synthesis methods* costs almost nothing (0.04% → 1.40%);
generalizing to unseen *recording conditions* costs an order of magnitude
more (1.40% → 13.34%). This rules out the naive explanation ("the model
just memorized its training generators") and points at acoustic domain
shift specifically.

Rationale-judge mean faithfulness on this run: 4.54/5 (n=50).

## 4. Phase 2 — Failure analysis

Examining the In-the-Wild errors by direction and by speaker (rather than
only the aggregate EER) surfaces a mechanism:

- **False negatives** (fakes scored as real) cluster on speakers whose
  source audio is degraded or archival — e.g. per-speaker miss rates up to
  60–64% for speakers with old/noisy source recordings.
- **False positives** (genuine audio scored as fake) cluster on speakers
  with clean, studio-processed audio — up to ~60% false-alarm rates.
- The operating threshold itself is not stable across domains: it sits at
  ~0.66 in-domain and collapses to ~0.0029 on In-the-Wild — the score
  distribution shifts wholesale toward "real" out of domain, indicating the
  detector's confidence is not calibrated across acoustic conditions.

**Interpretation:** the detector has learned to associate *low recording
quality with authenticity*, because in ASVspoof5 that association happens
to hold (bonafide = natural imperfect recordings, spoof = clean TTS
output). In the wild this association is not just absent but frequently
inverted.

## 5. Phase 3 — Channel-augmentation experiment (pre-registered)

**Hypothesis:** if channel quality is acting as a spurious authenticity
signal, decorrelating it from the label at training time should reduce the
In-the-Wild gap.

**Method:** 50% of training chunks, selected independently of label, are
passed through one of three channel-degrading transforms (lossy codec
round-trip at 16–96 kbps, additive noise at 5–25dB SNR, or band-limiting)
before feature extraction; the remaining 50% stay clean. Model architecture,
hyperparameters, and evaluation sets are otherwise identical to Phase 1.

**Pre-registered success criterion:** In-the-Wild EER < 10% with dev EER
< 3%.

| Eval condition | Phase 1 (no augmentation) | Phase 3 (augmented) |
|---|---|---|
| Held-out split | 0.04% | 0.16% |
| ASVspoof5 dev | 1.40% | 1.18% |
| In-the-Wild | 13.34% | **11.30%** |
| Gap (ITW − dev) | 11.94 pts | 10.12 pts |

**Outcome: hypothesis partially confirmed; success criterion not met.**
Every metric improved, at zero cost to in-domain performance — the shortcut
is real and the fix is real — but the pre-registered bar was missed by a
wide margin, meaning channel quality explains only a modest share (roughly
one-sixth) of the observed gap. Per-speaker failure analysis after
augmentation shows the mechanism directly: the clearest "archival fake
passes as real" case is fixed (that speaker drops out of the top misses
entirely), but archival *genuine* speakers newly enter the false-alarm list.
The model no longer equates degradation with authenticity in a fixed
direction — but heavily degraded audio of either class remains a genuinely
hard case, indicating residual domain mismatch beyond the specific shortcut
tested.

Rationale-judge mean faithfulness: 4.73/5 (n=50).

## 6. Phase 4 — Fine-tuning experiment (negative result)

Having answered the "why" of Phases 1–3, Phase 4 shifts from research to
engineering: attempt to close the remaining gap by fine-tuning the SSL
front-end end-to-end, rather than keeping it frozen.

**Method:** partial fine-tuning of `wav2vec2-base` (95M parameters; the
CNN feature extractor and bottom 6 of 12 transformer layers frozen, top 6
tuned), with a trainable multi-head factorized attention (MHFA) pooling
head replacing mean pooling, trained end-to-end on the Phase-3 augmented
data for 2 epochs. This deliberately confounds two changes at once
(a smaller backbone, now tuned, plus a different pooling mechanism) —
appropriate for an engineering attempt aimed at a better artifact, not a
controlled experiment isolating one variable.

| Eval condition | Phase 3 (frozen XLS-R + MLP) | Phase 4 (tuned wav2vec2-base + MHFA) |
|---|---|---|
| Held-out split (seen attacks) | 0.16% | 0.32% |
| ASVspoof5 dev (unseen attacks) | 1.18% | **6.19%** |
| In-the-Wild | 11.30% | **24.40%** |
| Gap (ITW − dev) | 10.12 pts | 18.21 pts |

**Outcome: success criterion failed decisively.** The fine-tuned model is
worse on every held-out condition than the frozen-backbone model, despite
being far more accurate on its own training distribution (0.32% seen vs.
6.19% unseen vs. 24.40% real-world — a textbook overfitting signature). Its
out-of-domain decision threshold collapses to ~1e-8, i.e. it scores nearly
all out-of-distribution audio as genuine. Per-speaker analysis shows its
worst false alarms are archival genuine recordings at rates of 67–78%,
substantially worse than the same failure mode in Phase 1.

**Interpretation.** We read this as evidence that the frozen
`wav2vec2-xls-r-300m` representation was itself a major source of the
robustness observed in Phases 1–3: it is pretrained on 128 languages and a
wide variety of recording channels, whereas `wav2vec2-base` is pretrained
on ~960 hours of clean English audiobook narration. Fine-tuning on a single
corpus's eight attack types pulled the smaller, less-robust representation
toward exactly the generator-specific fingerprints the whole study exists
to avoid overfitting to. The practical implication: at this scale of data
and compute, a large frozen multilingual SSL backbone outperforms a small
fine-tuned one for this task, and any future fine-tuning attempt should
tune the large backbone itself (which requires more compute than a laptop
provides) under stronger regularization toward its pretrained weights,
rather than substitute a smaller model.

Rationale-judge mean faithfulness held steady at 4.41/5 (n=50), indicating
the explainability layer's reliability is independent of detector quality.

## 7. Discussion

Three results triangulate the same conclusion from different angles.
Phase 1 showed the gap is driven by domain shift, not attack novelty.
Phase 2 named the specific shortcut (channel quality as authenticity
proxy). Phase 3 showed removing that shortcut helps but doesn't close the
gap. Phase 4 showed — by failing — that the *source* of whatever
robustness the detector does have is concentrated in the frozen front-end's
pretraining distribution, not in any component we can cheaply retrain.
Together these suggest the highest-leverage remaining intervention is not
more architecture engineering on top of a small model, but either (a)
fine-tuning the large multilingual backbone itself under careful
regularization, or (b) further augmentation strategies that more precisely
target the residual domain mismatch Phase 3 left unaddressed (e.g. stacked
or stronger transforms, or transforms sourced from real channel
distributions rather than synthetic degradation).

## 8. Limitations

- Training data is one of five official ASVspoof5 train tars (~20% of the
  full official training set); more data was not exhausted before moving
  to architectural experiments.
- Phase 4 confounds backbone size and pooling mechanism; it cannot on its
  own attribute the regression to either change independently.
- P2V, the third dataset originally scoped, was not available during this
  study.
- Explainability rationales are graded for internal faithfulness to
  provided evidence, not for correctness against ground-truth forensic
  analysis — faithfulness and correctness are related but not identical
  properties.
- All experiments ran on a single consumer laptop (Apple M4, 16GB); this
  bounded model scale and epoch count, most consequentially in Phase 4.

## 9. Future work

In priority order: (1) fine-tune the XLS-R backbone itself, on
GPU-adequate compute, with parameter-efficient methods (e.g. LoRA) and/or
attack-held-out early stopping so unseen-attack performance — not seen-attack
loss — governs when training stops; (2) MHFA attention pooling on the
*frozen* XLS-R representation as an isolated test of pooling alone; (3)
stronger or stacked channel augmentation; (4) cross-domain score
calibration, independent of ranking quality, since threshold collapse was
observed in every phase; (5) incorporate P2V if and when it is released.

## Appendix: reproducibility

Full commands, configs, and directory layout: [README.md](README.md).
Raw evaluation reports: `reports/eval_report.json` (Phase 1),
`reports/eval_report_aug.json` (Phase 3), `reports/eval_report_ft.json`
(Phase 4). Model checkpoints and their status: [MODEL_CARD.md](MODEL_CARD.md).
