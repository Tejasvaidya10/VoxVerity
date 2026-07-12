"""
Standard ASVspoof-style metrics. EER (equal error rate) is the field's
primary metric — the point where false acceptance rate equals false
rejection rate. Lower is better.
"""

import numpy as np


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple:
    """
    bonafide_scores: model scores for genuine samples (higher = more likely spoof,
                      matching the convention used in detector.py's logits)
    spoof_scores: model scores for spoof samples
    Returns (eer, threshold) where eer is a fraction in [0, 1].
    """
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.unique(all_scores)

    fars, frrs = [], []
    for t in thresholds:
        # false acceptance: bonafide scored above threshold (misclassified as spoof)
        far = np.mean(bonafide_scores >= t)
        # false rejection: spoof scored below threshold (misclassified as bonafide)
        frr = np.mean(spoof_scores < t)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)
    diffs = np.abs(fars - frrs)
    idx = np.argmin(diffs)
    eer = (fars[idx] + frrs[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_accuracy_at_threshold(bonafide_scores: np.ndarray, spoof_scores: np.ndarray,
                                    threshold: float) -> float:
    correct = np.sum(bonafide_scores < threshold) + np.sum(spoof_scores >= threshold)
    total = len(bonafide_scores) + len(spoof_scores)
    return correct / total


def summarize_scores(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> dict:
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    acc = compute_accuracy_at_threshold(bonafide_scores, spoof_scores, threshold)
    return {
        "eer": eer,
        "eer_pct": eer * 100,
        "threshold": threshold,
        "accuracy_at_eer_threshold": acc,
        "n_bonafide": len(bonafide_scores),
        "n_spoof": len(spoof_scores),
    }
