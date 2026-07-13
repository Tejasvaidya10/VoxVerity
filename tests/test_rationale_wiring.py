import numpy as np
import soundfile as sf
from dataclasses import asdict
from src.explain.rationale_agent import DetectionEvidence
from src.eval.harness import build_evidence


def test_detection_evidence_has_extended_fields():
    fields = set(DetectionEvidence.__dataclass_fields__)
    assert {"spectral_rolloff_95_hz", "unvoiced_energy_ratio",
            "high_band_energy_fraction"} <= fields


def test_build_evidence_fills_real_values(tmp_path):
    rng = np.random.default_rng(1)
    wav = tmp_path / "clip.wav"
    sf.write(wav, rng.standard_normal(16000).astype(np.float32) * 0.3, 16000)
    ev = build_evidence(str(wav), spoof_probability=0.9)
    d = asdict(ev)
    assert d["file_id"] == "clip"
    assert d["spoof_probability"] == 0.9
    assert d["spectral_flatness"] > 0  # not the old -1.0 placeholder
    assert d["pitch_variance"] >= 0
