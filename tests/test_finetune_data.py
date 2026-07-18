import csv
import numpy as np
import soundfile as sf
import torch
from src.models.finetune import AudioChunkDataset, FinetuneModel, score_manifest_audio

SR = 16000
MHFA = {"embedding_dim": 768, "num_heads": 32, "compression_dim": 128, "dropout": 0.1}


def _manifest(tmp_path, n=4, samples=SR):
    d = tmp_path / "wav"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        p = d / f"c{i}.wav"
        sf.write(p, (0.2 * rng.standard_normal(samples)).astype(np.float32), SR, subtype="PCM_16")
        rows.append({"path": str(p), "label": "spoof" if i % 2 else "bonafide",
                     "speaker_id": "s", "dataset_source": "x", "attack_type": "u",
                     "chunk_idx": "0", "augmentation": "none"})
    m = tmp_path / "m.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return m, rows


def test_dataset_fixed_length_and_labels(tmp_path):
    m, _ = _manifest(tmp_path, samples=SR // 2)  # short clip -> padded
    ds = AudioChunkDataset(m, samples=SR)
    wave, label = ds[0]
    assert wave.shape == (SR,)
    assert wave.dtype == torch.float32
    assert float(label) in (0.0, 1.0)
    assert abs(float(wave.mean())) < 1e-3  # normalized


def test_score_manifest_audio_returns_probs(tmp_path):
    m, rows = _manifest(tmp_path)
    model = FinetuneModel("facebook/wav2vec2-base", MHFA).eval()
    scores = score_manifest_audio(model, rows, device="cpu", batch_size=2)
    assert set(scores) == {r["path"] for r in rows}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
