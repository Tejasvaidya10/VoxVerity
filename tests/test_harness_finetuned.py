import csv
import numpy as np
import soundfile as sf
from src.eval.harness import score_manifest_ft
from src.models.finetune import FinetuneModel

SR = 16000
MHFA = {"embedding_dim": 768, "num_heads": 32, "compression_dim": 128, "dropout": 0.1}


def test_score_manifest_ft_shape(tmp_path):
    d = tmp_path / "wav"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(6):
        p = d / f"c{i}.wav"
        sf.write(p, (0.2 * rng.standard_normal(SR)).astype(np.float32), SR, subtype="PCM_16")
        rows.append({"path": str(p), "label": "spoof" if i % 2 else "bonafide",
                     "speaker_id": "s", "dataset_source": "x", "attack_type": "u"})
    m = tmp_path / "m.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    model = FinetuneModel("facebook/wav2vec2-base", MHFA).eval()
    out = score_manifest_ft(model, m, device="cpu")
    assert set(out) == {"metrics", "per_file_scores"}
    assert len(out["per_file_scores"]) == 6
    assert "eer_pct" in out["metrics"]
