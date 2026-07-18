import numpy as np
import torch
from src.models.finetune import (FinetuneModel, freeze_bottom, normalize_wave,
                                 save_finetuned, load_finetuned)

MHFA = {"embedding_dim": 768, "num_heads": 32, "compression_dim": 128, "dropout": 0.1}
MODEL = "facebook/wav2vec2-base"


def _model():
    return FinetuneModel(MODEL, MHFA)


def test_forward_returns_batch_logits():
    m = _model().eval()
    wave = torch.randn(2, 16000)
    with torch.no_grad():
        out = m(wave)
    assert out.shape == (2,)


def test_freeze_bottom_leaves_top_layers_and_head_trainable():
    m = _model()
    freeze_bottom(m, n_frozen_layers=6)
    assert not any(p.requires_grad for p in m.backbone.feature_extractor.parameters())
    assert not any(p.requires_grad for p in m.backbone.encoder.layers[0].parameters())
    assert not any(p.requires_grad for p in m.backbone.encoder.layers[5].parameters())
    assert all(p.requires_grad for p in m.backbone.encoder.layers[6].parameters())
    assert all(p.requires_grad for p in m.backbone.encoder.layers[11].parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_normalize_wave_zero_mean_unit_var():
    w = np.random.default_rng(0).uniform(-0.5, 0.5, 16000).astype(np.float32) + 0.1
    n = normalize_wave(w)
    assert abs(float(n.mean())) < 1e-4
    assert abs(float(n.std()) - 1.0) < 1e-2


def test_save_load_roundtrip(tmp_path):
    m = _model().eval()
    p = tmp_path / "ft.pt"
    save_finetuned(m, MODEL, p)
    m2 = load_finetuned(p, MHFA, "cpu").eval()
    wave = torch.randn(1, 16000)
    with torch.no_grad():
        a, b = m(wave), m2(wave)
    assert torch.allclose(a, b, atol=1e-5)
