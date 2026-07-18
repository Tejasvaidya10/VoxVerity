"""
Phase 4: end-to-end partial fine-tuning of a wav2vec2 backbone with MHFA
attention pooling. The cached-embedding shortcut used in phases 1-3 does not
survive fine-tuning; this trainer reads audio and backprops through the top
transformer layers. See docs/superpowers/specs/2026-07-18-finetune-design.md.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from src.models.detector import MHFAHead


class FinetuneModel(nn.Module):
    def __init__(self, model_name: str, mhfa: dict):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.backbone.config.apply_spec_augment = False  # no train-time masking surprises
        self.head = MHFAHead(
            embedding_dim=mhfa["embedding_dim"],
            num_heads=mhfa.get("num_heads", mhfa.get("num_attention_heads")),
            compression_dim=mhfa["compression_dim"],
            dropout=mhfa["dropout"],
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(wave).last_hidden_state  # (batch, time, dim)
        return self.head(hidden)


def freeze_bottom(model: FinetuneModel, n_frozen_layers: int) -> None:
    frozen_modules = [model.backbone.feature_extractor,
                      model.backbone.feature_projection,
                      model.backbone.encoder.pos_conv_embed,
                      model.backbone.encoder.layer_norm]
    frozen_modules += list(model.backbone.encoder.layers[:n_frozen_layers])
    for mod in frozen_modules:
        for p in mod.parameters():
            p.requires_grad = False
    if model.backbone.config.mask_time_prob > 0:  # unused with apply_spec_augment=False
        model.backbone.masked_spec_embed.requires_grad = False


def normalize_wave(wave: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-variance, matching Wav2Vec2FeatureExtractor(do_normalize=True)."""
    wave = wave.astype(np.float32)
    return (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)


def save_finetuned(model: FinetuneModel, model_name: str, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model_name, "state_dict": model.state_dict()}, path)


def load_finetuned(path: Path, mhfa: dict, device: str) -> FinetuneModel:
    ckpt = torch.load(path, map_location=device)
    model = FinetuneModel(ckpt["backbone"], mhfa)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
