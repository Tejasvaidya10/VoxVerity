"""
Phase 4: end-to-end partial fine-tuning of a wav2vec2 backbone with MHFA
attention pooling. The cached-embedding shortcut used in phases 1-3 does not
survive fine-tuning; this trainer reads audio and backprops through the top
transformer layers. See docs/superpowers/specs/2026-07-18-finetune-design.md.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
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


def load_fixed_chunk(path: str, samples: int = 64000) -> torch.Tensor:
    """Read a wav, mono-mix, pad/trim to a fixed length, normalize."""
    wave, _ = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if len(wave) < samples:
        wave = np.pad(wave, (0, samples - len(wave)))
    return torch.from_numpy(normalize_wave(wave[:samples]))


class AudioChunkDataset(Dataset):
    def __init__(self, manifest_path: Path, samples: int = 64000):
        with open(manifest_path) as f:
            self.rows = list(csv.DictReader(f))
        self.samples = samples

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = 1.0 if row["label"] == "spoof" else 0.0
        return load_fixed_chunk(row["path"], self.samples), torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def score_manifest_audio(model: FinetuneModel, rows: list, device: str,
                         batch_size: int = 16) -> dict:
    model.eval()
    scores = {}
    batch_paths, batch_waves = [], []

    def flush():
        if not batch_waves:
            return
        waves = torch.stack(batch_waves).to(device)
        logits = model(waves).float().cpu().numpy()
        for p, lg in zip(batch_paths, logits):
            scores[p] = float(1 / (1 + np.exp(-lg)))
        batch_paths.clear()
        batch_waves.clear()

    for row in rows:
        try:
            wave = load_fixed_chunk(row["path"])
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
            continue
        batch_paths.append(row["path"])
        batch_waves.append(wave)
        if len(batch_waves) >= batch_size:
            flush()
    flush()
    return scores


def train_finetune(config: dict, train_manifest: Path, val_manifest: Path,
                   checkpoint_out: Path, device: str) -> None:
    ft = config["finetune"]
    model = FinetuneModel(ft["model_name"], config["model"]).to(device)
    freeze_bottom(model, ft["n_frozen_layers"])

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": ft["learning_rate"]},
        {"params": backbone_params, "lr": ft["learning_rate"] * ft["ssl_learning_rate_scale"]},
    ])
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(AudioChunkDataset(train_manifest), batch_size=ft["batch_size"],
                              shuffle=True, num_workers=ft["num_workers"])
    val_loader = DataLoader(AudioChunkDataset(val_manifest), batch_size=ft["batch_size"],
                            num_workers=ft["num_workers"])

    def validate() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for wave, label in val_loader:
                wave, label = wave.to(device), label.to(device)
                total += criterion(model(wave), label).item() * wave.size(0)
                n += wave.size(0)
        model.train()
        return total / max(n, 1)

    best_val = float("inf")
    step = 0
    t0, chunks = time.time(), 0
    model.train()
    for epoch in range(ft["max_epochs"]):
        for wave, label in train_loader:
            wave, label = wave.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(wave), label)
            loss.backward()
            optimizer.step()
            step += 1
            chunks += wave.size(0)
            if step % 200 == 0:
                rate = chunks / (time.time() - t0)
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"({rate:.1f} chunks/s)", flush=True)
            if step % ft["val_every_steps"] == 0:
                val_loss = validate()
                print(f"VAL step {step}: {val_loss:.4f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    save_finetuned(model, ft["model_name"], checkpoint_out)
                    print(f"saved checkpoint (val {val_loss:.4f})", flush=True)
        val_loss = validate()
        print(f"EPOCH {epoch} END val {val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            save_finetuned(model, ft["model_name"], checkpoint_out)
            print(f"saved checkpoint (val {val_loss:.4f})", flush=True)
    print(f"DONE best_val {best_val:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/detector_ft.pt"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_finetune(cfg, args.train_manifest, args.val_manifest,
                   args.checkpoint_out, args.device)
