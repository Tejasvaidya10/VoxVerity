"""
Train the classifier head on cached wav2vec2 embeddings from ASVspoof5.
Do not point --manifest at P2V or In-the-Wild — those are eval-only, kept
out of training on purpose to measure real generalization.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from src.models.detector import build_head


class EmbeddingDataset(Dataset):
    def __init__(self, manifest_path: Path, features_dir: Path):
        with open(manifest_path) as f:
            self.rows = list(csv.DictReader(f))
        self.features_dir = features_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        stem = Path(row["path"]).stem
        emb_path = self.features_dir / f"{stem}.npy"
        emb = np.load(emb_path).astype(np.float32)
        label = 1.0 if row["label"] == "spoof" else 0.0
        return torch.from_numpy(emb), torch.tensor(label, dtype=torch.float32)


def train(config_path: str, train_manifest: Path, val_manifest: Path,
          features_dir: Path, checkpoint_out: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_head(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_ds = EmbeddingDataset(train_manifest, features_dir)
    val_ds = EmbeddingDataset(val_manifest, features_dir)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss = 0.0
        for emb, label in train_loader:
            emb, label = emb.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(emb)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * emb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb, label in val_loader:
                emb, label = emb.to(device), label.to(device)
                logits = model(emb)
                loss = criterion(logits, label)
                val_loss += loss.item() * emb.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_out)
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    print(f"Best val loss: {best_val_loss:.4f}. Checkpoint saved to {checkpoint_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/detector.pt"))
    args = parser.parse_args()

    train(args.config, args.train_manifest, args.val_manifest, args.features_dir, args.checkpoint_out)
