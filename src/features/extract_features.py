"""
Extract wav2vec2 SSL embeddings for each preprocessed audio chunk and cache
them as .npy files. Embeddings are pooled (mean over time by default) so the
downstream classifier head takes a fixed-size vector per chunk.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def load_model(model_name: str, device: str):
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    return extractor, model


@torch.no_grad()
def extract_embedding(path: str, extractor, model, device: str, layer: int, pooling: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer].squeeze(0)  # (time, dim)

    if pooling == "mean":
        pooled = hidden.mean(dim=0)
    elif pooling == "attention":
        # simple unweighted fallback here; swap in a learned attention pool
        # in src/models/detector.py if you want a trainable pooling layer instead
        pooled = hidden.mean(dim=0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return pooled.cpu().numpy()


def process_manifest(manifest_path: Path, out_dir: Path, model_name: str,
                      layer: int, pooling: str, device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor, model = load_model(model_name, device)

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        stem = Path(row["path"]).stem
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists():
            continue
        try:
            emb = extract_embedding(row["path"], extractor, model, device, layer, pooling)
            np.save(out_path, emb)
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
        if (i + 1) % 200 == 0:
            print(f"Extracted {i + 1}/{len(rows)}")

    print(f"Done. Embeddings cached in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    process_manifest(args.manifest, args.out_dir, args.model_name, args.layer, args.pooling, args.device)
