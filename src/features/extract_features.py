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

from src.provenance import check_resume, read_sidecar, sha256_file, write_sidecar

FEATURES_COMPARE_KEYS = ["preprocessing", "input_manifest_sha256", "model_name", "layer", "pooling"]


def features_meta(manifest_path: Path, model_name: str, layer: int, pooling: str) -> dict:
    """Provenance for a features folder: the preprocessing settings recorded
    next to the chunk manifest, plus the extraction settings."""
    side = read_sidecar(Path(str(manifest_path) + ".meta.json"))
    return {"stage": "features",
            "preprocessing": side["settings"] if side else "unrecorded (pre-Phase 5)",
            "input_manifest_sha256": sha256_file(manifest_path),
            "model_name": model_name, "layer": layer, "pooling": pooling}


def prepare_features_dir(out_dir: Path, meta: dict, allow_legacy: bool) -> None:
    """Refuse to resume into a folder made from different settings/inputs
    (the extractor skips existing .npy by name, so mixing would be silent),
    then record this run's provenance."""
    out_dir = Path(out_dir)
    has_outputs = out_dir.exists() and any(out_dir.glob("*.npy"))
    check_resume(out_dir / "_meta.json", meta, FEATURES_COMPARE_KEYS,
                 has_outputs=has_outputs, allow_legacy=allow_legacy)
    write_sidecar(out_dir / "_meta.json", meta)


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


@torch.no_grad()
def extract_batch(audios: list, extractor, model, device: str, layer: int,
                  pooling: str) -> np.ndarray:
    """Batched extraction for equal-length chunks (padding handles stragglers)."""
    inputs = extractor(audios, sampling_rate=16000, return_tensors="pt",
                       padding=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # (batch, time, dim)
    if pooling in ("mean", "attention"):
        pooled = hidden.mean(dim=1)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    return pooled.cpu().numpy()


def process_manifest(manifest_path: Path, out_dir: Path, model_name: str,
                      layer: int, pooling: str, device: str,
                      batch_size: int = 16, allow_legacy_dir: bool = False) -> None:
    prepare_features_dir(out_dir, features_meta(manifest_path, model_name, layer, pooling),
                         allow_legacy=allow_legacy_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor, model = load_model(model_name, device)

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    pending = []  # (out_path, audio) accumulated until a full batch
    done = 0

    def flush():
        nonlocal done
        if not pending:
            return
        embs = extract_batch([a for _, a in pending], extractor, model,
                             device, layer, pooling)
        for (out_path, _), emb in zip(pending, embs):
            np.save(out_path, emb)
        done += len(pending)
        pending.clear()

    for i, row in enumerate(rows):
        stem = Path(row["path"]).stem
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists():
            continue
        try:
            audio, _ = sf.read(row["path"], dtype="float32")
            pending.append((out_path, audio))
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
        if len(pending) >= batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"Progress {i + 1}/{len(rows)} ({done} newly extracted)", flush=True)
    flush()

    print(f"Done. Embeddings cached in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-legacy-dir", action="store_true")
    args = parser.parse_args()

    process_manifest(args.manifest, args.out_dir, args.model_name, args.layer,
                     args.pooling, args.device, batch_size=args.batch_size,
                     allow_legacy_dir=args.allow_legacy_dir)
