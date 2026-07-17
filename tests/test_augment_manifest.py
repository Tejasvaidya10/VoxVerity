import csv
import numpy as np
import soundfile as sf
from pathlib import Path
from src.preprocessing.augment import augment_manifest

SR = 16000


def _make_manifest(tmp_path, n=40):
    audio_dir = tmp_path / "chunks"
    audio_dir.mkdir()
    rows = []
    rng = np.random.default_rng(3)
    for i in range(n):
        p = audio_dir / f"clip_{i}.wav"
        sf.write(p, (0.3 * rng.standard_normal(SR)).astype(np.float32), SR, subtype="PCM_16")
        rows.append({"path": str(p), "label": "spoof" if i % 2 else "bonafide",
                     "speaker_id": f"sp{i % 4}", "dataset_source": "x",
                     "attack_type": "u", "chunk_idx": "0"})
    mpath = tmp_path / "chunks.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return mpath


def _run(tmp_path, seed=42):
    mpath = tmp_path / "chunks.csv"
    if not mpath.exists():
        mpath = _make_manifest(tmp_path)
    out_audio = tmp_path / f"aug_{seed}"
    out_manifest = tmp_path / f"aug_{seed}.csv"
    augment_manifest(mpath, out_audio, out_manifest, seed=seed)
    return list(csv.DictReader(open(out_manifest)))


def test_rows_preserved_and_annotated(tmp_path):
    rows = _run(tmp_path)
    assert len(rows) == 40
    assert all(r["augmentation"] in ("none", "codec", "noise", "bandlimit") for r in rows)


def test_clean_rows_keep_original_path_augmented_get_new_files(tmp_path):
    rows = _run(tmp_path)
    for r in rows:
        if r["augmentation"] == "none":
            assert "__aug_" not in r["path"]
        else:
            assert f"__aug_{r['augmentation']}" in r["path"]
            assert Path(r["path"]).exists()
            audio, sr = sf.read(r["path"])
            assert sr == SR and len(audio) == SR


def test_roughly_half_augmented_and_label_independent(tmp_path):
    rows = _run(tmp_path)
    aug = [r for r in rows if r["augmentation"] != "none"]
    assert 10 <= len(aug) <= 30  # ~50% of 40, loose bound
    by_label = {"spoof": 0, "bonafide": 0}
    for r in aug:
        by_label[r["label"]] += 1
    assert by_label["spoof"] > 0 and by_label["bonafide"] > 0


def test_deterministic_across_runs(tmp_path):
    a = _run(tmp_path, seed=7)
    out_manifest2 = tmp_path / "aug_7_again.csv"
    augment_manifest(tmp_path / "chunks.csv", tmp_path / "aug_7_again", out_manifest2, seed=7)
    b = list(csv.DictReader(open(out_manifest2)))
    assert [r["augmentation"] for r in a] == [r["augmentation"] for r in b]
