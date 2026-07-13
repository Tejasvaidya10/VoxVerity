import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "split_manifest", Path("scripts/split_manifest.py"))
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def _rows():
    return [{"path": f"/a/{s}_{i}.wav", "label": "spoof" if i % 2 else "bonafide",
             "speaker_id": s, "dataset_source": "x", "attack_type": "u"}
            for s in ["sp1", "sp2", "sp3", "sp4", "sp5"] for i in range(4)]


def test_split_is_by_speaker_with_no_overlap():
    train, val = sm.split_manifest(_rows(), val_fraction=0.2, seed=7)
    train_sp = {r["speaker_id"] for r in train}
    val_sp = {r["speaker_id"] for r in val}
    assert train_sp and val_sp
    assert not (train_sp & val_sp)
    assert len(train) + len(val) == 20


def test_split_is_deterministic():
    a = sm.split_manifest(_rows(), val_fraction=0.2, seed=7)
    b = sm.split_manifest(_rows(), val_fraction=0.2, seed=7)
    assert a == b
