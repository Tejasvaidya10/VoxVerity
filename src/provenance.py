"""
Provenance sidecars: every pipeline stage records the resolved settings and
the checksum of its input so any result can say exactly what produced it,
and resumable stages refuse to mix outputs from different configurations.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class ProvenanceMismatch(RuntimeError):
    """An output folder was produced with different settings or inputs."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def write_sidecar(path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {**meta, "created_at": datetime.now(timezone.utc).isoformat(),
               "git_commit": git_commit()}
    path.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")


def read_sidecar(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_resume(dir_meta_path: Path, current: dict, compare_keys: list,
                 has_outputs: bool, allow_legacy: bool) -> None:
    existing = read_sidecar(dir_meta_path)
    if existing is None:
        if has_outputs and not allow_legacy:
            raise ProvenanceMismatch(
                f"{Path(dir_meta_path).parent} already has outputs but no provenance sidecar "
                "(legacy folder). Use a fresh folder, or pass --allow-legacy-dir.")
        return
    differing = [k for k in compare_keys if existing.get(k) != current.get(k)]
    if differing:
        raise ProvenanceMismatch(
            f"{Path(dir_meta_path).parent} was produced with different {', '.join(differing)}: "
            f"recorded {[existing.get(k) for k in differing]}, now {[current.get(k) for k in differing]}. "
            "Use a separate folder for each configuration.")
