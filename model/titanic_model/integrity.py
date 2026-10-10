"""Artifact integrity: sha256 goes into the manifest at training, is checked before unpickling.

A joblib file is a pickle, and ``joblib.load`` runs code while it reads. So the
bytes are checked against the manifest BEFORE loading: a swapped or corrupted
file is rejected before it gets a chance to execute.

The hash ships in the same release as the file, so it catches corruption, mixed
versions and a partial swap. Someone who re-uploads both assets is stopped by the
attestation (``gh attestation verify`` in docker.yml) and by immutable releases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB: never read the whole artifact into memory


class ArtifactIntegrityError(RuntimeError):
    """The artifact does not match its manifest and must not be loaded."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    """Manifest entry for the model file: name, size and sha256."""
    path = Path(path)
    return {"file": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def verify_artifact(path: Path, manifest: dict[str, Any]) -> None:
    """Raise ArtifactIntegrityError unless the file is the one the manifest describes.

    A manifest with no ``artifact`` entry is rejected too: "nothing to check
    against" must not quietly become "load it unchecked".
    """
    path = Path(path)
    record = manifest.get("artifact")
    if not record or "sha256" not in record:
        raise ArtifactIntegrityError(
            f"the manifest has no artifact.sha256, refusing to load {path.name}. "
            "The artifact predates the check: cut a new model release."
        )
    if record.get("file") != path.name:
        raise ArtifactIntegrityError(
            f"the manifest describes {record.get('file')!r}, but {path.name!r} is being loaded"
        )
    actual = file_sha256(path)
    if actual != record["sha256"]:
        raise ArtifactIntegrityError(
            f"sha256 of {path.name} does not match the manifest: "
            f"expected {record['sha256'][:16]}…, got {actual[:16]}…"
        )
