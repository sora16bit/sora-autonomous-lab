"""Safe boundary for evaluating code imported from other team repositories.

Git mirrors are data until an explicit adapter declares how to call them.  The
adapter contract keeps the fixed official kit and private dist outside the
candidate checkout, bounds execution time, and records a content hash for
comparison.  The autonomous intake loop only triages adapters; it does not
execute arbitrary repository code.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    mirror: Path
    entrypoint: Path
    command: tuple[str, ...]


def validate_adapter(spec: AdapterSpec) -> list[str]:
    errors: list[str] = []
    root = spec.mirror.resolve()
    entry = spec.entrypoint.resolve()
    if not root.is_dir():
        errors.append("mirror_missing")
    if root not in entry.parents and entry != root:
        errors.append("entrypoint_outside_mirror")
    if not entry.exists():
        errors.append("entrypoint_missing")
    if not spec.command or any(not str(part).strip() for part in spec.command):
        errors.append("empty_command")
    if any(part in {"sh", "bash", "zsh", "fish", "cmd", "powershell"} for part in spec.command):
        errors.append("shell_command_forbidden")
    return errors


def adapter_digest(spec: AdapterSpec) -> str:
    """Hash the adapter declaration and entrypoint bytes for provenance."""
    h = hashlib.sha256()
    h.update(spec.name.encode())
    h.update(b"\0")
    h.update("\0".join(spec.command).encode())
    h.update(b"\0")
    h.update(spec.entrypoint.read_bytes())
    return h.hexdigest()


def run_adapter(spec: AdapterSpec, *, dist: Path, kit_dir: Path, out_dir: Path,
                timeout_seconds: int = 600) -> dict[str, object]:
    """Run one reviewed adapter with bounded, provenance-preserving I/O."""
    errors = validate_adapter(spec)
    if errors:
        raise ValueError("adapter rejected: " + ",".join(errors))
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1",
           "SORA_DIST": str(dist.resolve()), "SORA_KIT_DIR": str(kit_dir.resolve()),
           "SORA_OUT_DIR": str(out_dir.resolve())}
    proc = subprocess.run(list(spec.command), cwd=str(spec.mirror.resolve()), env=env,
                          capture_output=True, text=True, timeout=timeout_seconds, check=False)
    return {"name": spec.name, "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
            "adapter_sha256": adapter_digest(spec), "timeout_seconds": timeout_seconds}
