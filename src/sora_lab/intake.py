"""Read-only GitHub/team-code intake for the autonomous lab.

The intake mirror is deliberately separate from the working checkout.  A
commit is only a candidate until an adapter has been reviewed and tested under
the fixed official kit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    branch: str = "main"


def _git(path: Path, *args: str, timeout: int = 60) -> str:
    proc = subprocess.run(["git", "-C", str(path), *args], text=True,
                          capture_output=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()[-500:]}")
    return proc.stdout.strip()


def update_mirror(spec: RepoSpec, mirror_root: str | Path) -> dict[str, Any]:
    root = Path(mirror_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec.name
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["git", "clone", "--no-checkout", spec.url, str(path)], text=True,
                              capture_output=True, timeout=300, check=False)
        if proc.returncode:
            raise RuntimeError(f"git clone failed: {proc.stderr.strip()[-500:]}")
    old = None
    try:
        old = _git(path, "rev-parse", f"origin/{spec.branch}")
    except RuntimeError:
        pass
    _git(path, "fetch", "--prune", "origin", spec.branch, timeout=300)
    new = _git(path, "rev-parse", f"origin/{spec.branch}")
    changed = old != new
    return {"name": spec.name, "url": spec.url, "branch": spec.branch,
            "mirror": str(path), "previous_sha": old, "sha": new,
            "changed": changed, "fetched_at": time.time()}


def changed_paths(mirror: str | Path, old_sha: str | None, new_sha: str, *, max_files: int = 100) -> list[dict[str, Any]]:
    path = Path(mirror)
    if old_sha:
        raw = _git(path, "diff", "--name-status", old_sha, new_sha)
    else:
        raw = _git(path, "show", "--format=", "--name-status", new_sha)
    rows = []
    for line in raw.splitlines()[:max_files]:
        parts = line.split("\t")
        if len(parts) >= 2:
            status, rel = parts[0], parts[-1]
            if Path(rel).suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md", ".toml"}:
                rows.append({"status": status, "path": rel})
    return rows


def diff_excerpt(mirror: str | Path, old_sha: str | None, new_sha: str, *, max_chars: int = 12000) -> str:
    """Return a small redacted code diff for triage, never for execution."""
    path = Path(mirror)
    args = ["diff", "--unified=1", old_sha, new_sha] if old_sha else ["show", "--format=", "--unified=1", new_sha]
    raw = _git(path, *args, timeout=120)
    safe: list[str] = []
    for line in raw.splitlines():
        low = line.lower()
        if any(word in low for word in ("token", "password", "secret", "api_key", "private_key")):
            safe.append("[redacted sensitive-looking line]")
        elif len(line) <= 500:
            safe.append(line)
        if sum(len(x) + 1 for x in safe) >= max_chars:
            break
    return "\n".join(safe)[:max_chars]


def make_candidate_record(update: dict[str, Any], paths: list[dict[str, Any]], diff: str = "") -> dict[str, Any]:
    payload = {"repo": update.get("name"), "sha": update.get("sha"), "paths": paths, "diff": diff}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"candidate_id": digest[:16], "source_repo": update.get("name"),
            "commit": update.get("sha"), "paths": paths, "diff_excerpt": diff,
            "state": "discovered",
            "created_at": time.time(), "content_hash": digest}


def load_specs(path: str | Path) -> list[RepoSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("repo spec must be a JSON list")
    specs = []
    for item in payload:
        name = str(item["name"])
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe repository mirror name: {name!r}")
        specs.append(RepoSpec(name, str(item["url"]), str(item.get("branch", "main"))))
    return specs
