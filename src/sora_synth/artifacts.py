from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import Dataset, sha256_file
from .official import hash_frame


def write_candidate(c: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(p, index=False, lineterminator="\n", float_format="%.17g")
    return sha256_file(p)


def write_manifest(path: str | Path, *, dataset: Dataset, kit_dir: str | Path, c: pd.DataFrame | None = None, **extra) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kit_dir": str(Path(kit_dir).resolve()),
        "input_hashes": dataset.hash_inputs(),
        "n_rows_B": len(dataset.b),
        "b_censor_horizon": dataset.horizon,
        "c_sha256": hash_frame(c) if c is not None else None,
        **extra,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
