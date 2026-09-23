from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTINUOUS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
CAT = ("sex", "smoking", "prefecture")
SUBMITTED = ("age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death")
EVENT = ("time", "onset", "death")


@dataclass(frozen=True)
class Dataset:
    root: Path
    b: pd.DataFrame
    b_self: pd.DataFrame
    a_bg: pd.DataFrame
    schema: dict
    utility_ref: dict
    config: dict

    @property
    def horizon(self) -> float:
        cens = self.b.loc[(self.b["onset"] == 0) & (self.b["death"] == 0), "time"]
        return float(cens.max()) if len(cens) else float(self.config.get("onset", {}).get("horizon_years", 10.0))

    @property
    def rare_truth(self) -> np.ndarray:
        if "is_rare" not in self.b_self:
            raise ValueError("B_self.csv に is_rare 列がありません")
        return self.b_self["is_rare"].to_numpy(dtype=bool)

    def hash_inputs(self) -> dict[str, str]:
        return {name: sha256_file(self.root / name) for name in ("B.csv", "B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json")}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def load_config(kit_dir: str | Path) -> dict:
    import yaml

    candidates = [Path(kit_dir) / "participant_data" / "scoring_config.yaml", Path(kit_dir) / "starter" / "src" / "pwscup2026" / "kit" / "kit_config.yaml"]
    for p in candidates:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("本戦キットの scoring_config.yaml / kit_config.yaml が見つかりません")


def load_dataset(dist_dir: str | Path, kit_dir: str | Path) -> Dataset:
    root = Path(dist_dir)
    required = ("B.csv", "B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"本戦配布物が不足しています: {', '.join(missing)}")
    b = _read_csv(root / "B.csv")
    bs = _read_csv(root / "B_self.csv")
    a = _read_csv(root / "A_bg.csv")
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    ref = json.loads((root / "utility_ref.json").read_text(encoding="utf-8"))
    config = load_config(kit_dir)
    if len(b) != len(bs):
        raise ValueError(f"B/B_selfの行数が違います: {len(b)} != {len(bs)}")
    if "record_id" in b and "record_id" in bs and not np.array_equal(b.record_id.to_numpy(), bs.record_id.to_numpy()):
        raise ValueError("B.csv と B_self.csv のrecord_id順が一致しません")
    if set(b.columns) != set(SUBMITTED) | {"record_id"}:
        raise ValueError(f"B.csvの列が本戦配布形式と一致しません: {list(b.columns)}")
    return Dataset(root, b, bs, a, schema, ref, config)


def rare_mask(df: pd.DataFrame, utility_ref: dict) -> np.ndarray:
    rd = utility_ref
    cols = rd.get("columns") or ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
    mu = np.asarray(rd["mu"], dtype=float)
    sigma = np.asarray(rd["sigma"], dtype=float)
    band = rd.get("band", {})
    x = df[cols].to_numpy(dtype=float) - mu
    inv = np.linalg.pinv(sigma)
    distance = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", x, inv, x)))
    return distance >= float(band.get("lo", 4.0))


def validate_frame(c: pd.DataFrame, dataset: Dataset) -> list[str]:
    errors: list[str] = []
    if list(c.columns) != list(SUBMITTED):
        errors.append(f"列順/列名が不正: {list(c.columns)}")
    if len(c) != len(dataset.b):
        errors.append(f"行数が不正: {len(c)} != {len(dataset.b)}")
    for col in ("age", "sex", "smoking", "onset", "death"):
        if col in c and not np.isfinite(pd.to_numeric(c[col], errors="coerce")).all():
            errors.append(f"{col}に非有限値があります")
    for col, bounds in dataset.schema.get("ranges", {}).items():
        if col in c:
            lo, hi = map(float, bounds)
            x = c[col].to_numpy(dtype=float)
            if np.any((x < lo) | (x > hi)):
                errors.append(f"{col}が値域外です")
    if set(zip(c.onset.astype(int), c.death.astype(int))) - {(0, 0), (1, 0), (0, 1)}:
        errors.append("onset/deathが排他的な3状態ではありません")
    if np.any(c.time.to_numpy(dtype=float) <= 0):
        errors.append("timeに0以下があります")
    return errors
