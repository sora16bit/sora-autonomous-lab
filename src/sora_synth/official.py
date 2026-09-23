from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from .data import Dataset, SUBMITTED, validate_frame


def _kit_path(kit_dir: str | Path) -> Path:
    p = Path(kit_dir)
    src = p / "starter" / "src"
    if not src.exists():
        raise FileNotFoundError(f"公式キットのstarter/srcがありません: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def official_score(c: pd.DataFrame, dataset: Dataset, kit_dir: str | Path) -> dict:
    """公式キットの純粋な自己採点器を呼ぶ。秘密ファイルは出力しない。"""
    _kit_path(kit_dir)
    from pwscup2026.scoring.result import ReportLevel
    from pwscup2026.scoring.utility import UtilityReference, score_utility
    from pwscup2026.kit.codabench.scoring_io import _canonicalize_c

    errors = validate_frame(c, dataset)
    if errors:
        raise ValueError("Cの検証に失敗しました: " + "; ".join(errors))
    b = dataset.b.copy()
    c_canon = _canonicalize_c(c.copy())
    ref = UtilityReference(
        mu=pd.Series(dataset.utility_ref["mu"]).to_numpy(float),
        sigma=pd.DataFrame(dataset.utility_ref["sigma"]).to_numpy(float),
        band=dict(dataset.utility_ref["band"]),
        b_is_rare=dataset.rare_truth,
        km_signal_D=dataset.utility_ref.get("km_signal_D"),
        km_floor=dataset.utility_ref.get("km_floor"),
        a_bg=dataset.a_bg.copy(),
    )
    result = score_utility(c_canon, b, ref, dataset.config, level=ReportLevel.BREAKDOWN)
    return {"utility": float(result.score), "facets": {k: float(v) for k, v in result.facets.items()}, "detail": result.detail or {}}


def validate_with_kit(c: pd.DataFrame, dataset: Dataset, kit_dir: str | Path) -> list[str]:
    """公式validateの結果を文字列化する。失敗時も例外にせず監査結果へ残す。"""
    _kit_path(kit_dir)
    from pwscup2026.common.submission_schema import validate_c_submission

    # The kit's stable pure validator operates on a frame and config. Package/zip checks
    # are intentionally left to the final package step, where token.txt is available.
    errors = validate_frame(c, dataset)
    try:
        out = validate_c_submission(c, dataset.config, n_expected=len(dataset.b))
        errors.extend(str(e) for e in out.errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"公式validator呼出し失敗: {type(exc).__name__}: {exc}")
    return errors


def hash_frame(c: pd.DataFrame) -> str:
    payload = c.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    return hashlib.sha256(payload).hexdigest()
