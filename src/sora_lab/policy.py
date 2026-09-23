from __future__ import annotations

from typing import Any, Iterable


def _finite(value: Any) -> bool:
    try:
        import math
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def proxy_score(row: dict[str, Any], *, utility_buffer: float = 0.01) -> float | None:
    """Conservative comparison score; never presents itself as official Anon."""
    utility = row.get("utility")
    if not _finite(utility):
        return None
    facets = [row.get(k) for k in ("U_gen", "U_spec", "U_rare", "U_valid")]
    if not all(_finite(x) for x in facets):
        return None
    mia = row.get("r_MIA")
    aia = row.get("r_AIA")
    if not (_finite(mia) and _finite(aia)):
        return None
    u = min(float(utility), *(float(x) for x in facets))
    # The buffer is an engineering margin for the rare facet only.  Applying it
    # to all facets changes the objective and can hide a real utility regression.
    u = min(float(utility), float(row["U_gen"]), float(row["U_spec"]),
            max(0.0, float(row["U_rare"]) - utility_buffer), float(row["U_valid"]))
    anon_proxy = max(0.0, 1.0 - max(float(mia), float(aia)))
    return min(u, anon_proxy)


def admissible(row: dict[str, Any], *, min_utility: float = 0.0, max_mia: float = 1.0, max_aia: float = 1.0) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("validation_ok") is not True:
        reasons.append("validation_failed")
    if row.get("audit_status") not in {"complete", "proxy_complete"}:
        reasons.append("audit_incomplete")
    if not _finite(row.get("utility")) or float(row["utility"]) < min_utility:
        reasons.append("utility_below_floor")
    if not _finite(row.get("r_MIA")) or float(row["r_MIA"]) > max_mia:
        reasons.append("mia_risk_above_limit")
    if not _finite(row.get("r_AIA")) or float(row["r_AIA"]) > max_aia:
        reasons.append("aia_risk_above_limit")
    return not reasons, reasons


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize flattened sora_synth result keys for lab policy code."""
    normalized = dict(row)
    for facet in ("U_gen", "U_spec", "U_rare", "U_valid"):
        normalized.setdefault(facet, row.get(f"facet_{facet}"))
    # Distribution diagnostics are deliberately not risk estimates.
    normalized.setdefault("r_MIA", row.get("mia_risk"))
    normalized.setdefault("r_AIA", row.get("aia_risk"))
    return normalized


def pareto_front(rows: Iterable[dict[str, Any]], *, maximize: tuple[str, ...] = ("utility",), minimize: tuple[str, ...] = ("r_MIA", "r_AIA")) -> list[dict[str, Any]]:
    rows = [dict(r) for r in rows]
    usable = [r for r in rows if all(_finite(r.get(k)) for k in (*maximize, *minimize))]
    front: list[dict[str, Any]] = []
    for candidate in usable:
        dominated = False
        for other in usable:
            if other is candidate:
                continue
            no_worse = all(float(other[k]) >= float(candidate[k]) for k in maximize) and all(float(other[k]) <= float(candidate[k]) for k in minimize)
            strictly_better = any(float(other[k]) > float(candidate[k]) for k in maximize) or any(float(other[k]) < float(candidate[k]) for k in minimize)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def choose_verified(rows: Iterable[dict[str, Any]], *, min_utility: float = 0.0,
                    max_mia: float = 0.05, max_aia: float = 0.10) -> dict[str, Any]:
    """Choose the best row only after a complete, bounded proxy audit.

    Missing or distribution-only diagnostics are rejected.  The MIA default is
    the near-zero guard implied by the preliminary scoring threshold; callers
    can tighten it further for a final submission.
    """
    candidates: list[dict[str, Any]] = []
    for raw in rows:
        row = normalize_result_row(dict(raw))
        ok, _ = admissible(row, min_utility=min_utility, max_mia=max_mia, max_aia=max_aia)
        if ok:
            candidates.append(row)
    if not candidates:
        raise ValueError("監査済みの採用候補がありません")
    # Utility alone can select a candidate whose privacy risk dominates the
    # competition score.  Prefer the conservative proxy objective; retain
    # utility as a deterministic tie-breaker so a defense tie does not cause
    # arbitrary churn.
    return max(candidates, key=lambda row: (
        float(proxy_score(row) if proxy_score(row) is not None else float("-inf")),
        float(row["utility"]),
    ))
