from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .data import CONTINUOUS, rare_mask


@dataclass(frozen=True)
class AuditResult:
    values: dict[str, float]
    status: dict[str, str]


def _numeric(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].astype(float).to_numpy()


def _roc_tpr(y: np.ndarray, score: np.ndarray, fpr: float = 0.01) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    pos, neg = int(y.sum()), len(y) - int(y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    # A ROC operating point moves entire tied score blocks, matching the official attack code.
    end = np.r_[np.flatnonzero(np.diff(score) != 0), len(score) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[end] / pos]
    fpr_values = np.r_[0.0, np.cumsum(1 - y)[end] / neg]
    return float(np.interp(fpr, fpr_values, tpr))


def _perm_beta(y: np.ndarray, rare: np.ndarray, score: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    t_all = _roc_tpr(y, score)
    if int((y & rare).sum()) > 0:
        t_rare = _roc_tpr(y[rare], score[rare])
        observed = max(t_all, t_rare)
    else:
        t_rare = float("nan")
        observed = t_all
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(y))
        yp, rp = y[p], rare[p]
        all_v = _roc_tpr(yp, score)
        rare_v = _roc_tpr(yp[rp], score[rp]) if int((yp & rp).sum()) else float("nan")
        null[i] = max(all_v, rare_v) if np.isfinite(rare_v) else all_v
    chance = float(np.nanpercentile(null, 95))
    beta_raw = float(observed)
    beta = max(beta_raw - chance + 0.01, 0.01)
    ratio = beta / 0.01
    a_mia = float(np.clip((ratio - 1.0) / 11.0, 0.0, 1.0))
    return {"tpr_all": t_all, "tpr_rare": t_rare, "beta_raw": beta_raw, "beta_chance": chance, "beta": beta, "ratio": ratio, "a_mia": a_mia}


def _attack_features(df: pd.DataFrame) -> np.ndarray:
    cols = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death"]
    return df[cols].astype(float).to_numpy()


def _distance_score(c: pd.DataFrame, candidates: pd.DataFrame, ref: pd.DataFrame, k: int = 1,
                    columns: list[str] | None = None) -> np.ndarray:
    cols = columns or ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"]
    scaler = StandardScaler().fit(pd.concat([c[cols], ref[cols]], ignore_index=True).astype(float))
    c_x, q_x, r_x = scaler.transform(c[cols]), scaler.transform(candidates[cols]), scaler.transform(ref[cols])
    kk_c, kk_r = min(int(k), len(c_x)), min(int(k), len(r_x))
    dc = NearestNeighbors(n_neighbors=kk_c).fit(c_x).kneighbors(q_x)[0].mean(axis=1)
    dr = NearestNeighbors(n_neighbors=kk_r).fit(r_x).kneighbors(q_x)[0].mean(axis=1)
    return -(dc / np.clip(dr, 1e-9, None))


def _blocked_distance_score(c: pd.DataFrame, candidates: pd.DataFrame, ref: pd.DataFrame, k: int = 1) -> np.ndarray:
    """性別×県ブロック内の最近傍比。空ブロックは全体距離へ安全にフォールバックする。"""
    num = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"]
    scaler = StandardScaler().fit(pd.concat([c[num], ref[num]], ignore_index=True).astype(float))
    cx, qx, rx = scaler.transform(c[num]), scaler.transform(candidates[num]), scaler.transform(ref[num])
    global_c = NearestNeighbors(n_neighbors=min(k, len(cx))).fit(cx)
    global_r = NearestNeighbors(n_neighbors=min(k, len(rx))).fit(rx)
    out = np.empty(len(candidates), dtype=float)
    c_keys = c[["sex", "prefecture"]].astype(str).agg("/".join, axis=1).to_numpy()
    q_keys = candidates[["sex", "prefecture"]].astype(str).agg("/".join, axis=1).to_numpy()
    r_keys = ref[["sex", "prefecture"]].astype(str).agg("/".join, axis=1).to_numpy()
    for key in np.unique(q_keys):
        qix = np.flatnonzero(q_keys == key)
        cix, rix = np.flatnonzero(c_keys == key), np.flatnonzero(r_keys == key)
        if len(cix) and len(rix):
            kk_c, kk_r = min(k, len(cix)), min(k, len(rix))
            dc = NearestNeighbors(n_neighbors=kk_c).fit(cx[cix]).kneighbors(qx[qix])[0].mean(axis=1)
            dr = NearestNeighbors(n_neighbors=kk_r).fit(rx[rix]).kneighbors(qx[qix])[0].mean(axis=1)
        else:
            dc = global_c.kneighbors(qx[qix])[0].mean(axis=1)
            dr = global_r.kneighbors(qx[qix])[0].mean(axis=1)
        out[qix] = -(dc / np.clip(dr, 1e-9, None))
    return out


def _marginal_score(c: pd.DataFrame, candidates: pd.DataFrame, ref: pd.DataFrame,
                    columns: list[str] | None = None) -> np.ndarray:
    scores = np.zeros(len(candidates))
    for col in columns or ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]:
        vc, vr = np.sort(c[col].astype(float).to_numpy()), np.sort(ref[col].astype(float).to_numpy())
        q = candidates[col].astype(float).to_numpy()
        ic = np.clip(np.searchsorted(vc, q), 1, len(vc) - 1)
        ir = np.clip(np.searchsorted(vr, q), 1, len(vr) - 1)
        dc = np.minimum(abs(q - vc[ic - 1]), abs(q - vc[ic]))
        dr = np.minimum(abs(q - vr[ir - 1]), abs(q - vr[ir]))
        scores += -(dc / np.clip(dr, 1e-9, None))
    return scores / 7.0


def _shadow_score(c: pd.DataFrame, candidates: pd.DataFrame, ref: pd.DataFrame, seed: int,
                  *, max_iter: int = 160, max_leaf_nodes: int = 15,
                  l2_regularization: float = 0.0) -> np.ndarray:
    cols = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death"]
    x = pd.concat([c[cols], ref[cols]], ignore_index=True).astype(float)
    y = np.r_[np.ones(len(c)), np.zeros(len(ref))]
    clf = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                                         l2_regularization=l2_regularization,
                                         random_state=seed).fit(x, y)
    return clf.predict_proba(candidates[cols].astype(float))[:, 1]


def _independent_shadow_score(shadow_members: list[pd.DataFrame],
                              shadow_nonmembers: list[pd.DataFrame],
                              candidates: pd.DataFrame, *, seed: int,
                              max_iter: int = 160, max_leaf_nodes: int = 15,
                              l2_regularization: float = 1.0) -> np.ndarray:
    """Fit membership on independent worlds, then score the target world.

    This is intentionally separate from ``_shadow_score``.  The latter is a
    cheap C-versus-reference distribution diagnostic; this path gives the
    classifier member labels from independently generated worlds and never
    trains on the target candidate/reference rows.
    """
    if not shadow_members or len(shadow_members) != len(shadow_nonmembers):
        raise ValueError("shadow member/nonmember pairs are required")
    cols = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death"]
    train = pd.concat([*shadow_members, *shadow_nonmembers], ignore_index=True)
    y = np.r_[*[np.ones(len(x), dtype=int) for x in shadow_members],
              *[np.zeros(len(x), dtype=int) for x in shadow_nonmembers]]
    clf = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                                         l2_regularization=l2_regularization,
                                         random_state=seed).fit(train[cols].astype(float), y)
    return clf.predict_proba(candidates[cols].astype(float))[:, 1]


def evaluate_mia(
    c: pd.DataFrame,
    member: pd.DataFrame,
    nonmember: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    rare_ref: dict | None = None,
    rare_labels: np.ndarray | None = None,
    attack_specs: list[dict] | None = None,
    shadow_members: list[pd.DataFrame] | None = None,
    shadow_nonmembers: list[pd.DataFrame] | None = None,
    n_perm: int = 500,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """再fitを伴うmember/nonmember実験。cはmemberだけから作られたCであることを呼出側が保証する。"""
    if len(member) == 0 or len(nonmember) == 0:
        raise ValueError("MIA評価にはmember/nonmemberの両方が必要です")
    if reference is None:
        raise ValueError("MIA監査には評価対象と独立した reference 集合が必要です")
    # A reference row must never also be scored as an evaluation candidate.
    # Fingerprints work even when record_id has intentionally been removed.
    def fingerprints(frame: pd.DataFrame) -> set[tuple[str, ...]]:
        clean = frame.drop(columns=["record_id"], errors="ignore").copy()
        # ``astype(str).to_numpy().tobytes()`` serializes object pointers, not
        # cell contents.  It can report false overlaps across independent
        # frames.  Tuples of strings are deterministic content fingerprints.
        return {tuple(row) for row in clean.astype(str).itertuples(index=False, name=None)}
    ref_fp = fingerprints(reference)
    if ref_fp & (fingerprints(member) | fingerprints(nonmember)):
        raise ValueError("MIAのreferenceと評価対象が重複しています")
    candidates = pd.concat([member, nonmember], ignore_index=True)
    y = np.r_[np.ones(len(member), dtype=int), np.zeros(len(nonmember), dtype=int)]
    if rare_labels is not None:
        # A holdout audit knows the cohort's rare-member truth.  Using that
        # label keeps the rare ROC aligned with the attack definition rather
        # than silently substituting a detector fit on the scored rows.
        rare = np.asarray(rare_labels, dtype=bool)
        if len(rare) != len(candidates):
            raise ValueError("rare_labelsの長さが評価候補と一致しません")
        rare_kind = "provided_labels"
    else:
        rare_cols = list(CONTINUOUS)
        if rare_ref is None:
            rare_ref = {
                "columns": rare_cols,
                "mu": candidates[rare_cols].mean().to_numpy(),
                "sigma": np.cov(candidates[rare_cols].astype(float).to_numpy(), rowvar=False),
                "band": {"lo": 4.0},
            }
        rare = rare_mask(candidates, rare_ref)
        rare_kind = "detector"
    ref = reference.drop(columns=["record_id"], errors="ignore")
    results: dict[str, dict[str, float]] = {}
    specs = attack_specs or [
        {"attack_id": "nn_ratio", "family": "distance", "params": {"k": 1}},
        {"attack_id": "knn5_ratio", "family": "distance", "params": {"k": 5}},
        {"attack_id": "blocked_nn", "family": "blocked_distance", "params": {"k": 1}},
        {"attack_id": "marginal_support", "family": "marginal", "params": {}},
        {"attack_id": "shadow_c_vs_ref", "family": "shadow", "params": {}},
    ]
    attacks: dict[str, np.ndarray] = {}
    ensemble_specs: list[tuple[str, dict]] = []
    for spec in specs:
        name = str(spec.get("attack_id", spec.get("name", "attack")))
        family = str(spec.get("family", ""))
        params = dict(spec.get("params", {}))
        if family == "distance":
            attacks[name] = _distance_score(c, candidates, ref, int(params.get("k", 1)), list(params.get("columns", ())) or None)
        elif family == "blocked_distance":
            attacks[name] = _blocked_distance_score(c, candidates, ref, int(params.get("k", 1)))
        elif family == "marginal":
            attacks[name] = _marginal_score(c, candidates, ref, list(params.get("columns", ())) or None)
        elif family == "shadow":
            if shadow_members is not None or shadow_nonmembers is not None:
                attacks[name] = _independent_shadow_score(
                    shadow_members or [], shadow_nonmembers or [], candidates, seed=seed,
                    max_iter=int(params.get("max_iter", 160)),
                    max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                    l2_regularization=float(params.get("l2_regularization", 1.0)))
            else:
                # Backward-compatible distribution diagnostic when callers do
                # not provide an independent shadow world.  The metadata makes
                # the weaker path visible to downstream promotion gates.
                attacks[name] = _shadow_score(c, candidates, ref, seed,
                                               max_iter=int(params.get("max_iter", 160)),
                                               max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                                               l2_regularization=float(params.get("l2_regularization", 0.0)))
        elif family == "ensemble":
            # Resolve after primitive attacks so child order in a serialized
            # suite cannot change whether an ensemble is evaluable.
            ensemble_specs.append((name, params))
        else:
            raise ValueError(f"unsupported MIA attack family: {family}")
    for name, params in ensemble_specs:
        children = [str(child) for child in params.get("children", ())]
        if not children or any(child not in attacks for child in children):
            raise ValueError(f"ensemble {name} refers to an unavailable child")
        ranks = []
        for child in children:
            values = attacks[child]
            order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
            ranks.append(order.astype(float) / max(1, len(values) - 1))
        attacks[name] = np.mean(np.vstack(ranks), axis=0)
    for name, score in attacks.items():
        results[name] = _perm_beta(y, rare, score, n_perm, seed)
    # Keep this metadata inside the result so downstream promotion logic can
    # tell whether the rare-specific path was actually exercised.
    results["_meta"] = {"rare_kind": rare_kind, "rare_n": int(rare.sum()), "candidate_n": len(candidates),
                         "shadow_kind": "independent_world" if shadow_members is not None or shadow_nonmembers is not None
                         else "c_vs_reference_diagnostic"}
    return results


def evaluate_shadow_mia(c: pd.DataFrame, shadow_members: list[pd.DataFrame],
                        shadow_nonmembers: list[pd.DataFrame], member: pd.DataFrame,
                        nonmember: pd.DataFrame, *, rare_ref: dict | None = None,
                        n_perm: int = 500, seed: int = 0) -> dict[str, float]:
    """Train a shadow membership classifier from independent member/nonmember worlds."""
    if not shadow_members or len(shadow_members) != len(shadow_nonmembers):
        raise ValueError("shadow member/nonmember pairs are required")
    train = pd.concat([*shadow_members, *shadow_nonmembers], ignore_index=True)
    y_train = np.r_[*[np.ones(len(x), dtype=int) for x in shadow_members],
                    *[np.zeros(len(x), dtype=int) for x in shadow_nonmembers]]
    features = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death"]
    clf = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, l2_regularization=1.0,
                                         random_state=seed).fit(train[features].astype(float), y_train)
    candidates = pd.concat([member, nonmember], ignore_index=True)
    y = np.r_[np.ones(len(member), dtype=int), np.zeros(len(nonmember), dtype=int)]
    rare = rare_mask(candidates, rare_ref) if rare_ref is not None else np.zeros(len(candidates), dtype=bool)
    score = clf.predict_proba(candidates[features].astype(float))[:, 1]
    return _perm_beta(y, rare, score, n_perm, seed + 1)


def _time_hat(c: pd.DataFrame, q: pd.DataFrame, ref: pd.DataFrame, k: int = 1) -> np.ndarray:
    cols = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
    pre = ColumnTransformer([("num", StandardScaler(), [0, 1, 2, 3, 4]), ("cat", OneHotEncoder(handle_unknown="ignore"), [5, 6])])
    # NearestNeighbors cannot be the final string step in a pipeline; fit transformed data explicitly.
    tx = pre.fit_transform(pd.concat([c[cols], ref[cols]], ignore_index=True))
    # ColumnTransformer may return a scipy sparse matrix when categorical
    # columns are present.  ``len(sparse_matrix)`` is intentionally ambiguous
    # in scipy; use shape[0] so the AIA path works for both sparse and dense
    # encodings.
    c_n = len(c)
    tc, tq = tx[:c_n], pre.transform(q[cols])
    nn = NearestNeighbors(n_neighbors=min(k, tc.shape[0])).fit(tc)
    idx = nn.kneighbors(tq)[1]
    # Keep the requested neighbourhood.  Returning only idx[:, 0] made the
    # existing knn1/3/5 suite evaluate the same attack three times.
    times = c["time"].to_numpy(float)
    return times[idx].mean(axis=1)


def _aia_result(target: pd.DataFrame, control: pd.DataFrame, target_hat: np.ndarray,
                control_hat: np.ndarray, *, tau: float) -> dict[str, float]:
    pm = float(np.mean(np.abs(target_hat - target.time.to_numpy(float)) <= tau))
    pc = float(np.mean(np.abs(control_hat - control.time.to_numpy(float)) <= tau))
    r = float(np.clip((pm - pc) / max(1.0 - pc, 1e-9), 0.0, 1.0))
    return {"p_target": pm, "p_control": pc, "R": r}


def evaluate_aia_suite(c: pd.DataFrame, target: pd.DataFrame, control: pd.DataFrame, *, tau: float = 0.5,
                       seed: int = 0, attack_specs: list[dict] | None = None) -> dict[str, object]:
    """Run several C-only AIA attacks and retain the conservative maximum.

    The target/control ``time`` values are truth labels for scoring only.  They
    are never included in the attack reference or model fit; using them as a
    nearest-neighbour reference would leak the answer and inflate AIA.
    """
    if len(target) != len(control):
        raise ValueError("AIA target/controlの件数が一致しません")
    qi = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
    target = target.reset_index(drop=True)
    control = control.reset_index(drop=True)
    # C is the only labelled data an attacker is allowed to learn from.
    ref = c[qi + ["time"]].reset_index(drop=True)
    specs = attack_specs or [
        {"attack_id": "knn1", "family": "aia_knn", "params": {"k": 1}},
        {"attack_id": "knn3", "family": "aia_knn", "params": {"k": 3}},
        {"attack_id": "knn5", "family": "aia_knn", "params": {"k": 5}},
        {"attack_id": "regularized_tree", "family": "aia_tree", "params": {"max_iter": 120, "max_leaf_nodes": 15, "l2_regularization": 1.0}},
    ]
    methods: dict[str, dict[str, float]] = {}
    for spec in specs:
        name = str(spec.get("attack_id", "aia_attack"))
        family = str(spec.get("family", ""))
        params = dict(spec.get("params", {}))
        try:
            if family == "aia_knn":
                k = int(params.get("k", 1))
                ht = _time_hat(c, target[qi], ref, k)
                hc = _time_hat(c, control[qi], ref, k)
            elif family == "aia_tree":
                from sklearn.compose import ColumnTransformer
                from sklearn.ensemble import HistGradientBoostingRegressor
                from sklearn.preprocessing import OneHotEncoder

                num = ["age", "SBP", "TG", "HDL", "ALT"]
                cat = ["sex", "prefecture"]
                pre = ColumnTransformer([("num", "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
                x = pre.fit_transform(c[qi])
                if hasattr(x, "toarray"):
                    x = x.toarray()
                reg = HistGradientBoostingRegressor(
                    max_iter=int(params.get("max_iter", 120)),
                    max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                    l2_regularization=float(params.get("l2_regularization", 1.0)),
                    random_state=seed).fit(x, c["time"].to_numpy(float))
                target_x, control_x = pre.transform(target[qi]), pre.transform(control[qi])
                if hasattr(target_x, "toarray"):
                    target_x, control_x = target_x.toarray(), control_x.toarray()
                ht, hc = reg.predict(target_x), reg.predict(control_x)
            else:
                raise ValueError(f"unsupported AIA family: {family}")
            methods[name] = _aia_result(target, control, ht, hc, tau=tau)
        except Exception as exc:  # noqa: BLE001
            methods[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}  # type: ignore[assignment]
    complete = [row["R"] for row in methods.values() if isinstance(row, dict) and "R" in row]
    conservative = max(complete) if complete else float("nan")
    return {"methods": methods, "conservative_R": float(conservative), "reference": "C_only"}


def evaluate_aia(c: pd.DataFrame, target: pd.DataFrame, control: pd.DataFrame, *, tau: float = 0.5) -> dict[str, float]:
    """Backward-compatible single attack: C-only 1-nearest-neighbour AIA."""
    if len(target) != len(control):
        raise ValueError("AIA target/controlの件数が一致しません")
    qi = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
    target = target.reset_index(drop=True)
    control = control.reset_index(drop=True)
    ref = c[qi + ["time"]].reset_index(drop=True)
    return _aia_result(target, control, _time_hat(c, target[qi], ref, 1),
                       _time_hat(c, control[qi], ref, 1), tau=tau)


def audit_c_to_abg(c: pd.DataFrame, a_bg: pd.DataFrame, *, n_perm: int = 500, seed: int = 0) -> AuditResult:
    """C対A_bgの診断。これは公式MIAではなくdistribution_diagnosticとして保存する。"""
    a = a_bg.drop(columns=["record_id"], errors="ignore")
    c2 = c.drop(columns=["record_id"], errors="ignore")
    try:
        cols = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death"]
        x = pd.concat([c2[cols], a[cols]], ignore_index=True).astype(float)
        y = np.r_[np.ones(len(c2), dtype=int), np.zeros(len(a), dtype=int)]
        clf = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, random_state=seed).fit(x, y)
        auc = roc_auc_score(y, clf.predict_proba(x)[:, 1])
        values = {"c_vs_abg_shadow_auc": float(auc)}
        status = {"kind": "distribution_diagnostic"}
    except Exception as exc:  # noqa: BLE001
        values, status = {}, {"kind": "distribution_diagnostic", "error": f"{type(exc).__name__}: {exc}"}
    return AuditResult(values, status)
