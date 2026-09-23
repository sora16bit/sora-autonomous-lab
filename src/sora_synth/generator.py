from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import CAT, CONTINUOUS, Dataset, rare_mask
from .outcomes import OutcomeModel, attach_outcomes, fit_outcomes


def _sobol_uniforms(dimension: int, n: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    if n <= 0:
        return np.empty((0, dimension))
    m = int(np.ceil(np.log2(n)))
    points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(m)
    return np.clip(points[:n], 1e-9, 1.0 - 1e-9)


def _log_transform(values: np.ndarray, col: str) -> np.ndarray:
    return np.log1p(np.maximum(values, 0.0)) if col in {"TG", "ALT"} else values.astype(float)


def _inverse_transform(values: np.ndarray, col: str) -> np.ndarray:
    return np.expm1(values) if col in {"TG", "ALT"} else values


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    x, w = values[order].astype(float), weights[order].astype(float)
    cs = np.cumsum(w)
    if not len(x) or cs[-1] <= 0:
        return np.full(len(q), np.nan)
    return np.interp(np.clip(q, 0, 1), (cs - 0.5 * w) / cs[-1], x, left=x[0], right=x[-1])


def _nearest_corr(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    eig, vec = np.linalg.eigh(matrix)
    fixed = vec @ np.diag(np.clip(eig, 1e-5, None)) @ vec.T
    d = np.sqrt(np.clip(np.diag(fixed), 1e-9, None))
    fixed /= np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _weighted_corr(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a correlation matrix using the same mixture weights as marginals."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or len(values) <= 1 or len(weights) != len(values):
        return np.eye(values.shape[1] if values.ndim == 2 else 1)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return np.eye(values.shape[1])
    weights = weights / total
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    corr = covariance / np.outer(scale, scale)
    return _nearest_corr(corr)


def _rank_uniform(x: np.ndarray) -> np.ndarray:
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return ranks / (len(x) + 1.0)


def _sample_correlated_normals(corr: np.ndarray, n: int, rng: np.random.Generator, *, quasi: bool = False) -> np.ndarray:
    """有限標本の相関揺らぎを校正した潜在正規サンプルを作る。

    生の多変量正規抽選だけでは、n=数十の希少層で相関行列が大きく揺れ、U_genの
    相関指標を不要に落とす。行ごとのB値を再利用せず、乱数を白色化して目標相関へ
    再着色するだけなので、個人行のコピーにはならない。
    """
    if quasi:
        seed = int(rng.integers(0, 2**32 - 1))
        unit = _sobol_uniforms(corr.shape[0], n, seed)
        raw = norm.ppf(unit)
    else:
        raw = rng.standard_normal((n, corr.shape[0]))
    if n <= corr.shape[0] + 1:
        return raw @ np.linalg.cholesky(corr + 1e-8 * np.eye(corr.shape[0])).T
    sample_cov = np.cov(raw, rowvar=False)
    l_sample = np.linalg.cholesky(sample_cov + 1e-8 * np.eye(corr.shape[0]))
    l_target = np.linalg.cholesky(corr + 1e-8 * np.eye(corr.shape[0]))
    return raw @ np.linalg.inv(l_sample.T) @ l_target.T


@dataclass
class LayerModel:
    component: int
    corr: np.ndarray
    mean_by_sex: dict[int, np.ndarray]
    scale_by_sex: dict[int, np.ndarray]
    quantiles_by_sex: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]
    cat_probs: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]
    source_n: int
    b_n: int
    prior_n: int


@dataclass
class SoraModel:
    layers: dict[int, LayerModel]
    outcome: OutcomeModel
    schema: dict
    config: dict
    horizon: float
    n_rows: int
    b_hash: str = ""
    diagnostics: dict = field(default_factory=dict)

    def sample(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        covars = self.sample_covariates(seed)
        uniforms = None
        if self.config.get("rng_mode") == "sobol":
            unit = _sobol_uniforms(2, len(covars), seed + 17_001)
            uniforms = (unit[:, 0], unit[:, 1])
        out = attach_outcomes(covars, self.outcome, rng, uniforms)
        return out[["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]]

    def sample_covariates(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        counts = self._component_counts(rng)
        parts = [self._sample_layer(self.layers[k], n, rng) for k, n in counts.items() if n]
        covars = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(2**31 - 1))).reset_index(drop=True)
        return covars

    def sample_with_uniforms(self, covariates: pd.DataFrame, u_onset: np.ndarray, u_death: np.ndarray) -> pd.DataFrame:
        out = attach_outcomes(covariates, self.outcome, np.random.default_rng(0), (u_onset, u_death))
        # The submitted form has no record_id.  Keep a deterministic internal order until export.
        return out[["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]]

    def _component_counts(self, rng: np.random.Generator) -> dict[int, int]:
        rare_count = max(8, int(round(self.diagnostics.get("b_rare_detected", self.n_rows * 0.023) * self.diagnostics.get("rare_multiplier", 1.0))))
        rare_count = min(max(8, rare_count), self.n_rows - 1)
        return {1: rare_count, 0: self.n_rows - rare_count}

    def _sample_layer(self, layer: LayerModel, n: int, rng: np.random.Generator) -> pd.DataFrame:
        sex_values, sex_probs = layer.cat_probs["sex"][0]
        sex = rng.choice(sex_values, n, p=sex_probs).astype(int)
        z = _sample_correlated_normals(layer.corr, n, rng, quasi=self.config.get("rng_mode") == "sobol")
        jitter = self.config.get("jitter", {}).get("rare" if layer.component else "general", 0.02)
        z += rng.normal(0.0, float(jitter), z.shape)
        out = pd.DataFrame(index=np.arange(n))
        for j, col in enumerate(CONTINUOUS):
            vals = np.empty(n, dtype=float)
            for s in np.unique(sex):
                ix = sex == s
                params = layer.quantiles_by_sex.get(int(s), {}).get(col)
                if params is None:
                    params = next(iter(layer.quantiles_by_sex.values()))[col]
                u = norm.cdf(z[ix, j])
                vals[ix] = _inverse_transform(_weighted_quantile(params[0], params[1], u), col)
            out[col] = vals
        out["sex"] = sex
        for col in ("smoking", "prefecture"):
            vals = np.empty(n, dtype=object)
            by_sex = layer.cat_probs[col]
            for s in np.unique(sex):
                ix = sex == s
                cats, probs = by_sex.get(int(s), next(iter(by_sex.values())))
                vals[ix] = rng.choice(cats, int(ix.sum()), p=probs)
            out[col] = vals
        self._clip(out)
        return out[["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG"]]

    def _clip(self, out: pd.DataFrame) -> None:
        age_lo, age_hi = map(float, self.config.get("population", {}).get("age_range", self.schema.get("age_range", [40, 74])))
        out["age"] = np.floor(np.clip(out["age"].astype(float), age_lo, age_hi)).astype(int)
        for col, bounds in self.schema.get("ranges", {}).items():
            if col in out:
                lo, hi = map(float, bounds)
                out[col] = np.clip(out[col].astype(float), lo, hi)
        out["sex"] = np.rint(out["sex"].astype(float)).astype(int).clip(0, 1)
        out["smoking"] = np.rint(out["smoking"].astype(float)).astype(int).clip(0, 1)


def _weighted_layer_stats(b: pd.DataFrame, prior: pd.DataFrame, columns: list[str], tau: float, component: int, b_mask: np.ndarray, p_mask: np.ndarray) -> LayerModel:
    b_part, p_part = b.loc[b_mask].copy(), prior.loc[p_mask].copy()
    if len(b_part) == 0:
        b_part = b.copy()
    if len(p_part) == 0:
        p_part = prior.copy()
    all_df = pd.concat([b_part, p_part], ignore_index=True)
    w_b = len(b_part) / max(len(b_part) + tau, 1.0)
    weights = np.r_[np.full(len(b_part), w_b / len(b_part)), np.full(len(p_part), (1.0 - w_b) / len(p_part))]
    means: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    quantiles: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for s in (0, 1):
        ix = all_df.sex.to_numpy(dtype=int) == s
        if not ix.any():
            ix = np.ones(len(all_df), dtype=bool)
        vals = np.column_stack([_log_transform(all_df.loc[ix, c].to_numpy(float), c) for c in columns])
        ww = weights[ix]
        means[s] = np.average(vals, axis=0, weights=ww)
        scales[s] = np.sqrt(np.average((vals - means[s]) ** 2, axis=0, weights=ww)).clip(1e-6)
        quantiles[s] = {c: (_log_transform(all_df.loc[ix, c].to_numpy(float), c), ww) for c in columns}
    residual = []
    residual_weights = []
    for s in (0, 1):
        ix = all_df.sex.to_numpy(dtype=int) == s
        if ix.any():
            vals = np.column_stack([_log_transform(all_df.loc[ix, c].to_numpy(float), c) for c in columns])
            residual.append((vals - means[s]) / scales[s])
            residual_weights.append(weights[ix])
    z = np.vstack(residual) if residual else np.zeros((2, len(columns)))
    rw = np.concatenate(residual_weights) if residual_weights else np.ones(len(z))
    corr = _weighted_corr(z, rw) if len(z) > 2 else np.eye(len(columns))
    cat_probs: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for col in ("sex", "smoking", "prefecture"):
        cat_probs[col] = {}
        for s in (0, 1):
            ix = all_df.sex.to_numpy(dtype=int) == s if col != "sex" else np.ones(len(all_df), dtype=bool)
            raw_vals = all_df.loc[ix, col].to_numpy()
            vals = np.unique(raw_vals)
            local_weights = weights[ix]
            p = np.asarray([local_weights[raw_vals == value].sum() for value in vals], dtype=float)
            p = p / p.sum() if p.sum() > 0 else np.full(len(vals), 1.0 / max(1, len(vals)))
            cat_probs[col][s] = (vals, p)
    return LayerModel(component, corr, means, scales, quantiles, cat_probs, len(b_part), len(b_part), len(p_part))


def fit_model(dataset: Dataset, *, tau_general: float = 100.0, tau_rare: float = 100.0, rare_multiplier: float = 1.0, jitter_general: float = 0.02, jitter_rare: float = 0.05, penalizer: float = 0.0, rng_mode: str = "random") -> SoraModel:
    if rng_mode not in {"random", "sobol"}:
        raise ValueError("rng_mode must be random or sobol")
    b = dataset.b.drop(columns=["record_id"], errors="ignore").copy()
    a = dataset.a_bg.drop(columns=["record_id"], errors="ignore").copy()
    b_rare, a_rare = rare_mask(b, dataset.utility_ref), rare_mask(a, dataset.utility_ref)
    layers = {
        0: _weighted_layer_stats(b, a, list(CONTINUOUS), tau_general, 0, ~b_rare, ~a_rare),
        1: _weighted_layer_stats(b, a, list(CONTINUOUS), tau_rare, 1, b_rare, a_rare),
    }
    cfg = dict(dataset.config)
    cfg["jitter"] = {"general": jitter_general, "rare": jitter_rare}
    cfg["rng_mode"] = rng_mode
    outcome = fit_outcomes(b, dataset.horizon, penalizer=penalizer)
    return SoraModel(layers, outcome, dataset.schema, cfg, dataset.horizon, len(b), diagnostics={"b_rare_detected": int(b_rare.sum()), "rare_multiplier": rare_multiplier, "tau_general": tau_general, "tau_rare": tau_rare, "jitter_general": jitter_general, "jitter_rare": jitter_rare, "penalizer": penalizer})


def save_model_config(path: str | Path, **kwargs) -> None:
    Path(path).write_text(json.dumps(kwargs, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
