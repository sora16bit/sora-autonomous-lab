from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PREDICTORS = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]


@dataclass
class CauseModel:
    model: object
    times: np.ndarray
    cumulative_hazard: np.ndarray


@dataclass
class OutcomeModel:
    onset: CauseModel
    death: CauseModel
    horizon: float


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[PREDICTORS].copy().astype(float)
    x["TG"] = np.log1p(np.maximum(x["TG"], 0.0))
    x["ALT"] = np.log1p(np.maximum(x["ALT"], 0.0))
    if "time" in df:
        x["time"] = df["time"].to_numpy(dtype=float)
    return x


def fit_outcomes(b: pd.DataFrame, horizon: float, penalizer: float = 0.0) -> OutcomeModel:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError

    def fit(event: str) -> CauseModel:
        frame = _frame(b)
        frame["_event"] = b[event].astype(int).to_numpy()
        try:
            model = CoxPHFitter(penalizer=penalizer).fit(frame, "time", "_event")
        except (ConvergenceError, ValueError, np.linalg.LinAlgError):
            if penalizer > 0.0:
                raise
            model = CoxPHFitter(penalizer=0.02).fit(frame, "time", "_event")
        base = model.baseline_cumulative_hazard_.iloc[:, 0]
        return CauseModel(model, base.index.to_numpy(float), base.to_numpy(float))

    return OutcomeModel(fit("onset"), fit("death"), float(horizon))


def _sample_event(cause: CauseModel, x: pd.DataFrame, rng: np.random.Generator, uniforms: np.ndarray | None = None) -> np.ndarray:
    q = _frame(x)
    ph = cause.model.predict_partial_hazard(q).to_numpy(float)
    u = rng.random(len(x)) if uniforms is None else np.asarray(uniforms, dtype=float)
    target = -np.log(np.clip(u, 1e-12, 1.0)) / np.clip(ph, 1e-12, None)
    h = cause.cumulative_hazard
    t = np.interp(target, h, cause.times, left=cause.times[0], right=np.inf)
    return np.where(target > h[-1], np.inf, t)


def attach_outcomes(covariates: pd.DataFrame, model: OutcomeModel, rng: np.random.Generator, uniforms: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    u_onset, u_death = uniforms if uniforms is not None else (None, None)
    onset_t = _sample_event(model.onset, covariates, rng, u_onset)
    death_t = _sample_event(model.death, covariates, rng, u_death)
    event_t = np.minimum(onset_t, death_t)
    onset = (onset_t <= death_t) & (onset_t <= model.horizon)
    death = (death_t < onset_t) & (death_t <= model.horizon)
    cens = ~(onset | death)
    out = covariates.copy()
    out["time"] = np.where(cens, model.horizon, np.minimum(event_t, model.horizon))
    out["onset"] = onset.astype(int)
    out["death"] = death.astype(int)
    return out
