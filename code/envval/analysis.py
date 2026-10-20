"""Analysis: mixed-effects models and leave-one-environment-family-out evaluation.

Two design choices here are load-bearing for the paper's credibility:

1. **The independent unit is the environment**, clustered by task family.
   Seeds are repeats, not replicates. Treating 180 runs as 180 independent
   observations is the most common statistical error in this literature.

2. **The headline metric is leave-one-environment-family-out (LOFO) R².** The
   practical decision is "should I build environments like this kind?", which is
   a generalisation question. An in-sample R² on 30 environments would answer
   a question nobody is asking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "fit_mixed_model",
    "lofo_evaluate",
    "predictor_correlations",
    "transfer_matrix_from_frame",
    "bootstrap_ci",
]


@dataclass
class MixedModelResult:
    """Result of a grouped regression. Coefficients are (estimate, stderr, p)."""

    formula: str
    coefs: dict[str, tuple[float, float, float]]
    n_obs: int
    n_groups: int
    resid_sd: float

    def get(self, name: str) -> tuple[float, float, float]:
        return self.coefs.get(name, (math.nan, math.nan, math.nan))

    def summary(self) -> str:
        lines = [f"formula: {self.formula}", f"n_obs={self.n_obs}  n_groups={self.n_groups}"]
        lines.append(f"{'term':<42}{'coef':>12}{'stderr':>12}{'p':>10}")
        for k, (b, se, p) in self.coefs.items():
            lines.append(f"{k:<42}{b:>12.4g}{se:>12.4g}{p:>10.4g}")
        return "\n".join(lines)


def fit_mixed_model(
    df: pd.DataFrame,
    formula: str,
    group_col: str,
) -> MixedModelResult:
    """Fit a linear mixed-effects model with a random intercept per group.

    Uses statsmodels MixedLM. Falls back to OLS with cluster-robust standard
    errors if the mixed model fails to converge (which happens with small or
    degenerate group counts) — and records which path was taken so the paper can
    report it honestly rather than silently.
    """
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[group_col]).copy()
    try:
        md = smf.mixedlm(formula, d, groups=d[group_col])
        res = md.fit(reml=True, method="lbfgs", maxiter=2000)
        coefs: dict[str, tuple[float, float, float]] = {}
        for name in res.params.index:
            try:
                se = float(res.bse[name])
                p = float(res.pvalues[name])
            except (KeyError, TypeError):
                se, p = math.nan, math.nan
            coefs[str(name)] = (float(res.params[name]), se, p)
        return MixedModelResult(
            formula=formula,
            coefs=coefs,
            n_obs=int(res.nobs),
            n_groups=int(d[group_col].nunique()),
            resid_sd=float(np.sqrt(res.scale)),
        )
    except Exception:
        ols = smf.ols(formula, d).fit(
            cov_type="cluster", cov_kwds={"groups": d[group_col]}
        )
        coefs = {
            str(n): (float(ols.params[n]), float(ols.bse[n]), float(ols.pvalues[n]))
            for n in ols.params.index
        }
        return MixedModelResult(
            formula=formula + "  [OLS + cluster-robust]",
            coefs=coefs,
            n_obs=int(ols.nobs),
            n_groups=int(d[group_col].nunique()),
            resid_sd=float(np.sqrt(ols.mse_resid)),
        )


def lofo_evaluate(
    df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    family_col: str = "family",
    model: str = "gbm",
) -> pd.DataFrame:
    """Leave-one-environment-family-out evaluation.

    For each family: hold it out entirely, fit the predictor on the remaining
    families, predict the held-out family, and score. Reports both R² and
    Spearman rho on the pooled out-of-sample predictions.

    This is the number the paper leads with, because it answers the question a
    lab actually has: "we are about to build a new environment type, will this
    model tell us whether it's worth it?"
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=[target]).copy()
    # drop predictors that are constant or entirely missing in this pool
    usable = [
        p
        for p in predictors
        if p in d.columns and d[p].notna().sum() >= 2 and d[p].nunique(dropna=True) > 1
    ]
    if not usable:
        return pd.DataFrame()
    d = d.dropna(subset=usable)
    families = sorted(d[family_col].unique())
    rows: list[dict] = []
    preds: list[float] = []
    actuals: list[float] = []

    for fam in families:
        train = d[d[family_col] != fam]
        test = d[d[family_col] == fam]
        if len(train) < 5 or len(test) == 0:
            continue
        Xtr = train[usable].to_numpy(dtype=float)
        Xte = test[usable].to_numpy(dtype=float)
        ytr = train[target].to_numpy(dtype=float)
        yte = test[target].to_numpy(dtype=float)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        if model == "gbm":
            # depth and iteration count deliberately small: with tens of
            # environments a flexible model memorises the training families
            est = HistGradientBoostingRegressor(
                max_depth=2, max_iter=150, learning_rate=0.05,
                min_samples_leaf=3, random_state=0,
            )
        elif model == "ridge":
            est = Ridge(alpha=1.0)
        else:
            raise ValueError(f"unknown model {model}")

        est.fit(Xtr_s, ytr)
        p = est.predict(Xte_s)
        preds.extend(p.tolist())
        actuals.extend(yte.tolist())
        for a, pr in zip(yte, p):
            rows.append({"family": fam, "actual": float(a), "predicted": float(pr)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    preds_arr = np.array(preds)
    actual_arr = np.array(actuals)
    r2 = r2_score(actual_arr, preds_arr) if len(set(actual_arr)) > 1 else math.nan
    rho = out["actual"].corr(out["predicted"], method="spearman")
    out.attrs["lofo_r2"] = float(r2)
    out.attrs["lofo_spearman"] = float(rho)
    out.attrs["n_families"] = len(families)
    out.attrs["n_predictions"] = len(out)
    out.attrs["predictors_used"] = usable
    return out


def predictor_correlations(
    df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlation of each predictor with the target, overall and within family.

    Within-family correlations are the ones that matter: a predictor that
    correlates only because different families differ in difficulty is not
    capturing environment value.
    """
    rows = []
    for p in predictors:
        if p not in df.columns:
            continue
        overall = df[[p, target]].corr(method=method).iloc[0, 1]
        within = []
        for fam, sub in df.groupby("family"):
            if len(sub) >= 4 and sub[[p, target]].notna().all(axis=None):
                within.append(sub[[p, target]].corr(method=method).iloc[0, 1])
        rows.append(
            {
                "predictor": p,
                f"rho_{method}_overall": overall,
                f"rho_{method}_within_family_mean": float(np.mean(within)) if within else math.nan,
                "n_families_within": len(within),
            }
        )
    return pd.DataFrame(rows).sort_values(
        f"rho_{method}_within_family_mean", ascending=False
    )


def transfer_matrix_from_frame(
    df: pd.DataFrame,
    train_col: str = "train_env",
    eval_col: str = "eval_env",
    value_col: str = "gain",
) -> pd.DataFrame:
    """Pivot a long transfer-gain frame into a train x eval matrix."""
    return df.pivot_table(
        index=train_col, columns=eval_col, values=value_col, aggfunc="mean"
    )


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean."""
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def kish_design_effect(
    cluster_sizes: Sequence[int],
    rho: float,
) -> float:
    """Kish design effect for clustered data: deff = 1 + (m_bar - 1) * rho.

    Use to convert a clustered sample size into an effective independent n, and
    report that in the paper — many RL papers silently ignore it.
    """
    m = np.asarray(cluster_sizes, dtype=float)
    if m.size == 0:
        return math.nan
    m_bar = m.mean()
    return float(1.0 + (m_bar - 1.0) * rho)


def effective_n(n: int, cluster_sizes: Sequence[int], rho: float) -> float:
    """Effective independent sample size after clustering."""
    deff = kish_design_effect(cluster_sizes, rho)
    if not np.isfinite(deff) or deff <= 0:
        return float(n)
    return float(n / deff)
