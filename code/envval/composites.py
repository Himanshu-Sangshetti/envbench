"""Composite component scores.

The paper's decomposition has **three components**, not twenty statistics.
This module collapses each group of statistics into one composite score with a
documented sign convention, so that:

  * the statistical model has three primary predictors plus a few baselines,
    which is what 30-50 environments can actually support;
  * collinearity between members of a group cannot blow up the fit;
  * each component becomes a single citable handle.

Members are z-scored across the environment pool (so the pool, not the
individual environment, sets the scale), then averaged. Missing members are
skipped and the denominator is the number of members actually present.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "COMPONENTS",
    "build_composites",
    "zscore_median",
]

# component -> list of (column, sign) where sign=+1 means "higher is better"
COMPONENTS: dict[str, list[tuple[str, int]]] = {
    # A: how much true signal the verifier carries
    "A_verifier_informativeness": [
        ("A1_verifier_false_positive_rate", -1),
        ("A2_verifier_false_negative_rate", -1),
        ("A3_verifier_brittleness", -1),
        ("A6_near_miss_acceptance", -1),
        ("A4_reward_granularity", +1),
    ],
    # B: how diverse the learning signal is
    "B_signal_diversity": [
        ("B1_strategy_diversity", +1),
        ("B2_pass_at_k_gap", +1),
        ("B3_null_model_gain", +1),
        ("B6_partial_credit_density", +1),
    ],
    # C: how far the learned capability travels
    "C_transfer_radius": [
        ("C1_embedding_spread", +1),
        ("C3_subtask_breadth", +1),
        ("C2_skill_taxonomy_coverage", +1),
        ("effective_rank", +1),
    ],
}

# The naive baselines the predictor must beat.
# `n_tasks` is included because RLVE's lever is environment *count/scale*: if
# simply providing more tasks beats our predictor, the paper's contribution
# evaporates. This is the baseline most likely to kill the thesis, so it is
# tested explicitly rather than left implicit.
BASELINE_PREDICTORS = ["base_difficulty", "B4_outcome_entropy", "n_tasks"]


def zscore_median(x: pd.Series) -> pd.Series:
    """Robust z-score using the median and MAD.

    Median/MAD rather than mean/SD because a single broken environment with a
    pathological verifier statistic should not rescale every other environment.
    Falls back to the standard z-score if the MAD is zero.
    """
    v = pd.to_numeric(x, errors="coerce")
    med = v.median()
    mad = (v - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        sd = v.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(v)), index=x.index)
        return (v - med) / sd
    return (v - med) / (1.4826 * mad)


def build_composites(
    df: pd.DataFrame,
    components: dict[str, list[tuple[str, int]]] | None = None,
) -> pd.DataFrame:
    """Add one composite column per component to a copy of df.

    Also records how many members were present per component, so the paper can
    report coverage rather than silently averaging over a different set of
    statistics for different environments.
    """
    comps = components or COMPONENTS
    out = df.copy()
    for name, members in comps.items():
        zs: list[pd.Series] = []
        present: list[str] = []
        for col, sign in members:
            if col not in out.columns:
                continue
            s = zscore_median(out[col])
            if s.notna().sum() < 2:
                continue
            zs.append(sign * s)
            present.append(col)
        if not zs:
            out[name] = np.nan
            out[f"{name}__n_members"] = 0
            continue
        mat = pd.concat(zs, axis=1)
        out[name] = mat.mean(axis=1, skipna=True)
        out[f"{name}__n_members"] = mat.notna().sum(axis=1)
        out[f"{name}__members"] = "|".join(present)
    return out


def composite_table(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable description of what each composite is made of."""
    rows = []
    for name, members in COMPONENTS.items():
        rows.append(
            {
                "component": name,
                "n_members": len(members),
                "members": ", ".join(
                    f"{c} ({'+' if s > 0 else '-'})" for c, s in members
                ),
            }
        )
    return pd.DataFrame(rows)
