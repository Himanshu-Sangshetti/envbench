"""Transfer-radius statistics (Group C).

"Transfer radius" is a proposed coinage: how far, in task space, the capability
acquired in one environment carries to another. These functions define the
geometry and provide the estimator.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "effective_rank",
    "embedding_spread",
    "skill_taxonomy_coverage",
    "subtask_breadth",
    "transfer_decay",
    "transfer_matrix",
]


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank of a set of embeddings (Roy & Golub, 2020).

    Counts the number of directions that actually carry variance. An
    environment whose tasks all lie on one line has an effective rank near 1
    no matter how many tasks it has, because it teaches one thing.

    embeddings: (n, d) array of task embeddings.
    """
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return math.nan
    x = x - x.mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return math.nan
    if sv.sum() <= 0:
        return math.nan
    p = sv / sv.sum()
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float(np.exp(entropy))


def embedding_spread(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance of task embeddings.

    Complements effective_rank: spread measures how far apart tasks are,
    effective_rank measures how many independent directions they occupy.
    """
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return math.nan
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    xn = x / norms
    sim = xn @ xn.T
    iu = np.triu_indices(x.shape[0], k=1)
    d = 1.0 - sim[iu]
    return float(np.mean(d))


def skill_taxonomy_coverage(
    assigned: Sequence[str],
    taxonomy: Sequence[str],
) -> float:
    """Fraction of a declared skill taxonomy covered by the environment's tasks.

    `assigned` are the skill labels attached to the environment's tasks;
    `taxonomy` is the full label set the environment claims to address.
    """
    if not taxonomy:
        return math.nan
    covered = set(assigned) & set(taxonomy)
    return len(covered) / len(set(taxonomy))


def subtask_breadth(subtasks: Sequence[str]) -> int:
    """Number of distinct subtask types present in the environment."""
    return len(set(subtasks))


def transfer_matrix(
    train_envs: Sequence[str],
    eval_envs: Sequence[str],
    gains: dict[tuple[str, str], float],
) -> np.ndarray:
    """Assemble a train-env x eval-env transfer matrix.

    gains maps (train_env, eval_env) -> capability gain on eval_env.
    Missing entries are NaN, which the analysis treats as "not measured" rather
    than zero — an important distinction when the matrix is sparse.
    """
    ti = {e: i for i, e in enumerate(train_envs)}
    ei = {e: i for i, e in enumerate(eval_envs)}
    m = np.full((len(train_envs), len(eval_envs)), np.nan)
    for (t, e), v in gains.items():
        if t in ti and e in ei:
            m[ti[t], ei[e]] = v
    return m


def transfer_decay(
    distances: Sequence[float],
    gains: Sequence[float],
) -> dict[str, float]:
    """Fit gain as a decaying function of distance.

    Uses a simple exponential model  gain(d) = A * exp(-d / r)  fitted by
    least squares on log(gain) where gain > 0. `r` is the transfer radius: the
    distance at which transfer gain falls to 1/e of its peak.

    Returns the fitted amplitude A, radius r, and the coefficient of
    determination of the log-linear fit.
    """
    d = np.asarray(distances, dtype=float)
    g = np.asarray(gains, dtype=float)
    mask = np.isfinite(d) & np.isfinite(g) & (g > 0)
    if mask.sum() < 3:
        return {"amplitude": math.nan, "radius": math.nan, "r2": math.nan, "n": int(mask.sum())}
    dd, gg = d[mask], g[mask]
    slope, intercept = np.polyfit(dd, np.log(gg), 1)
    pred = slope * dd + intercept
    ss_res = float(np.sum((np.log(gg) - pred) ** 2))
    ss_tot = float(np.sum((np.log(gg) - np.log(gg).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    radius = -1.0 / slope if slope < 0 else math.inf
    return {
        "amplitude": float(math.exp(intercept)),
        "radius": float(radius),
        "r2": float(r2),
        "n": int(mask.sum()),
    }


def pairwise_transfer_radius(
    env_ids: Sequence[str],
    distances: dict[tuple[str, str], float],
    gains: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Fit one transfer curve per environment against a shared distance metric.

    For each training environment, regress its outgoing transfer gains on the
    distance to each evaluation environment. The fitted radius is that
    environment's transfer radius under a common metric, which is what makes
    radii comparable across environments.
    """
    out: dict[str, float] = {}
    for t in env_ids:
        ds, gs = [], []
        for e in env_ids:
            if e == t:
                continue
            key = (t, e)
            if key in distances and key in gains:
                ds.append(distances[key])
                gs.append(gains[key])
        fit = transfer_decay(ds, gs)
        out[t] = fit["radius"]
    return out
