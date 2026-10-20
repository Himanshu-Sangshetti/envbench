"""Learning-signal diversity statistics (Group B).

The claim these test: the diversity that matters is the diversity of the
*learning signal* an environment induces, not the diversity of its task list.

All functions are pure and deterministic given their inputs, so they can be
unit-tested and re-run by anyone.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Sequence

import numpy as np

from ..types import Rollout

__all__ = [
    "pass_at_k",
    "outcome_entropy",
    "strategy_diversity",
    "pass_at_k_gap",
    "length_variance",
    "partial_credit_density",
    "reward_granularity",
    "null_model_gain",
    "difficulty",
]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021, "Evaluating LLMs Trained on Code").

    Uses the combinatorial form rather than the naive 1 - (1-p)^k, which is
    badly biased at small n and therefore at small n the difference matters.

    n: number of samples drawn
    c: number of correct samples
    k: number of draws considered
    """
    if n == 0 or k <= 0:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    k = min(k, n)
    if k == 1:
        return c / n
    if n - c < k:
        # C(n-c, k) == 0 -> every k-subset contains a correct sample
        return 1.0
    # 1 - C(n-c, k) / C(n, k), computed in log space with expm1 for precision
    log_ratio = (math.lgamma(n - c + 1) - math.lgamma(n - c - k + 1)) - (
        math.lgamma(n + 1) - math.lgamma(n - k + 1)
    )
    return -math.expm1(log_ratio)


def outcome_entropy(rollouts: Sequence[Rollout]) -> float:
    """Binary entropy of the pass/fail outcome distribution.

    This is the RAGEN-2 metric, included deliberately as a **baseline expected to
    be anti-predictive** of transfer gain. Reported in every results table.
    """
    if not rollouts:
        return math.nan
    p = sum(r.passed for r in rollouts) / len(rollouts)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _normalised_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two completions.

    Uses token-set Jaccard, which is robust to whitespace and formatting and
    therefore does not mistake a formatting difference for a strategy difference.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


def strategy_diversity(rollouts: Sequence[Rollout], threshold: float = 0.8) -> float:
    """Normalised pairwise disagreement rate among sampled rollouts.

    A pair counts as *different* when their token-set similarity falls below
    `threshold`. The result is the fraction of disagreeing pairs, so 0.0 means
    every rollout took the same path and 1.0 means every pair differed.

    This measures diversity of *behaviour*, which is the quantity a policy
    gradient can actually exploit. Task-list diversity, by contrast, can be high
    while every task is solved by the same single strategy.
    """
    n = len(rollouts)
    if n < 2:
        return math.nan
    completions = [r.completion for r in rollouts]
    disagree = 0
    total = 0
    for a, b in combinations(range(n), 2):
        total += 1
        if _normalised_similarity(completions[a], completions[b]) < threshold:
            disagree += 1
    return disagree / total


def pass_at_k_gap(rollouts: Sequence[Rollout], k: int = 8) -> float:
    """pass@k minus pass@1: the diversity of *correct* solutions.

    High gap means the environment admits many valid strategies; low gap means
    there is one route, so the gradient signal is effectively one-dimensional.
    """
    n = len(rollouts)
    if n == 0:
        return math.nan
    c = sum(r.passed for r in rollouts)
    return pass_at_k(n, c, k) - pass_at_k(n, c, 1)


def length_variance(rollouts: Sequence[Rollout]) -> float:
    """Coefficient of variation of completion length.

    Normalised by the mean so it is comparable across environments with very
    different output lengths.
    """
    lens = np.array([len(r.completion) for r in rollouts], dtype=float)
    if lens.size == 0 or lens.mean() == 0:
        return math.nan
    return float(lens.std(ddof=1) / lens.mean()) if lens.size > 1 else 0.0


def partial_credit_density(rollouts: Sequence[Rollout]) -> float:
    """Fraction of rollouts receiving a reward strictly between 0 and 1.

    Dense (partial) reward gives a per-step learning signal; sparse binary
    reward does not. A cheap structural property of the environment that a
    policy gradient can use.
    """
    if not rollouts:
        return math.nan
    partial = sum(1 for r in rollouts if 0.0 < r.reward < 1.0)
    return partial / len(rollouts)


def reward_granularity(rollouts: Sequence[Rollout]) -> float:
    """Number of distinct reward values observed, normalised by rollout count.

    1.0 means every rollout got a different reward (fully dense); ~0.0 means a
    single binary reward.
    """
    if not rollouts:
        return math.nan
    distinct = len({round(r.reward, 6) for r in rollouts})
    return distinct / len(rollouts)


def difficulty(rollouts: Sequence[Rollout]) -> float:
    """Base-policy pass rate: the field's default proxy for environment quality.

    Reported as the naive baseline that the thesis argues against.
    """
    if not rollouts:
        return math.nan
    return sum(r.passed for r in rollouts) / len(rollouts)


def null_model_gain(
    env_rollouts: Sequence[Rollout],
    null_rollouts: Sequence[Rollout],
) -> float:
    """Learning signal of an environment minus that of a trivial solver.

    `env_rollouts` are sampled from the environment under a weak-but-nonzero
    policy; `null_rollouts` are sampled from a *trivial* solver on the same
    environment (e.g. a format-only heuristic, or a much smaller model).

    The difference isolates what the environment actually *teaches* from what a
    weak agent already knew. An environment with high difficulty but low
    null-model gain is hard for reasons that carry no learning signal.

    Returns nan if either side is empty.
    """
    if not env_rollouts or not null_rollouts:
        return math.nan
    p_env = sum(r.passed for r in env_rollouts) / len(env_rollouts)
    p_null = sum(r.passed for r in null_rollouts) / len(null_rollouts)
    return p_env - p_null


def solution_diversity_matrix(rollouts: Sequence[Rollout]) -> np.ndarray:
    """Full pairwise similarity matrix — used for hierarchical clustering of
    strategies and for visualising a strategy space."""
    n = len(rollouts)
    m = np.eye(n)
    for i, j in combinations(range(n), 2):
        s = _normalised_similarity(rollouts[i].completion, rollouts[j].completion)
        m[i, j] = m[j, i] = s
    return m


def reward_distribution(rollouts: Sequence[Rollout]) -> Counter:
    """Raw reward histogram, for inspecting reward shaping."""
    return Counter(round(r.reward, 6) for r in rollouts)
