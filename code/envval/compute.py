"""Aggregation: turn raw rollouts + an audit into an EnvStats row."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .audit import audit_verifier, verifier_complexity
from .stats import radius as radius_stats
from .stats import signal as signal_stats
from .types import EnvStats, EnvironmentSpec, Rollout, TaskFamily, VerifierAudit

__all__ = ["compute_env_stats", "compute_all"]


def compute_env_stats(
    spec: EnvironmentSpec,
    rollouts: Sequence[Rollout],
    audit: VerifierAudit | None = None,
    null_rollouts: Sequence[Rollout] | None = None,
    embeddings: np.ndarray | None = None,
    subtasks: Sequence[str] | None = None,
    taxonomy: Sequence[str] | None = None,
    verifier_source: str | None = None,
    k: int = 8,
) -> EnvStats:
    """Compute every pre-training statistic for one environment.

    rollouts: sampled from a weak-but-nonzero base policy (no training).
    null_rollouts: sampled from a trivial solver on the same environment.
    embeddings: (n_tasks, d) task embeddings for the transfer-radius group.
    subtasks / taxonomy: optional skill metadata.
    """
    s = EnvStats(env_id=spec.env_id, family=spec.family, n_rollouts=len(rollouts))
    s.audit = audit

    # --- Group B: learning-signal diversity ---
    s.B1_strategy_diversity = signal_stats.strategy_diversity(rollouts)
    s.B2_pass_at_k_gap = signal_stats.pass_at_k_gap(rollouts, k=k)
    s.B4_outcome_entropy = signal_stats.outcome_entropy(rollouts)
    s.B5_length_variance = signal_stats.length_variance(rollouts)
    s.B6_partial_credit_density = signal_stats.partial_credit_density(rollouts)
    s.base_difficulty = signal_stats.difficulty(rollouts)
    s.n_tasks = float(spec.n_tasks) if spec.n_tasks is not None else math.nan
    s.verifiability_rate = (
        float(np.mean([r.passed for r in rollouts])) if rollouts else math.nan
    )
    if null_rollouts is not None:
        s.B3_null_model_gain = signal_stats.null_model_gain(rollouts, null_rollouts)

    # reward granularity: distinct rewards observed
    s.A4_reward_granularity = signal_stats.reward_granularity(rollouts)

    # --- Group A: verifier informativeness ---
    if audit is not None:
        s.A1_verifier_false_positive_rate = audit.false_positive_rate
        s.A2_verifier_false_negative_rate = audit.false_negative_rate
        s.A3_verifier_brittleness = audit.brittleness
        s.A6_near_miss_acceptance = (
            audit.near_miss_accepted / audit.n_incorrect if audit.n_incorrect else math.nan
        )
    if verifier_source:
        s.A5_verifier_complexity = verifier_complexity(verifier_source)

    # --- Group C: transfer radius ---
    if embeddings is not None and len(embeddings) >= 2:
        s.C1_embedding_spread = radius_stats.embedding_spread(embeddings)
        s.extra["effective_rank"] = radius_stats.effective_rank(embeddings)
    if subtasks is not None:
        s.C3_subtask_breadth = float(radius_stats.subtask_breadth(subtasks))
    if taxonomy is not None and subtasks is not None:
        s.C2_skill_taxonomy_coverage = radius_stats.skill_taxonomy_coverage(
            subtasks, taxonomy
        )

    return s


def compute_all(
    specs: Sequence[EnvironmentSpec],
    rollouts_by_env: dict[str, Sequence[Rollout]],
    audits: dict[str, VerifierAudit] | None = None,
    null_rollouts_by_env: dict[str, Sequence[Rollout]] | None = None,
    embeddings_by_env: dict[str, np.ndarray] | None = None,
) -> list[EnvStats]:
    """Compute stats for a whole registry."""
    audits = audits or {}
    nulls = null_rollouts_by_env or {}
    embs = embeddings_by_env or {}
    out = []
    for spec in specs:
        out.append(
            compute_env_stats(
                spec,
                rollouts_by_env.get(spec.env_id, []),
                audit=audits.get(spec.env_id),
                null_rollouts=nulls.get(spec.env_id),
                embeddings=embs.get(spec.env_id),
            )
        )
    return out
