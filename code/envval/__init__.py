"""envval — pre-training statistics for RL environment quality.

Three groups, matching the paper's decomposition:

    A. verifier informativeness   -> what the reward function actually measures
    B. learning-signal diversity  -> what the gradient can exploit
    C. transfer radius            -> how far the learned capability travels

The public entry point is `compute_env_stats`.
"""

from __future__ import annotations

from .types import (
    EnvStats,
    EnvironmentSpec,
    Rollout,
    TaskFamily,
    VerifierAudit,
    read_jsonl,
    write_jsonl,
)
from .audit import (
    audit_verifier,
    known_bad_controls,
    near_miss_mutations,
    neutral_perturbations,
    verifier_complexity,
)
from .compute import compute_env_stats, compute_all
from .stats.signal import (
    difficulty,
    length_variance,
    null_model_gain,
    outcome_entropy,
    partial_credit_density,
    pass_at_k,
    pass_at_k_gap,
    reward_granularity,
    strategy_diversity,
)
from .stats.radius import (
    effective_rank,
    embedding_spread,
    skill_taxonomy_coverage,
    subtask_breadth,
    transfer_decay,
    transfer_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "EnvStats",
    "EnvironmentSpec",
    "Rollout",
    "TaskFamily",
    "VerifierAudit",
    "read_jsonl",
    "write_jsonl",
    "audit_verifier",
    "known_bad_controls",
    "near_miss_mutations",
    "neutral_perturbations",
    "verifier_complexity",
    "compute_env_stats",
    "compute_all",
    "difficulty",
    "length_variance",
    "null_model_gain",
    "outcome_entropy",
    "partial_credit_density",
    "pass_at_k",
    "pass_at_k_gap",
    "reward_granularity",
    "strategy_diversity",
    "effective_rank",
    "embedding_spread",
    "skill_taxonomy_coverage",
    "subtask_breadth",
    "transfer_decay",
    "transfer_matrix",
]
