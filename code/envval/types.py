"""Core data types for envval.

The library models an RL training environment as three things we can measure
*before* any training happens:

    1. its verifier        -> verifier informativeness
    2. its learning signal -> learning-signal diversity
    3. its task geometry   -> transfer radius

Everything here is deliberately plain: no framework coupling, no torch import at
module scope, so the statistics can be computed on a laptop while training runs
on a rented GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable, Sequence


class TaskFamily(str, Enum):
    """Task families used for leave-one-family-out evaluation.

    The family is the unit of generalisation: the practical question is "should I
    build environments like *this kind*?", never "is environment #17 good?".
    """

    MATH = "math"
    CODE = "code"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    WEB = "web"
    GAMES = "games"
    SCIENCE = "science"
    INSTRUCTION = "instruction"
    OTHER = "other"


@dataclass(frozen=True)
class Rollout:
    """A single sampled trajectory from an environment."""

    prompt: str
    completion: str
    reward: float
    passed: bool
    info: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    family: TaskFamily = TaskFamily.OTHER
    seed: int | None = None


@dataclass(frozen=True)
class EnvironmentSpec:
    """Static description of an environment, sufficient to compute statistics."""

    env_id: str
    family: TaskFamily
    source: str
    source_url: str
    license: str = "unknown"
    n_tasks: int | None = None
    verifier_kind: str = "unknown"  # rule | program | llm_judge | unknown
    subtasks: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class VerifierAudit:
    """Result of auditing one environment's verifier.

    This is the Stage 0 artifact: the fraction of environments that turn out to
    be unusable, and how badly their verifiers fail, is itself a publishable
    finding about the field's public registries.
    """

    env_id: str
    n_samples: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    false_positives: int = 0  # incorrect solution accepted
    false_negatives: int = 0  # correct solution rejected
    near_miss_accepted: int = 0  # semantically-wrong solution accepted
    neutral_perturbation_failed: int = 0  # correct solution broken by a no-op edit
    neutral_perturbation_trials: int = 0
    notes: str = ""

    @property
    def false_positive_rate(self) -> float:
        if self.n_incorrect == 0:
            return math.nan
        return self.false_positives / self.n_incorrect

    @property
    def false_negative_rate(self) -> float:
        if self.n_correct == 0:
            return math.nan
        return self.false_negatives / self.n_correct

    @property
    def brittleness(self) -> float:
        """Fraction of *correct* solutions that a semantics-preserving edit breaks.

        High brittleness means the verifier is testing string formatting rather
        than the thing it claims to test. This is the cheapest high-signal audit
        we have.
        """
        if self.neutral_perturbation_trials == 0:
            return math.nan
        return self.neutral_perturbation_failed / self.neutral_perturbation_trials

    @property
    def usable(self) -> bool:
        """A verifier must accept correct solutions and reject incorrect ones."""
        return (
            not math.isnan(self.false_negative_rate)
            and not math.isnan(self.false_positive_rate)
            and self.false_negative_rate <= 0.15
            and self.false_positive_rate <= 0.15
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["false_positive_rate"] = self.false_positive_rate
        d["false_negative_rate"] = self.false_negative_rate
        d["brittleness"] = self.brittleness
        d["usable"] = self.usable
        return d


@dataclass
class EnvStats:
    """All pre-training statistics for one environment.

    Keys are the predictor IDs used in the paper (A1..A6, B1..B6, C1..C4) so the
    table in the paper and this dict stay in sync.
    """

    env_id: str
    family: TaskFamily
    n_rollouts: int = 0

    # --- Group A: verifier informativeness ---
    A1_verifier_false_positive_rate: float = math.nan
    A2_verifier_false_negative_rate: float = math.nan
    A3_verifier_brittleness: float = math.nan
    A4_reward_granularity: float = math.nan  # distinct reward values / n
    A5_verifier_complexity: float = math.nan  # proxy, e.g. AST nodes
    A6_near_miss_acceptance: float = math.nan

    # --- Group B: learning-signal diversity ---
    B1_strategy_diversity: float = math.nan  # pairwise disagreement rate
    B2_pass_at_k_gap: float = math.nan
    B3_null_model_gain: float = math.nan
    B4_outcome_entropy: float = math.nan  # BASELINE (RAGEN-2) - expected to be anti-predictive
    B5_length_variance: float = math.nan
    B6_partial_credit_density: float = math.nan

    # --- Group C: transfer radius ---
    C1_embedding_spread: float = math.nan
    C2_skill_taxonomy_coverage: float = math.nan
    C3_subtask_breadth: float = math.nan
    C4_verifier_discriminability: float = math.nan

    # --- naive baselines the predictor must beat ---
    base_difficulty: float = math.nan  # base-model pass rate
    n_tasks: float = math.nan
    verifiability_rate: float = math.nan

    audit: VerifierAudit | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    PREDICTOR_IDS = (
        "A1_verifier_false_positive_rate",
        "A2_verifier_false_negative_rate",
        "A3_verifier_brittleness",
        "A4_reward_granularity",
        "A5_verifier_complexity",
        "A6_near_miss_acceptance",
        "B1_strategy_diversity",
        "B2_pass_at_k_gap",
        "B3_null_model_gain",
        "B4_outcome_entropy",
        "B5_length_variance",
        "B6_partial_credit_density",
        "C1_embedding_spread",
        "C2_skill_taxonomy_coverage",
        "C3_subtask_breadth",
        "C4_verifier_discriminability",
        "base_difficulty",
        "n_tasks",
        "verifiability_rate",
    )

    def to_row(self) -> dict[str, Any]:
        row = {k: getattr(self, k) for k in self.PREDICTOR_IDS}
        row["env_id"] = self.env_id
        row["family"] = self.family.value
        row["n_rollouts"] = self.n_rollouts
        if self.audit is not None:
            row["audit_usable"] = self.audit.usable
            row["audit_notes"] = self.audit.notes
        return row

    def to_json(self) -> str:
        return json.dumps(self.to_row(), indent=2, default=str)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
