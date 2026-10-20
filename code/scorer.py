"""The fragility scorer, re-implemented from the released artifact.

Why this module exists
----------------------
The manuscript described its *strict* score as "the five strongest modes" in
the tables and as "nine modes minus the two weak modes" in the definition.
Neither was true: the strict score is computed over **four** modes. This
module is the single source of truth, and it is checked against the released
`fragility_score_strict` column on every row, so the description and the
implementation cannot drift apart again.

The two scores
--------------
full    all nine detected modes
strict  the four *answer-matching* modes -- substring_match,
        startswith_endswith, set_membership, exact_string_eq -- i.e. the
        modes by which a verifier can be satisfied without the task being
        solved.

Severity
--------
presence-graded for every mode except regex_num_extract, whose severity is
min(count/4, 1). The legacy count-graded variant is retained for the
sensitivity analysis and for the before/after recalibration table.
"""

from __future__ import annotations

import collections
import csv

WEIGHTS = {
    "substring_match": 0.34,
    "startswith_endswith": 0.26,
    "set_membership": 0.26,
    "exact_string_eq": 0.30,
    "float_eq_no_tol": 0.18,
    "regex_num_extract": 0.14,
    "hardcoded_answers": 0.10,
    "judge_no_reference": 0.28,
    "no_execution": 0.14,
}

# The four modes that make a verifier satisfiable without solving the task.
STRICT_MODES = ["substring_match", "startswith_endswith", "set_membership",
                "exact_string_eq"]
FULL_MODES = list(WEIGHTS)

# Modes that do not by themselves let a wrong answer through, and which
# therefore separate the strict score from the full score.
WEAK_MODES = ["float_eq_no_tol", "regex_num_extract", "hardcoded_answers",
               "judge_no_reference", "no_execution"]

DISPLAY_TO_KEY = {
    "substring_match": "substring_match",
    "startswith/endswith": "startswith_endswith",
    "set_membership": "set_membership",
    "unnormalised string equality": "exact_string_eq",
    "float equality without tolerance": "float_eq_no_tol",
    "regex numeric extraction": "regex_num_extract",
    "hardcoded answer literals": "hardcoded_answers",
    "LLM judge without reference anchoring": "judge_no_reference",
    "test suite without executing candidate": "no_execution",
}

# Legacy (pre-recalibration) count saturation points.
COUNT_SAT = {"substring_match": 4, "startswith_endswith": 4, "set_membership": 4,
              "exact_string_eq": 5, "float_eq_no_tol": 3, "regex_num_extract": 4}


def parse_modes(cell: str) -> list[str]:
    out = []
    for m in (cell or "").split("|"):
        m = m.strip()
        if not m:
            continue
        key = DISPLAY_TO_KEY.get(m.split(" x")[0])
        if key is None:
            raise SystemExit(f"unknown mode name in artifact: {m!r}")
        out.append(key)
    return out


def parse_counts(cell: str) -> dict[str, int]:
    out: dict[str, int] = collections.defaultdict(int)
    for m in (cell or "").split("|"):
        m = m.strip()
        if not m:
            continue
        parts = m.split(" x")
        key = DISPLAY_TO_KEY.get(parts[0])
        if key is None:
            raise SystemExit(f"unknown mode name in artifact: {m!r}")
        out[key] += int(parts[1]) if len(parts) > 1 else 1
    return dict(out)


def severities(modes, counts, grading: str = "presence"):
    sev = {}
    for m in modes:
        c = counts.get(m, 0)
        if c == 0:
            sev[m] = 0.0
        elif grading == "presence":
            sev[m] = min(c / 4.0, 1.0) if m == "regex_num_extract" else 1.0
        else:
            if m in ("judge_no_reference", "no_execution", "hardcoded_answers"):
                sev[m] = 1.0
            elif m in COUNT_SAT:
                sev[m] = min(c / COUNT_SAT[m], 1.0)
            else:
                sev[m] = 1.0
    return sev


def noisy_or(modes, weights, sev):
    s = 1.0
    for m in modes:
        w, v = weights.get(m, 0.0), sev.get(m, 0.0)
        if v > 0:
            s *= 1.0 - w * v
    return 1.0 - s


def score_row(row, modes=STRICT_MODES, weights=WEIGHTS, grading="presence"):
    ms = parse_modes(row["defect_modes"])
    ct = parse_counts(row["defect_modes"])
    sev = severities(ms, ct, grading)
    return noisy_or(modes, weights, sev), ms, sev


def load(csv_path: str) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    return [r for r in rows
            if (r["deliberate_adversarial"] or "").lower() in ("false", "0", "")]


def self_check(rows, tol: float = 0.005) -> float:
    """Re-score every row and confirm it matches the released column."""
    worst = 0.0
    for r in rows:
        s, _, _ = score_row(r)
        worst = max(worst, abs(s - float(r["fragility_score_strict"])))
    if worst > tol:
        raise SystemExit(
            f"re-scoring does not reproduce the released strict score (max diff {worst:.4f})")
    return worst


if __name__ == "__main__":
    rows = load("results/envbench/envbench_registry.csv")
    w = self_check(rows)
    print(f"strict score reproduces the released column on {len(rows)} rows (max diff {w:.5f})")
