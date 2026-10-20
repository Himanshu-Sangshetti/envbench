"""Verifier auditing (Group A + Stage 0).

The central claim of this module: a training environment is only as good as its
verifier, and the verifiers shipped in public RL registries are largely
unaudited. Auditing them is cheap, is exactly the kind of work no lab has an
incentive to do, and produces a finding in its own right.

Three probe families:

* **neutral perturbations** — semantics-preserving edits (reformatting,
  whitespace, renaming a local variable). A verifier that fails these is
  testing string shape rather than the task.
* **near-miss constructions** — deliberately wrong but plausible solutions.
  A verifier that accepts these is not measuring the task.
* **known-bad controls** — obviously wrong solutions (empty, truncated,
  copy-of-prompt). A verifier that accepts *these* is broken outright.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Callable, Sequence

from .types import Rollout, VerifierAudit

__all__ = [
    "neutral_perturbations",
    "known_bad_controls",
    "near_miss_mutations",
    "audit_verifier",
    "verifier_complexity",
]

Verdict = Callable[[str], bool]


# --------------------------------------------------------------------------
# Probe families
# --------------------------------------------------------------------------

_FORMAT_ONLY = [
    ("trailing_whitespace", lambda s: s + "   \n"),
    ("leading_whitespace", lambda s: "   \n" + s),
    ("blank_lines", lambda s: s.replace("\n", "\n\n")),
    ("crlf", lambda s: s.replace("\n", "\r\n")),
    ("trailing_newline", lambda s: s.rstrip("\n") + "\n"),
]


def _rename_first_identifier(source: str, new_name: str) -> str | None:
    """Rename a local identifier. Semantics-preserving for code.

    Prefers renaming a function *argument* over renaming the function itself,
    because renaming the public function changes its API and would not be a
    semantics-preserving edit.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    candidates: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            candidates.append(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            candidates.append(node.id)
    if not candidates:
        return None
    for target in candidates:
        pattern = re.compile(rf"\b{re.escape(target)}\b")
        if pattern.search(source):
            return pattern.sub(new_name, source)
    return None


def neutral_perturbations(solution: str, kind: str = "code") -> list[tuple[str, str]]:
    """Semantics-preserving variants of a correct solution.

    Returns a list of (probe_name, mutated_solution). Every entry must remain
    correct; a verifier that rejects one is brittle.
    """
    probes: list[tuple[str, str]] = []
    for name, fn in _FORMAT_ONLY:
        try:
            probes.append((name, fn(solution)))
        except Exception:  # pragma: no cover - defensive
            continue
    if kind == "code":
        renamed = _rename_first_identifier(solution, "zz_renamed_tmp")
        if renamed is not None:
            probes.append(("rename_identifier", renamed))
    return probes


def known_bad_controls(solution: str, kind: str = "code") -> list[tuple[str, str]]:
    """Solutions that are unambiguously wrong. A verifier accepting these is broken."""
    controls = [
        ("empty", ""),
        ("whitespace_only", "   \n\t  \n"),
        ("truncated_half", solution[: max(1, len(solution) // 2)]),
        ("single_char", "x"),
    ]
    if kind == "code":
        controls.append(("comment_only", "# nothing here\n"))
    return controls


_MATH_MUTATIONS = [
    ("off_by_one", lambda s: re.sub(r"-?\d+", lambda m: str(int(m.group()) + 1), s, count=1)),
    ("sign_flip", lambda s: re.sub(r"-?\d+", lambda m: str(-int(m.group())), s, count=1)),
    ("truncate_expression", lambda s: s[: max(1, int(len(s) * 0.7))]),
]


def near_miss_mutations(solution: str, kind: str = "code") -> list[tuple[str, str]]:
    """Plausible-but-wrong solutions.

    These are the hardest probes: the verifier must reject a solution that looks
    like the right shape but is semantically wrong.
    """
    if kind == "code":
        probes = []
        for name, fn in _MATH_MUTATIONS:
            try:
                probes.append((name, fn(solution)))
            except Exception:  # pragma: no cover
                continue
        # strip the final return statement of the last function definition
        try:
            tree = ast.parse(solution)
            fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if fns:
                fn = fns[-1]
                if fn.body and isinstance(fn.body[-1], ast.Return):
                    trimmed = ast.unparse(
                        ast.Module(body=fn.body[:-1], type_ignores=[])
                    )
                    probes.append(("drop_return", trimmed))
        except (SyntaxError, ValueError):
            pass
        return probes
    return []


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------


def audit_verifier(
    env_id: str,
    correct_solutions: Sequence[str],
    verdict: Verdict,
    kind: str = "code",
    near_miss_source: Sequence[str] | None = None,
) -> VerifierAudit:
    """Audit one environment's verifier.

    correct_solutions: solutions known to be correct (from a reference solver
    or human-verified set).
    verdict: callable that returns True iff the given solution passes the
    environment's verifier.
    near_miss_source: solutions to mutate into near misses; defaults to
    correct_solutions.

    The audit reports false negatives (correct solutions rejected), false
    positives (incorrect solutions accepted), near-miss acceptance, and
    brittleness (correct solutions broken by a no-op edit).
    """
    a = VerifierAudit(env_id=env_id)

    for sol in correct_solutions:
        a.n_correct += 1
        try:
            ok = bool(verdict(sol))
        except Exception:
            ok = False
        if not ok:
            a.false_negatives += 1

    # near-miss / known-bad solutions count toward the incorrect pool
    source = list(near_miss_source if near_miss_source is not None else correct_solutions)
    for sol in source:
        for name, mut in known_bad_controls(sol, kind=kind):
            a.n_incorrect += 1
            try:
                ok = bool(verdict(mut))
            except Exception:
                ok = False
            if ok:
                a.false_positives += 1
        for name, mut in near_miss_mutations(sol, kind=kind):
            a.n_incorrect += 1
            try:
                ok = bool(verdict(mut))
            except Exception:
                ok = False
            if ok:
                a.false_positives += 1
                a.near_miss_accepted += 1

    # brittleness: correct solutions broken by semantics-preserving edits
    for sol in correct_solutions:
        for _name, mut in neutral_perturbations(sol, kind=kind):
            a.neutral_perturbation_trials += 1
            try:
                ok = bool(verdict(mut))
            except Exception:
                ok = False
            if not ok:
                a.neutral_perturbation_failed += 1

    a.n_samples = a.n_correct + a.n_incorrect
    return a


def verifier_complexity(verifier_source: str) -> float:
    """Structural proxy for verifier strength.

    Counts AST nodes of the verifier implementation. More nodes means a more
    elaborate check, which is a weak but free proxy for verifier capacity.
    Returns nan for non-Python verifiers.
    """
    try:
        tree = ast.parse(verifier_source)
    except SyntaxError:
        return math.nan
    return float(sum(1 for _ in ast.walk(tree)))
