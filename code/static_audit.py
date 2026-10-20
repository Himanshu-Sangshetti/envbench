"""Static verifier audit at registry scale.

Reads the *source* of an environment's verifier and scores it for fragility —
the class of defect that silently inflates rewards: exact string equality
without normalisation, substring matching, un-canonicalised numeric extraction,
float comparison without tolerance, and LLM judges with no reference anchoring.

This is the "static leakiness audit" idea from the MBPP-vs-MBPP+ comparison
(where a cheap static audit computed before training predicted the mass of
falsely-rewarded solutions at Spearman 0.80), applied to a whole registry
rather than one dataset.

It needs no GPU, no model, and no account: `prime env inspect` serves the
source directly.

Usage:
    python scripts/static_audit.py --limit 40 --out results/static_audit.jsonl
    python scripts/static_audit.py --env puiji/flawed-judge --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

PRIME = shutil.which("prime") or os.path.expanduser("~/.local/bin/prime")

# files that plausibly contain the scoring logic
VERIFIER_FILE_RE = re.compile(
    r"(verif|score|reward|grade|check|judge|answer|eval|test|reward_fn)", re.I
)
# files that are plumbing rather than scoring logic
SKIP_FILE_RE = re.compile(r"(pyproject|README|__init__|setup\.cfg|\.lock)", re.I)
# directories that are never the verifier: analysis scripts, test suites, CI
SKIP_DIR_RE = re.compile(r"(^|/)(analysis|analyses|scripts|tools|tests?|ci|docs?|assets|notebooks?|examples?)(/|$)", re.I)


@dataclass
class StaticAudit:
    env_id: str = ""
    ok: bool = False
    error: str = ""
    n_files: int = 0
    verifier_files: list[str] = field(default_factory=list)
    loc: int = 0
    # --- fragility signals ---
    exact_string_eq: int = 0
    substring_match: int = 0
    startswith_endswith: int = 0
    float_eq_no_tol: int = 0
    regex_num_extract: int = 0
    set_membership: int = 0
    # --- hygiene signals (presence) ---
    has_lower: bool = False
    has_strip: bool = False
    has_unicodedata: bool = False
    has_approx: bool = False
    has_float_cast: bool = False
    # --- structure ---
    is_llm_judge: bool = False
    judge_has_reference: bool = False
    executes_candidate: bool = False
    runs_test_suite: bool = False
    has_hardcoded_answers: bool = False
    # --- aggregate ---
    fragility_score: float = 0.0
    fragility_flags: list[str] = field(default_factory=list)
    # True when the environment's PURPOSE is adversarial detection, so a string
    # match in its verifier is the task rather than a defect
    deliberate_adversarial: bool = False
    # registry metadata joined on afterwards (family, verifier hint, stars, tags)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# CLI plumbing
# ─────────────────────────────────────────────────────────────────────────────

TREE_RE = re.compile(r"│\s*(dir|file)\s*│\s*([^│]+?)\s*│")
PLAIN_RE = re.compile(r"^(dir|file)\s+(\S+)")


def _run(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(
            [PRIME, *args], capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, ""
    except Exception as e:
        return 1, str(e)


def _parse_tree(out: str) -> tuple[list[str], list[str]]:
    """Parse `prime env inspect --plain` output into (dirs, files)."""
    dirs: list[str] = []
    files: list[str] = []
    for line in out.splitlines():
        s = line.strip()
        if not s or s.startswith("Source:") or s.startswith("Type "):
            continue
        if s.startswith("Inspect a file") or s.startswith("("):
            continue
        m = PLAIN_RE.match(s) or TREE_RE.search(line)
        if not m:
            continue
        kind, path = m.group(1), m.group(2).strip()
        if kind == "dir":
            dirs.append(path)
        elif kind == "file":
            files.append(path)
    return dirs, files


def list_files(env: str, path: str = "", _depth: int = 0, _seen: set[str] | None = None) -> list[str]:
    """Recursively list every file in an environment.

    The CLI takes `<env>@<version> <path>`; directories are walked explicitly.
    """
    if _seen is None:
        _seen = set()
    if _depth > 5:
        return []
    key = f"{env}@{path}"
    if key in _seen:
        return []
    _seen.add(key)

    args = ["env", "inspect", f"{env}@latest"]
    if path:
        args.append(path)
    rc, out = _run([*args, "--plain"], timeout=90)
    if rc != 0:
        return []
    dirs, files = _parse_tree(out)
    for d in dirs:
        files.extend(list_files(env, d, _depth + 1, _seen))
    seen: set[str] = set()
    out_files: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out_files.append(f)
    return out_files


def read_file(env: str, path: str) -> str:
    rc, out = _run(["env", "inspect", f"{env}@latest", path], timeout=90)
    if rc != 0:
        return ""
    # the CLI echoes "<env>@latest\n<path>\n\n" before the body
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == path.strip():
            return "\n".join(lines[i + 1 :])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Static analysis
# ─────────────────────────────────────────────────────────────────────────────

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
LLM_MARKERS = (
    "judge",
    "llm_judge",
    "rubric",
    "completion(",
    "chat.completions",
    "responses.create",
)


def analyse_source(src: str) -> dict:
    """AST + regex analysis of one verifier file."""
    sig = dict(
        exact_string_eq=0,
        substring_match=0,
        startswith_endswith=0,
        float_eq_no_tol=0,
        regex_num_extract=0,
        set_membership=0,
        has_lower=False,
        has_strip=False,
        has_unicodedata=False,
        has_approx=False,
        has_float_cast=False,
        is_llm_judge=False,
        judge_has_reference=False,
        executes_candidate=False,
        runs_test_suite=False,
        has_hardcoded_answers=False,
    )
    low = src.lower()
    sig["has_lower"] = ".lower()" in low or ".casefold()" in low
    sig["has_strip"] = ".strip()" in low
    sig["has_unicodedata"] = "unicodedata" in low
    sig["has_approx"] = "isclose" in low or "math.isclose" in low or "approx" in low
    sig["has_float_cast"] = "float(" in low
    sig["is_llm_judge"] = any(m in low for m in LLM_MARKERS)
    sig["judge_has_reference"] = (
        "reference" in low or "ground_truth" in low or "expected" in low
    )
    sig["executes_candidate"] = (
        "subprocess" in low
        or "exec(" in low
        or "eval(" in low
        or "run_python" in low
        or "sandbox" in low
    )
    sig["runs_test_suite"] = "pytest" in low or "unittest" in low or "test_" in low
    sig["regex_num_extract"] = len(
        re.findall(r"re\.(?:findall|search|match)\([^)]*\\d", src)
    )
    sig["has_hardcoded_answers"] = bool(
        re.search(r'["\']\s*(?:-?\d+(?:\.\d+)?)\s*["\']\s*,', src)
    )

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return sig

    for node in ast.walk(tree):
        # string equality
        if isinstance(node, ast.Compare):
            for op, cmp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    if _is_str_like(cmp) or _is_str_like(node.left):
                        sig["exact_string_eq"] += 1
                if isinstance(op, ast.In) and _is_str_like(cmp):
                    sig["substring_match"] += 1
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    if _is_float_like(cmp) or _is_float_like(node.left):
                        # equality on floats without an isclose nearby
                        if not sig["has_approx"]:
                            sig["float_eq_no_tol"] += 1
        # str.startswith / endswith
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("startswith", "endswith"):
                sig["startswith_endswith"] += 1
        # membership in a set/list of literals
        if isinstance(node, ast.Compare) and any(
            isinstance(op, ast.In) for op in node.ops
        ):
            for cmp in node.comparators:
                if isinstance(cmp, (ast.Set, ast.List, ast.Tuple)) and cmp.elts:
                    if all(isinstance(e, ast.Constant) for e in cmp.elts):
                        sig["set_membership"] += 1

    return sig


def _is_str_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("strip", "lower", "format"):
            return True
    if isinstance(node, ast.JoinedStr):
        return True
    return False


def _is_float_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "float":
            return True
    return False


ADVERSARIAL_NAME_RE = re.compile(
    r"(backdoor|goat|goblin|hack|leak|inject|jailbreak|exploit|attack|poison|"
    r"adversar|red[-_]?team|defen[cs]e|canary|trojan|trap|cheat)",
    re.I,
)


def is_deliberate_adversarial(name: str, tags: list[str] | None = None) -> bool:
    """True when the environment's *purpose* is to detect adversarial behaviour.

    These environments are supposed to look for hidden words, backdoor triggers
    and prompt injections, so a string match in their verifier is the task, not
    a defect. Mixing them into a registry-wide fragility rate is a selection
    bias, and they are analysed separately instead.
    """
    tl = {t.lower() for t in (tags or [])}
    if tl & {"reward-hacking", "reward-hack", "adversarial", "backdoor", "safety"}:
        return True
    return bool(ADVERSARIAL_NAME_RE.search(name or ""))


def score(a: StaticAudit, presence: bool = True) -> StaticAudit:
    """Fragility score in [0,1]; higher = more fragile verifier.

    Combined with a **noisy-OR** over independent defect modes:

        score = 1 - prod_i (1 - w_i * severity_i)

    where each severity_i is a count saturated at a threshold. The noisy-OR is
    used instead of a capped sum because a sum saturates: many environments
    scored a flat 1.00, destroying resolution. Noisy-OR is monotone, bounded,
    and interpretable - a single severe defect mode gives a meaningful score
    and additional modes raise it without pinning it at 1.

    Weights are ordered by how much each defect inflates rewards:
      accepting a near-miss (substring, startswith, set membership) is worse
      than being pedantic about formatting.

    `presence` selects how severity is graded:

      * ``presence=True``  (current, used everywhere): a SINGLE occurrence of a
        severe defect mode is already a full-severity signal, because one
        substring match is enough to let a near-miss pass. The severe modes are
        graded on presence; only the weaker modes (regex extraction, hardcoded
        literals) are count-graded.

      * ``presence=False`` (legacy, retained for reproducibility): every
        count-graded mode saturates at a fixed count. This is the detector as
        it stood *before* the labelled-set validation forced the recalibration.
        It is kept so the paper's reported before/after numbers can be
        regenerated rather than taken on trust.
    """
    sev = {}

    def s(count: int, sat: int) -> float:
        return min(count / sat, 1.0) if sat else 0.0

    if presence:
        sev["substring_match"] = (0.34, 1.0 if a.substring_match else 0.0)
        sev["startswith_endswith"] = (0.26, 1.0 if a.startswith_endswith else 0.0)
        sev["set_membership"] = (0.26, 1.0 if a.set_membership else 0.0)
        sev["exact_string_eq"] = (
            0.30 if (a.exact_string_eq and not (a.has_lower or a.has_strip)) else 0.0,
            1.0 if (a.exact_string_eq and not (a.has_lower or a.has_strip)) else 0.0,
        )
        sev["float_eq_no_tol"] = (0.18, 1.0 if a.float_eq_no_tol else 0.0)
    else:
        sev["substring_match"] = (0.34, s(a.substring_match, 4))
        sev["startswith_endswith"] = (0.26, s(a.startswith_endswith, 4))
        sev["set_membership"] = (0.26, s(a.set_membership, 4))
        sev["exact_string_eq"] = (
            0.30 if (a.exact_string_eq and not (a.has_lower or a.has_strip)) else 0.0,
            s(a.exact_string_eq, 5),
        )
        sev["float_eq_no_tol"] = (0.18, s(a.float_eq_no_tol, 3))
    sev["regex_num_extract"] = (0.14, s(a.regex_num_extract, 4))
    sev["hardcoded_answers"] = (0.10, 1.0 if a.has_hardcoded_answers else 0.0)
    sev["judge_no_reference"] = (
        0.28 if (a.is_llm_judge and not a.judge_has_reference) else 0.0,
        1.0 if (a.is_llm_judge and not a.judge_has_reference) else 0.0,
    )
    sev["no_execution"] = (
        0.14 if (a.runs_test_suite and not a.executes_candidate) else 0.0,
        1.0 if (a.runs_test_suite and not a.executes_candidate) else 0.0,
    )

    survive = 1.0
    for w, sv in sev.values():
        if sv > 0:
            survive *= 1.0 - w * sv
    a.fragility_score = float(round(1.0 - survive, 4))

    flags = []
    if a.substring_match:
        flags.append(f"substring_match x{a.substring_match}")
    if a.startswith_endswith:
        flags.append(f"startswith/endswith x{a.startswith_endswith}")
    if a.set_membership:
        flags.append(f"set_membership x{a.set_membership}")
    if a.exact_string_eq and not (a.has_lower or a.has_strip):
        flags.append("unnormalised string equality")
    if a.float_eq_no_tol:
        flags.append("float equality without tolerance")
    if a.regex_num_extract:
        flags.append(f"regex numeric extraction x{a.regex_num_extract}")
    if a.is_llm_judge and not a.judge_has_reference:
        flags.append("LLM judge without reference anchoring")
    if a.has_hardcoded_answers:
        flags.append("hardcoded answer literals")
    if a.runs_test_suite and not a.executes_candidate:
        flags.append("test suite without executing candidate")
    a.fragility_flags = flags
    return a


# ─────────────────────────────────────────────────────────────────────────────

def audit_env(env_id: str, verbose: bool = False) -> StaticAudit:
    a = StaticAudit(env_id=env_id)
    files = list_files(env_id)
    if not files:
        a.error = "no files returned (or env not found)"
        return a
    a.n_files = len(files)
    if verbose:
        print(f"  {env_id}: {len(files)} files -> {files[:8]}")
    pyfiles = [
        f
        for f in files
        if f.endswith(".py") and not SKIP_FILE_RE.search(f) and not SKIP_DIR_RE.search(f)
    ]
    candidates = [f for f in pyfiles if VERIFIER_FILE_RE.search(f)]
    if not candidates:
        candidates = pyfiles[:3]
    # a monolithic module is usually where the scoring lives; keep it
    a.verifier_files = candidates[:6]

    merged = dict(
        exact_string_eq=0, substring_match=0, startswith_endswith=0,
        float_eq_no_tol=0, regex_num_extract=0, set_membership=0,
    )
    for f in candidates:
        src = read_file(env_id, f)
        if not src:
            continue
        a.loc += src.count("\n") + 1
        sig = analyse_source(src)
        for k in merged:
            merged[k] += sig.get(k, 0)
        for k in (
            "has_lower", "has_strip", "has_unicodedata", "has_approx",
            "has_float_cast", "is_llm_judge", "judge_has_reference",
            "executes_candidate", "runs_test_suite", "has_hardcoded_answers",
        ):
            a.__dict__[k] = bool(a.__dict__[k] or sig.get(k, False))
    for k, v in merged.items():
        a.__dict__[k] = v
    a.ok = True
    a = score(a)
    if verbose:
        print(
            f"  {env_id}: {a.n_files} files, verifier={a.verifier_files[:2]}, "
            f"loc={a.loc}, fragility={a.fragility_score:.2f} {a.fragility_flags}"
        )
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="envs/registry_generated.yaml")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/static_audit.jsonl")
    ap.add_argument("--env", default=None, help="audit one environment by name")
    ap.add_argument("--min-priority", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.env:
        a = audit_env(args.env, verbose=True)
        print(json.dumps(a.to_dict(), indent=1))
        return 0

    import yaml

    reg = yaml.safe_load(open(args.registry, encoding="utf-8"))
    envs = reg["environments"]
    # de-duplicate by name; the hub has several versions of the same env
    seen: set[str] = set()
    pool = []
    for e in envs:
        n = e["name"]
        if n in seen:
            continue
        seen.add(n)
        if e["audit_priority"] >= args.min_priority:
            pool.append(e)
    pool.sort(key=lambda e: (-e["audit_priority"], -e["stars"], e["name"]))
    pool = pool[: args.limit]

    print(f"statically auditing {len(pool)} environments")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    ok = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, e in enumerate(pool, 1):
            a = audit_env(e["env_id"], verbose=args.verbose)
            a.env_id = e["env_id"]
            if a.ok:
                ok += 1
            fh.write(json.dumps(a.to_dict()) + "\n")
            if i % 5 == 0 or i == len(pool):
                print(f"  [{i}/{len(pool)}] ok={ok} elapsed={time.time()-t0:.0f}s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
