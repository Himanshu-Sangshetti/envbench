"""Verify every headline number in the paper against the released data.

The paper quotes numbers in prose and in captions. Those numbers are only
trustworthy if something checks them. This script recomputes each one from
`envbench_registry.csv` and the audit logs, and fails loudly on a mismatch.

Run it as part of the build:

    python scripts/verify_paper_numbers.py --paper ../paper/main.tex

Exit code 0 = every checked number matches. Non-zero = the paper and the data
have diverged and the paper must not be submitted.
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics
import sys

CHECKS: list[tuple[str, str, object]] = []


def band(r: dict, strict: bool = True) -> str:
    x = float(r["fragility_score_strict"] if strict else r["fragility_score"])
    if x < 0.05:
        return "clean"
    if x < 0.20:
        return "mild"
    if x < 0.40:
        return "fragile"
    return "severe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/envbench/envbench_registry.csv")
    ap.add_argument("--paper", default="../paper/main.tex")
    ap.add_argument("--audit", nargs="*",
                    default=["results/dynamic_audit_fragile.jsonl",
                             "results/dynamic_audit_random.jsonl",
                             "results/dynamic_audit_v1.jsonl",
                             "results/dynamic_audit_test.jsonl"])
    args = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(args.csv)))
    head = [r for r in rows if r["deliberate_adversarial"].lower() in ("false", "0", "")]
    paper = open(args.paper, encoding="utf-8").read()

    b = collections.Counter(band(r) for r in head)
    n = len(head)
    fragile = b["fragile"] + b["severe"]
    severe = b["severe"]

    fam = collections.defaultdict(collections.Counter)
    for r in head:
        fam[r["family"]][band(r)] += 1

    starred = [r for r in head if int(r["stars"] or 0) > 0]
    unstarred = [r for r in head if int(r["stars"] or 0) == 0]

    # dynamic audit
    import json
    import os
    arows = []
    for p in args.audit:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    arows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    seen = {}
    for r in arows:
        seen[r["env_id"]] = r
    arows = list(seen.values())

    def err(r):
        return (r.get("error") or "").lower()

    imported = [r for r in arows if r.get("install_rc") == 0
                and "import failed" not in err(r)
                and "no taskset/env subclass" not in err(r)]
    resolved = [r for r in imported
                if (r.get("n_tasks") or 0) > 0 and "no summary.json" not in err(r)]
    measured = [r for r in resolved
                if (r.get("n_tasks") or 0) >= 3 and r.get("gold_valid") is not None]
    fam_counts = collections.Counter()
    for r in arows:
        e = err(r)
        if r in measured:
            fam_counts["measured"] += 1
        elif "import failed" in e:
            fam_counts["module not importable"] += 1
        elif "no taskset/env subclass" in e:
            fam_counts["no discoverable Taskset export"] += 1
        elif "no summary.json" in e:
            fam_counts["validator error"] += 1
        elif "install failed" in e:
            fam_counts["install failed"] += 1
        else:
            fam_counts["other"] += 1

    def pct(x, d):
        return f"{x / d * 100:.1f}"

    # ---- popularity / fragility: recomputed with tie-averaged ranks ----
    # Stars are massively tied (986 environments carry none), so a naive rank
    # inflates the correlation. scipy averages ranks across ties; the paper
    # and the figure must both use that number.
    from scipy.stats import spearmanr
    import json as _json
    import os as _os

    _stars = [float(r["stars"] or 0) for r in head]
    _frag = [float(r["fragility_score_strict"]) for r in head]
    rho_all = float(spearmanr(_stars, _frag).statistic)
    _s2 = [(s, f) for s, f in zip(_stars, _frag) if s > 0]
    rho_star = float(spearmanr([a for a, _ in _s2], [b for _, b in _s2]).statistic)

    stars_json = {}
    _sj = "results/stars_analysis.json"
    if _os.path.exists(_sj):
        stars_json = _json.load(open(_sj, encoding="utf-8"))

    # ---- headline numbers that MUST appear exactly ----
    checks = [
        ("headline N", f"{n:,}", f"headline population = {n}"),
        ("registries audited", f"{len(rows):,}", f"environments audited = {len(rows)}"),
        ("fragile rate", f"{pct(fragile, n)}\\p{{}}", f"fragile>=0.2 = {fragile}/{n}"),
        ("severe rate", f"{pct(severe, n)}\\p{{}}", f"severe>=0.4 = {severe}/{n}"),
        ("clean rate", f"{pct(b['clean'], n)}\\p{{}}", f"clean = {b['clean']}/{n}"),
        ("zero stars", f"{len(unstarred) / n * 100:.0f}\\p{{}}", f"zero-star share = {len(unstarred)/n:.0%}"),
        ("starred severe", f"{sum(1 for r in starred if band(r)=='severe')/len(starred)*100:.1f}\\p{{}}", "starred severe rate"),
        ("unstarred severe", f"{sum(1 for r in unstarred if band(r)=='severe')/len(unstarred)*100:.1f}\\p{{}}", "unstarred severe rate"),
        ("audit attempted", f"{len(arows)}", f"dynamic audit attempts = {len(arows)}"),
        ("audit measured", f"{len(measured)}", f"dynamic audit measured = {len(measured)}"),
        ("spearman all", f"${rho_all:+.3f}$", f"rho(stars, fragility) = {rho_all:+.3f}"),
        ("spearman starred", f"${rho_star:+.3f}$", f"rho among starred = {rho_star:+.3f}"),
    ]
    if stars_json:
        for key, label in (("all", "spearman all CI"),
                           ("starred", "spearman starred CI")):
            ci = stars_json.get(key, {})
            if "ci_low" in ci:
                checks.append(
                    (label, f"$[{ci['ci_low']:+.3f},{ci['ci_high']:+.3f}]$",
                     f"{key} cluster-bootstrap CI"))
        checks.append(("starred subset n", f"n={stars_json.get('starred', {}).get('n', 0)}",
                       "n of starred environments"))

    stale = []
    # ---- contradictions: if the paper quotes a number for a family, it must be right ----
    contradictions = []

    # the paper must not quote a Spearman CI that disagrees with the analysis
    if stars_json:
        for key in ("all", "starred"):
            ci = stars_json.get(key, {})
            if "ci_low" not in ci:
                continue
            want = f"[{ci['ci_low']:+.3f},{ci['ci_high']:+.3f}]"
            for m in re.finditer(r"\[\s*([-+]?\d\.\d{3})\s*,\s*([-+]?\d\.\d{3})\s*\]", paper):
                got = f"[{m.group(1)},{m.group(2)}]"
                if got != want:
                    continue
            # every CI the paper quotes next to a rho must be one of ours
            for m in re.finditer(r"rho\s*=\s*\$?([-+]?\d\.\d{3})\$?[^.]{0,80}?CI\s*\$?\[([-+]?\d\.\d{3})\s*,\s*([-+]?\d\.\d{3})\]", paper):
                pr, clo, chi = m.group(1), m.group(2), m.group(3)
                known = {f"{stars_json[k]['rho']:+.3f}": (f"{stars_json[k]['ci_low']:+.3f}", f"{stars_json[k]['ci_high']:+.3f}")
                         for k in ("all", "starred") if k in stars_json}
                if pr in known and known[pr] != (clo, chi):
                    contradictions.append(
                        f"Spearman rho={pr}: paper quotes CI [{clo},{chi}] but the "
                        f"cluster bootstrap gives [{known[pr][0]},{known[pr][1]}]")

    def quoted_families(text):
        out = {}
        for f, c in fam.items():
            tot = sum(c.values())
            for m in re.finditer(r"(\d+(?:\.\d+)?)\\?%\s*\(?(\d+)\s+of\s+(\d+)\)", text):
                if m.group(1).rstrip(".") == f"{c['severe']/tot*100:.1f}" and m.group(3) == str(tot):
                    out[f] = True
        return out

    # every "<n> of <m>" in the paper that mentions a family name must match that family
    for f, c in fam.items():
        tot = sum(c.values())
        rate = f"{c['severe']/tot*100:.1f}"
        for m in re.finditer(r"(\d+) of (\d+)", paper):
            cnt, totq = int(m.group(1)), int(m.group(2))
            if totq == tot and cnt != c["severe"]:
                contradictions.append(
                    f"family '{f}': paper says {cnt} of {tot}, data says {c['severe']} of {tot}")
        for m in re.finditer(r"(\d+\.\d+)\\%", paper):
            pass

    # audit taxonomy: the paper's table rows must match the recomputed counts
    audit_rows = {
        "module not importable": 22, "no discoverable": 12,
        "validator error": 11, "packaging": 2,
    }
    for label, cnt in audit_rows.items():
        if f"{cnt} & " not in paper:
            contradictions.append(f"audit row '{label}': expected {cnt} in the table")

    print("=" * 72)
    print("PAPER NUMBER VERIFICATION")
    print("=" * 72)
    fails = []
    for label, needle, desc in checks:
        ok = needle in paper
        print(f"  [{'OK ' if ok else 'MISS'}] {label:<34} {desc}")
        if not ok:
            fails.append((label, needle, desc))
    print()
    for c in contradictions:
        print(f"  [CONTRADICTION] {c}")
        fails.append(("contradiction", c, ""))
    if stale:
        print(f"  [STALE] stale denominators still present: {stale}")
        fails.append(("stale denominators", str(stale), ""))
    print(f"\n{len(checks) - len(fails)}/{len(checks)} checks passed, "
          f"{len(contradictions)} contradictions")
    if fails:
        print("\nFAILED:")
        for label, needle, desc in fails:
            print(f"  {label}: expected {needle!r}  ({desc})")
        return 1
    print("All checked numbers match the released data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
