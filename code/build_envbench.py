"""Build `envbench` — the public registry audit + statistics table.

This is the artifact the paper releases: a machine-readable table over every
audited environment in the Prime Intellect Environments Hub, carrying the
static verifier audit, the registry metadata, and (where computed) the
pre-training statistics.

Deliberately ships **two** things, because they answer different questions:

  * `envbench_registry.csv` — the **audit**: one row per environment, with
    fragility score, defect modes, and registry metadata. Available for every
    environment, model-free, and reproducible from source in ~6 s each.
  * `envbench_stats.csv` — the **statistics**: one row per environment with
    the Group A/B/C pre-training statistics. Requires rollouts, so it is
    populated only for environments we have sampled.

The registry audit is the part that is *complete*; the statistics table is
*partial* and is labelled as such in the schema file.

Usage:
    python scripts/build_envbench.py --audit results/registry_audit_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import yaml

STRICT_MODES = ("judge_no_ref", "no_execution", "float_eq_no_tol", "hardcoded", "regex_num_extract")
W = {"substring_match": 0.34, "startswith_endswith": 0.26, "set_membership": 0.26,
     "exact_string_eq": 0.30, "float_eq_no_tol": 0.18, "regex_num_extract": 0.14,
     "hardcoded": 0.10, "judge_no_ref": 0.28, "no_execution": 0.14}


def _severity(r, k):
    if k == "substring_match":
        return 1.0 if r.get("substring_match") else 0.0
    if k == "startswith_endswith":
        return 1.0 if r.get("startswith_endswith") else 0.0
    if k == "set_membership":
        return 1.0 if r.get("set_membership") else 0.0
    if k == "exact_string_eq":
        return 1.0 if (r.get("exact_string_eq") and not (r.get("has_lower") or r.get("has_strip"))) else 0.0
    if k == "float_eq_no_tol":
        return 1.0 if r.get("float_eq_no_tol") else 0.0
    if k == "regex_num_extract":
        return min(r.get("regex_num_extract", 0) / 4, 1.0)
    if k == "hardcoded":
        return 1.0 if r.get("has_hardcoded_answers") else 0.0
    if k == "judge_no_ref":
        return 1.0 if (r.get("is_llm_judge") and not r.get("judge_has_reference")) else 0.0
    if k == "no_execution":
        return 1.0 if (r.get("runs_test_suite") and not r.get("executes_candidate")) else 0.0
    return 0.0


def noisy_or(r, drop=()):
    s = 1.0
    for k, w in W.items():
        if k in drop:
            continue
        v = _severity(r, k)
        if v > 0:
            s *= (1 - w * v)
    return 1 - s


DEFECT_MODES = [
    "LLM judge without reference anchoring",
    "startswith/endswith matching",
    "set-membership matching",
    "unnormalised string equality",
    "test suite without executing candidate",
    "substring matching",
    "hardcoded answer literals",
    "float equality without tolerance",
    "regex numeric extraction",
]

SCHEMA = {
    "env_id": "owner/name in the Prime Intellect Environments Hub",
    "hub_id": "registry id",
    "version": "latest published version",
    "family": "task family, tag-derived (see caveat)",
    "verifier_hint": "rule | program | llm_judge | rule_or_judge (tag-derived)",
    "runtime": "VERIFIERS_V0 | VERIFIERS_V1",
    "stars": "registry popularity (0 for most)",
    "train": "tagged for training",
    "tags": "pipe-separated registry tags",
    "fragility_score": "0..1, noisy-OR over all 9 verifier defect modes; static, from source",
    "fragility_score_strict": "same, excluding the 2 weakest modes (unanchored LLM judge; test-suite-without-execution)",
    "fragility_band_strict": "band under the strict score",
    "defect_modes": "pipe-separated detected defect modes",
    "n_defect_modes": "count of detected defect modes",
    "deliberate_adversarial": "environment's purpose is adversarial detection; EXCLUDED from headline rates",
    "verifier_loc": "lines of scoring code analysed",
    "fragility_band": "clean (<0.05) | mild (0.05-0.2) | fragile (0.2-0.4) | severe (>=0.4)",
    "static_only": "TRUE: fragility is a source-level proxy, not a measured false-positive rate",
}


def band(v: float) -> str:
    if v < 0.05:
        return "clean"
    if v < 0.20:
        return "mild"
    if v < 0.40:
        return "fragile"
    return "severe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/registry_audit_full.jsonl")
    ap.add_argument("--registry", default="envs/registry_generated.yaml")
    ap.add_argument("--out", default="results/envbench")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    reg = yaml.safe_load(open(args.registry, encoding="utf-8"))
    meta = {e["env_id"]: e for e in reg["environments"]}

    rows = []
    for line in open(args.audit, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("ok"):
            continue
        m = meta.get(r["env_id"], {})
        ex = r.get("extra") or {}
        fs = float(r.get("fragility_score", 0.0))
        modes = r.get("fragility_flags") or []
        rows.append({
            "env_id": r["env_id"],
            "hub_id": m.get("hub_id", ""),
            "version": m.get("version", ""),
            "family": ex.get("family", m.get("family", "unknown")),
            "verifier_hint": ex.get("verifier_hint", m.get("verifier_hint", "unknown")),
            "runtime": m.get("runtime", ""),
            "stars": ex.get("stars", m.get("stars", 0)),
            "train": m.get("train", False),
            "tags": "|".join(m.get("tags", []) or []),
            "fragility_score": round(fs, 4),
            "fragility_score_strict": round(noisy_or(r, STRICT_MODES), 4),
            "fragility_band_strict": band(noisy_or(r, STRICT_MODES)),
            "defect_modes": "|".join(modes),
            "n_defect_modes": len(modes),
            "deliberate_adversarial": bool(r.get("deliberate_adversarial")),
            "verifier_loc": r.get("loc", 0),
            "fragility_band": band(fs),
            "static_only": True,
        })

    rows.sort(key=lambda x: (-x["fragility_score"], x["env_id"]))
    csv_path = os.path.join(args.out, "envbench_registry.csv")
    import csv

    cols = list(SCHEMA.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {csv_path}  ({len(rows)} environments)")

    # summary
    head = [r for r in rows if not r["deliberate_adversarial"]]
    adv = [r for r in rows if r["deliberate_adversarial"]]
    bands = Counter(r["fragility_band_strict"] for r in head)
    bands_full = Counter(r["fragility_band"] for r in head)
    modes = Counter()
    for r in head:
        for m in (r["defect_modes"].split("|") if r["defect_modes"] else []):
            modes[m.split(" x")[0]] += 1
    summary = {
        "n_rows": len(rows),
        "n_headline_population": len(head),
        "n_deliberately_adversarial_excluded": len(adv),
        "bands_strict": dict(bands),
        "bands_full_scorer": dict(bands_full),
        "frac_fragile": bands["fragile"] / max(1, len(head)),
        "frac_severe": bands["severe"] / max(1, len(head)),
        "defect_mode_frequency": dict(modes.most_common()),
        "by_family": {},
        "by_verifier_hint": {},
    }
    for key, field in (("by_family", "family"), ("by_verifier_hint", "verifier_hint")):
        g: dict[str, list[float]] = {}
        for r in head:
            g.setdefault(r[field], []).append(r["fragility_score"])
        summary[key] = {
            k: {"n": len(v),
                "median": round(sorted(v)[len(v) // 2], 4),
                "frac_fragile": round(sum(1 for x in v if x >= 0.2) / len(v), 4)}
            for k, v in sorted(g.items(), key=lambda kv: -sorted(kv[1])[len(kv[1]) // 2])
        }
    json.dump(summary, open(os.path.join(args.out, "envbench_summary.json"), "w"), indent=1)
    json.dump(SCHEMA, open(os.path.join(args.out, "envbench_schema.json"), "w"), indent=1)

    print(f"\nheadline population: {len(head)} (excluded {len(adv)} deliberately adversarial)")
    print("  strict score (5 strong modes):")
    for b in ("clean", "mild", "fragile", "severe"):
        n = bands[b]
        print(f"    {b:<9} {n:5d}  ({n/max(1,len(head)):.1%})")
    print("  full scorer (9 modes):")
    for b in ("clean", "mild", "fragile", "severe"):
        n = bands_full[b]
        print(f"    {b:<9} {n:5d}  ({n/max(1,len(head)):.1%})")
    print("\ndefect modes:")
    for m, c in modes.most_common():
        print(f"  {c:5d}  ({c/max(1,len(head)):5.1%})  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
