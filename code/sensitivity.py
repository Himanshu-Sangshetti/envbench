"""Sensitivity of the headline number to the choices we made.

The critique that matters here is: *"the weights and the thresholds are hand-picked
with no justification; how much does the headline move if you change them?"*

This script answers that directly, from the released table. It re-scores every
environment under a family of scorer variants and reports the headline fragility
rate under each, so the paper can state how much of the number is a consequence
of the scoring choices and how much is not.

Variants covered:

  thresholds   the strict-score threshold (0.15 / 0.20 / 0.25 / 0.30) and the
               severe threshold
  weights      the nine severity weights, perturbed as a block (+/-25%, +/-
               50%), and replaced by uniform weights
  mode set     strict (5 modes) vs full (9 modes) vs the 3 strongest modes
  grading      presence-based severity vs the legacy count-based severity

    python scripts/sensitivity.py --out results/sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The nine weights, in the order used by the paper.
from scorer import (  # noqa: E402
    FULL_MODES,
    STRICT_MODES,
    WEIGHTS,
    WEAK_MODES,
    load,
    noisy_or,
    parse_counts,
    parse_modes,
    score_row,
    severities,
)


def score_all(rows, modes, weights, grading="presence"):
    out = []
    for r in rows:
        ms = parse_modes(r["defect_modes"])
        counts = parse_counts(r["defect_modes"])
        sev = severities(ms, counts, grading)
        out.append(noisy_or(modes, weights, sev))
    return out


def rate(scores, thr):
    return sum(1 for s in scores if s >= thr) / len(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/envbench/envbench_registry.csv")
    ap.add_argument("--out", default="results/sensitivity.json")
    args = ap.parse_args()

    rows = load(args.csv)
    n = len(rows)
    results = {"population": n, "variants": {}}

    base = score_all(rows, STRICT_MODES, WEIGHTS)

    # self-consistency: the re-scored value must equal the released column
    from scorer import self_check
    worst = self_check(rows)
    print(f"  self-consistency: max |re-scored - released| = {worst:.5f}"
          f" over {len(rows)} rows")
    results["headline"] = {
        "strict_rate_ge_0.2": rate(base, 0.2),
        "severe_rate_ge_0.4": rate(base, 0.4),
        "note": "the paper's headline: 46.3% at least one strong mode, 13.6% severe",
    }
    print(f"population {n}")
    print(f"  baseline (strict, threshold 0.20): {rate(base,0.2):.3%}   "
          f"severe (0.40): {rate(base,0.4):.3%}")

    # ---- thresholds ----
    print("\n-- threshold sensitivity (strict score) --")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        r = rate(base, thr)
        results["variants"][f"threshold_{thr:.2f}"] = r
        print(f"  >= {thr:.2f}: {r:.3%}  ({r*n:.0f} environments)")

    # ---- weights ----
    print("\n-- weight sensitivity (strict score, threshold 0.20) --")
    for name, w in (("baseline", WEIGHTS),
                    ("all_weights_x0.75", {k: v * 0.75 for k, v in WEIGHTS.items()}),
                    ("all_weights_x1.25", {k: v * 1.25 for k, v in WEIGHTS.items()}),
                    ("all_weights_x0.50", {k: v * 0.50 for k, v in WEIGHTS.items()}),
                    ("all_weights_x1.50", {k: v * 1.50 for k, v in WEIGHTS.items()}),
                    ("uniform_0.25", {k: 0.25 for k in WEIGHTS})):
        s = score_all(rows, STRICT_MODES, w)
        r = rate(s, 0.2)
        rs = rate(s, 0.4)
        results["variants"][name] = {"fragile": r, "severe": rs}
        print(f"  {name:20s} fragile {r:.3%}   severe {rs:.3%}")

    # ---- mode set ----
    print("\n-- mode set (threshold 0.20) --")
    for name, modes in (("strict_5", STRICT_MODES),
                        ("full_9", FULL_MODES),
                        ("strongest_3", STRICT_MODES[:3]),
                        ("strict_plus_judge", STRICT_MODES + ["judge_no_reference"]),
                        ("strict_plus_noexec", STRICT_MODES + ["no_execution"])):
        s = score_all(rows, modes, WEIGHTS)
        r = rate(s, 0.2)
        rs = rate(s, 0.4)
        results["variants"][f"modeset_{name}"] = {"fragile": r, "severe": rs}
        print(f"  {name:22s} fragile {r:.3%}   severe {rs:.3%}")

    # ---- grading ----
    print("\n-- severity grading --")
    for grading in ("presence", "count"):
        s = score_all(rows, STRICT_MODES, WEIGHTS, grading=grading)
        r = rate(s, 0.2)
        rs = rate(s, 0.4)
        results["variants"][f"grading_{grading}"] = {"fragile": r, "severe": rs}
        print(f"  {grading:10s} fragile {r:.3%}   severe {rs:.3%}")

    # ---- how much does the headline move at all? ----
    fr = [v for k, v in results["variants"].items()
          if isinstance(v, float) and k.startswith("threshold_")]
    frag = [v["fragile"] for k, v in results["variants"].items() if isinstance(v, dict)]
    print(f"\n  threshold sweep: {min(fr):.1%} .. {max(fr):.1%}")
    print(f"  across all non-threshold variants: {min(frag):.1%} .. {max(frag):.1%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
