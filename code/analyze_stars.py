"""Popularity vs. verifier quality: the one analysis behind Table 5 and Figure 5.

Everything here is computed from the released artifact
(``results/envbench/envbench_registry.csv``) so that the numbers in the paper
and the numbers in the figure come from one place and cannot drift apart.

Method
------
Point estimate
    Spearman rank correlation between registry stars and the strict fragility
    score, with average ranks for ties (stars are massively tied: 986 of the
    1,496 headline environments carry zero stars, and many carry one).

Interval
    95% percentile interval from a **family-clustered** bootstrap: task
    families are resampled with replacement, which respects the fact that
    environments within a family are not independent. This is the same
    clustered convention the rest of the paper uses for rates.

Note on ties
    A naive rank (``argsort(argsort(x))``) breaks ties arbitrarily and inflates
    the correlation. That mistake is what an earlier draft of Figure 5 printed.
    ``scipy.stats.spearmanr`` is used here precisely because it averages ranks
    across ties.

    python scripts/analyze_stars.py --out results/stars_analysis.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20261020
BOOT = 4000
SCORE = "fragility_score_strict"


def load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    head = [r for r in rows
            if (r.get("deliberate_adversarial") or "").strip().lower()
            in ("false", "0", "")]
    stars = np.array([float(r.get("stars") or 0) for r in head])
    frag = np.array([float(r[SCORE]) for r in head])
    fam = np.array([(r.get("family") or "other") for r in head])
    return head, stars, frag, fam


def cluster_bootstrap(stars, frag, fam, mask, rng, B=BOOT):
    """Percentile CI from resampling task families with replacement."""
    fams = np.unique(fam[mask])
    idx = {f: np.where(mask & (fam == f))[0] for f in fams}
    reps = []
    for _ in range(B):
        pick = rng.choice(fams, size=len(fams), replace=True)
        sel = np.concatenate([idx[f] for f in pick])
        if sel.size > 10 and np.unique(stars[sel]).size > 1:
            r = spearmanr(stars[sel], frag[sel]).statistic
            if np.isfinite(r):
                reps.append(r)
    reps = np.asarray(reps)
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), reps.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/envbench/envbench_registry.csv")
    ap.add_argument("--out", default="results/stars_analysis.json")
    args = ap.parse_args()

    head, stars, frag, fam = load(Path(args.csv))
    rng = np.random.default_rng(SEED)

    out = {
        "population": len(head),
        "score": SCORE,
        "method": "Spearman with tie-averaged ranks; family-clustered percentile bootstrap",
        "seed": SEED,
        "bootstrap_reps": BOOT,
        "n_zero_stars": int((stars == 0).sum()),
    }

    for name, mask in (("all", np.ones(len(head), bool)), ("starred", stars > 0)):
        rho = float(spearmanr(stars[mask], frag[mask]).statistic)
        lo, hi, B = cluster_bootstrap(stars, frag, fam, mask, rng)
        out[name] = {"n": int(mask.sum()), "rho": rho, "ci_low": lo, "ci_high": hi,
                     "boot_reps_used": B}

    # starred vs unstarred comparison
    st = stars > 0
    for label, sel in (("starred", st), ("unstarred", ~st)):
        out.setdefault("by_popularity", {})[label] = {
            "n": int(sel.sum()),
            "median_fragility": float(np.median(frag[sel])),
            "severe_rate": float((frag[sel] >= 0.4).mean()),
            "strictly_fragile_rate": float((frag[sel] >= 0.2).mean()),
        }

    # family composition of the starred population
    out["starred_families"] = dict(collections.Counter(fam[st]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    a, s = out["all"], out["starred"]
    print(f"population            : {out['population']}  (zero stars: {out['n_zero_stars']})")
    print(f"Spearman, all         : rho = {a['rho']:+.3f}  "
          f"CI [{a['ci_low']:+.3f}, {a['ci_high']:+.3f}]  (n={a['n']})")
    print(f"Spearman, starred only: rho = {s['rho']:+.3f}  "
          f"CI [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  (n={s['n']})")
    for k, v in out["by_popularity"].items():
        print(f"  {k:10s}: median {v['median_fragility']:.3f}  "
              f"severe {v['severe_rate']:.1%}  strictly-fragile {v['strictly_fragile_rate']:.1%}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
