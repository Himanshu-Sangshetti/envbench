"""Aggregate the hand audit: does the static detector predict a real defect?

The hand audit is the answer to the central objection to this paper: that the
detector counts syntax patterns, and that correct verifiers are full of syntax
patterns. Each sampled environment was read by an auditor who could see the
verifier source but not the detector's score, and judged on one question: can a
plausible-but-wrong answer receive full reward?

This script joins those labels to the detector's output and reports:

  * precision and recall of the detector as a *classifier of FP risk*, with a
    Wilson interval, computed separately for the strict score and the full score;
  * per-mode precision, i.e. for each defect mode, the fraction of flagged
    environments that the auditor judged to have a real false-positive path;
  * the same, restricted to environments the auditor judged with high confidence
    and not ``unclear``;
  * agreement by stratum, which is the point of the stratified sample.

    python scripts/hand_audit_analysis.py
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import STRICT_MODES, FULL_MODES, load  # noqa: E402

FP_VERDICTS = {"fp_possible", "fp_possible_but_hard"}
ROBUST = {"fp_not_possible"}
STRICT_FP = {"fp_possible"}          # excludes the "but hard" borderline
OUT = Path("results/hand_audit_analysis.json")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else float("nan")
    lo, hi = wilson(tp, tp + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "precision_ci": [lo, hi], "recall": rec, "f1": f1,
            "n": tp + fp + fn + tn,
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn)}


def main() -> int:
    manifest = json.loads(Path("results/hand_audit/sample.json").read_text())
    labels = {}
    for m in manifest:
        d = Path(m["packet_dir"])
        lab = d / "label.json"
        if lab.exists():
            try:
                labels[m["env_id"]] = json.loads(lab.read_text())
            except json.JSONDecodeError:
                pass
    print(f"packets: {len(manifest)}   labelled: {len(labels)}")
    if not labels:
        print("no labels yet")
        return 1

    rows = load("results/envbench/envbench_registry.csv")
    by_id = {r["env_id"]: r for r in rows}

    out = {"n_packets": len(manifest), "n_labelled": len(labels),
           "verdicts": dict(collections.Counter(l["verdict"] for l in labels.values())),
           "by_stratum": {}, "per_mode": {}, "classifier": {}}

    # ── verdict x stratum ────────────────────────────────────────────────
    strat = collections.defaultdict(collections.Counter)
    for m in manifest:
        l = labels.get(m["env_id"])
        if l:
            strat[m["stratum"]][l["verdict"]] += 1
    for s, c in strat.items():
        tot = sum(c.values())
        fp = sum(c[v] for v in FP_VERDICTS)
        out["by_stratum"][s] = {"n": tot, "fp_rate": fp / tot if tot else None,
                                "verdicts": dict(c)}
        print(f"  {s:18s} n={tot:3d}  FP-judged {fp:3d} ({fp/tot:.1%})  {dict(c)}")

    # ── the detector as a classifier of FP risk ──────────────────────────
    for score_name, col in (("strict", "fragility_score_strict"),
                            ("full", "fragility_score")):
        for thr, name in ((0.2, "ge_0.2"), (0.4, "ge_0.4")):
            for subset, sname in ((None, "all_labelled"),
                                  ("confident", "confident_only")):
                tp = fp = fn = tn = 0
                for m in manifest:
                    l = labels.get(m["env_id"])
                    if not l:
                        continue
                    if subset == "confident" and (l.get("confidence") != "high"
                                                  or l["verdict"] == "unclear"):
                        continue
                    r = by_id.get(m["env_id"])
                    if r is None:
                        continue
                    flagged = float(r[col]) >= thr
                    real = l["verdict"] in FP_VERDICTS
                    if flagged and real:
                        tp += 1
                    elif flagged and not real:
                        fp += 1
                    elif not flagged and real:
                        fn += 1
                    else:
                        tn += 1
                key = f"{score_name}_{name}_{sname}"
                out["classifier"][key] = prf(tp, fp, fn, tn)
                c = out["classifier"][key]
                if c["n"] >= 10:
                    print(f"  {key:32s} n={c['n']:3d} P={c['precision']:.3f} "
                          f"R={c['recall']:.3f} F1={c['f1']:.3f}")

    # ── per-mode precision ───────────────────────────────────────────────
    mode_hits = collections.defaultdict(lambda: [0, 0])   # mode -> [fp_judged, flagged]
    for m in manifest:
        l = labels.get(m["env_id"])
        r = by_id.get(m["env_id"])
        if not l or r is None:
            continue
        real = l["verdict"] in FP_VERDICTS
        for mode in (r.get("defect_modes") or "").split("|"):
            mode = mode.split(" x")[0].strip()
            if not mode:
                continue
            mode_hits[mode][1] += 1
            if real:
                mode_hits[mode][0] += 1
    print("\n  per-mode precision (share of flagged environments the auditor judged FP-risk):")
    for mode, (hit, tot) in sorted(mode_hits.items(), key=lambda kv: -kv[1][1]):
        p = hit / tot if tot else float("nan")
        lo, hi = wilson(hit, tot)
        out["per_mode"][mode] = {"flagged": tot, "fp_judged": hit, "precision": p,
                                 "precision_ci": [lo, hi]}
        print(f"    {mode:42s} {hit:4d}/{tot:4d}  P={p:.3f} [{lo:.3f},{hi:.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
