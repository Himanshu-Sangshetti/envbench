"""Inter-rater agreement for the hand audit.

The auditors are LLM instances applying a written rubric, not human experts.
That is a real limitation and the paper has to state it, but the right way to
state it is with a number rather than a caveat: label a subset of the packets a
second time, blind to the first label, and report the agreement.

Two agreement measures are reported:

  * raw agreement on the 5-way verdict;
  * Cohen's kappa, which corrects for chance agreement and is the standard
    measure for nominal labels;
  * agreement on the coarser question the paper actually uses --- "does this
    verifier have a false-positive path?" --- collapsed to a binary
    FP-risk / no-FP-risk label.

    python scripts/interrater.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FP = {"fp_possible", "fp_possible_but_hard"}


def binary(v: str) -> str:
    return "fp" if v in FP else ("unclear" if v == "unclear" else "no_fp")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((Counter(a)[c] / n) * (Counter(b)[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    dup_dir = Path("results/hand_audit/rerun")
    if not dup_dir.exists():
        print(f"no second-pass labels in {dup_dir}")
        return 0

    base = Path("results/hand_audit/packets")
    pairs = []
    for f in sorted(dup_dir.glob("*/label.json")):
        first = base / f.parent.name / "label.json"
        if not first.exists():
            continue
        a = json.loads(first.read_text())
        b = json.loads(f.read_text())
        pairs.append((a, b))

    if not pairs:
        print("no comparable pairs")
        return 0

    va = [a["verdict"] for a, _ in pairs]
    vb = [b["verdict"] for _, b in pairs]
    ba = [binary(v) for v in va]
    bb = [binary(v) for v in vb]

    raw = sum(x == y for x, y in zip(va, vb)) / len(va)
    k5 = cohen_kappa(va, vb)
    rawb = sum(x == y for x, y in zip(ba, bb)) / len(ba)
    kb = cohen_kappa(ba, bb)

    print(f"double-labelled packets: {len(pairs)}")
    print(f"  raw agreement, 5-way verdict : {raw:.3f}")
    print(f"  Cohen's kappa, 5-way verdict : {k5:.3f}")
    print(f"  raw agreement, FP-risk binary: {rawb:.3f}")
    print(f"  Cohen's kappa, FP-risk binary: {kb:.3f}")
    print("  confusion (row = first pass, col = second pass):")
    cats = sorted(set(va) | set(vb))
    print("    " + " ".join(f"{c[:11]:>12s}" for c in cats))
    for c in cats:
        row = [sum(1 for x, y in zip(va, vb) if x == c and y == d) for d in cats]
        print(f"  {c[:12]:<13s}" + " ".join(f"{v:>12d}" for v in row))

    out = {"n": len(pairs), "raw_5way": raw, "kappa_5way": k5,
           "raw_binary": rawb, "kappa_binary": kb}
    (dup_dir / "agreement.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
