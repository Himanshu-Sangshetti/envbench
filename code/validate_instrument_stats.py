"""Reproduce the instrument-validation numbers, before and after recalibration.

The paper reports that the audit instrument reaches AUROC 0.700 on a labelled
set, up from 0.664 before a presence-based recalibration, with recall at
threshold 0.2 rising from 0.586 to 0.724. Those numbers must be regenerable
from the artifact, not taken on trust.

This script re-audits the labelled set under both scorings:

  * **before** -- count-graded severity (``score(a, presence=False)``), the
    detector as it stood before the validation forced a recalibration;
  * **after** -- presence-graded severity (``score(a, presence=True)``), used
    everywhere else in the paper.

and reports, for each: AUROC with a stratified bootstrap interval, and the
threshold metrics at 0.2.

    python scripts/validate_instrument_stats.py

Cost: $0. It pulls public environment sources and reads them; no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from static_audit import (  # noqa: E402
    SKIP_DIR_RE,
    SKIP_FILE_RE,
    VERIFIER_FILE_RE,
    analyse_source,
    is_deliberate_adversarial,
    score,
)

SEED = 20261020
BOOT = 10000
THRESHOLD = 0.20
PRIME = os.path.expanduser("~/.local/bin/prime")


def pull(env_id: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        p = subprocess.run(
            [PRIME, "env", "pull", env_id, "--target", str(dest), "--plain"],
            capture_output=True, text=True, timeout=300,
        )
        return p.returncode == 0
    except Exception:
        return False


def analyse(root: Path):
    """Return the StaticAudit with raw signals merged, before scoring."""
    from static_audit import StaticAudit

    a = StaticAudit()
    py = [p for p in root.rglob("*.py")
          if not SKIP_FILE_RE.search(str(p.relative_to(root)))
          and not SKIP_DIR_RE.search(str(p.relative_to(root)))]
    cands = [p for p in py if VERIFIER_FILE_RE.search(p.name)] or py[:3]
    merged = dict(exact_string_eq=0, substring_match=0, startswith_endswith=0,
                  float_eq_no_tol=0, regex_num_extract=0, set_membership=0)
    for p in cands:
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sig = analyse_source(src)
        for k in merged:
            merged[k] += sig.get(k, 0)
        for k in ("has_lower", "has_strip", "has_unicodedata", "has_approx",
                  "has_float_cast", "is_llm_judge", "judge_has_reference",
                  "executes_candidate", "runs_test_suite", "has_hardcoded_answers"):
            a.__dict__[k] = bool(a.__dict__[k] or sig.get(k, False))
    for k, v in merged.items():
        a.__dict__[k] = v
    return a


def labelled_set(registry: Path, limit: int):
    import yaml
    reg = yaml.safe_load(open(registry, encoding="utf-8"))
    pos, neg = [], []
    for e in reg["environments"]:
        if is_deliberate_adversarial(e["env_id"], e.get("tags", [])):
            pos.append(e)
        elif e["verifier_hint"] == "program":
            neg.append(e)

    def dedup(lst):
        seen, out = set(), []
        for e in sorted(lst, key=lambda x: -x["audit_priority"]):
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            out.append(e)
        return out

    pos, neg = dedup(pos), dedup(neg)
    k = min(len(pos), len(neg), limit)
    return pos[:k], neg[:k]


def boot_auc(y, s, B=BOOT, seed=SEED):
    """Stratified percentile bootstrap CI for the AUROC."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    reps = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
        try:
            reps.append(roc_auc_score(y[i], s[i]))
        except Exception:
            pass
    reps = np.asarray(reps)
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), reps.size


def metrics(y, s, thr=THRESHOLD):
    flagged = s >= thr
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(1, tp + tn + fp + fn)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="envs/registry_generated.yaml")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/instrument_validation_stats.json")
    args = ap.parse_args()

    pos, neg = labelled_set(Path(args.registry), args.limit)
    k = min(len(pos), len(neg))
    pos, neg = pos[:k], neg[:k]
    print(f"labelled set: {len(pos)} positive, {len(neg)} negative")

    rows = []
    for label, group in (("POS", pos), ("NEG", neg)):
        for e in group:
            tmp = Path(tempfile.mkdtemp(prefix="ivs_"))
            try:
                if not pull(e["env_id"], tmp):
                    print(f"  skip (pull failed): {e['env_id']}")
                    continue
                a = analyse(tmp)
                before = score(a, presence=False).fragility_score
                after = score(a, presence=True).fragility_score
                rows.append({"env_id": e["env_id"], "label": label,
                             "before": before, "after": after})
                print(f"  {label} {e['env_id'][:44]:<46} before={before:.4f} after={after:.4f}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    if len(rows) < 10:
        print("too few labelled environments analysed; aborting")
        return 1

    y = np.array([1 if r["label"] == "POS" else 0 for r in rows])
    out = {"n": len(rows), "threshold": THRESHOLD, "seed": SEED,
           "bootstrap_reps": BOOT, "rows": rows}

    for key in ("before", "after"):
        s = np.array([r[key] for r in rows])
        auroc = float(roc_auc_score(y, s))
        lo, hi, B = boot_auc(y, s)
        m = metrics(y, s)
        out[key] = {"auroc": auroc, "auroc_ci_low": lo, "auroc_ci_high": hi,
                    "boot_reps_used": B, **m}
        print(f"\n{key.upper():6s}: AUROC {auroc:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
              f"recall@0.2 {m['recall']:.3f}  precision {m['precision']:.3f}  "
              f"F1 {m['f1']:.3f}  acc {m['accuracy']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
