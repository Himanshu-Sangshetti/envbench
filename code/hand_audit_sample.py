"""Build a blind hand-audit sample: the fix for the paper's central objection.

The critique that matters is this: *"the defects are syntax patterns; they appear
in perfectly correct verifiers all the time."* The paper cannot answer that with
more statistics about the same patterns. It answers it by reading verifiers.

This script builds the sample:

  * stratified over the detector's output (severe / fragile / clean, plus a
    deliberate-adversarial slice that is *excluded* from the headline) so that
    precision and recall are estimable;
  * **blind**: the packet written for the labeller contains the environment's
    source, its task description and a sample task -- and NOT the fragility
    score, the defect modes, or the family rate. Labels are therefore a function
    of the code, not of the detector;
  * **seeded**, so the sample is reproducible and pre-specified.

    python scripts/hand_audit_sample.py --n 200 --out results/hand_audit

Outputs:
    results/hand_audit/packets/NNN_<env_id>/  packet.md, verifier.py, task.json
    results/hand_audit/sample.json           the manifest (with scores, for the
                                              analysis step -- not for the labeller)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRIME = os.path.expanduser("~/.local/bin/prime")

STRATA = [
    # name,            strict-score band,   n
    ("flagged_severe", "severe", 70),      # positives for the strict score
    ("flagged_fragile", "fragile", 70),    # positives for the strict score
    ("unflagged_clean", "clean", 35),      # negatives
]


def band(x: float) -> str:
    if x < 0.05:
        return "clean"
    if x < 0.2:
        return "mild"
    if x < 0.4:
        return "fragile"
    return "severe"


def pull(env_id: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.run(
            [PRIME, "env", "pull", env_id, "--target", str(dest), "--plain"],
            capture_output=True, text=True, timeout=300,
        )
        return p.returncode == 0
    except Exception:
        return False


def verifier_files(root: Path) -> list[Path]:
    from static_audit import SKIP_DIR_RE, SKIP_FILE_RE, VERIFIER_FILE_RE

    py = [p for p in root.rglob("*.py")
          if not SKIP_FILE_RE.search(str(p.relative_to(root)))
          and not SKIP_DIR_RE.search(str(p.relative_to(root)))]
    cands = [p for p in py if VERIFIER_FILE_RE.search(p.name)] or py[:3]
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/envbench/envbench_registry.csv")
    ap.add_argument("--out", default="results/hand_audit")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--n-weak-only", type=int, default=45)
    args = ap.parse_args()

    import csv

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    out = Path(args.out)
    (out / "packets").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    chosen = []
    for name, bnd, n in STRATA:
        pool = [r for r in rows if band(float(r["fragility_score_strict"])) == bnd]
        rng.shuffle(pool)
        for r in pool[:n]:
            chosen.append((name, r))

    # A fourth stratum: flagged by the FULL score but not by the strict one.
    # These are the environments whose only defects are the five weaker modes,
    # and they are where a hand audit says most about whether those modes are
    # real defects or noise.
    weak_only = [r for r in rows
                 if float(r["fragility_score_strict"]) < 0.2
                 and float(r["fragility_score"]) >= 0.2]
    rng.shuffle(weak_only)
    for r in weak_only[:args.n_weak_only]:
        chosen.append(("full_score_only", r))
    rng.shuffle(chosen)

    manifest = []
    for i, (stratum, r) in enumerate(chosen, 1):
        env_id = r["env_id"]
        slug = env_id.replace("/", "__")
        d = out / "packets" / f"{i:03d}_{slug}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="ha_"))
        ok = pull(env_id, tmp)
        if not ok:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  [{i}/{len(chosen)}] pull failed, skipped: {env_id}")
            continue

        vf = verifier_files(tmp)
        src = ""
        for f in vf:
            src += f"\n# ===== {f.name} =====\n" + f.read_text(encoding="utf-8", errors="ignore")
        (d / "verifier.py").write_text(src, encoding="utf-8")

        # task description + sample task, for judging what "correct" means
        task = {
            "env_id": env_id,
            "registry_family": r["family"] or "other",
            "registry_tags": r.get("tags", ""),
            "registry_train": r.get("train", ""),
            "readme": "",
        }
        for rm in ("README.md", "readme.md"):
            p = tmp / rm
            if p.exists():
                task["readme"] = p.read_text(encoding="utf-8", errors="ignore")[:6000]
                break
        for cf in ("tasks.py", "task.py", "dataset.py", "prompts.py"):
            p = tmp / cf
            if p.exists():
                task["task_source"] = p.read_text(encoding="utf-8", errors="ignore")[:6000]
                break
        (d / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")

        shutil.rmtree(tmp, ignore_errors=True)
        manifest.append({
            "packet_id": f"{i:03d}",
            "stratum": stratum,
            "env_id": env_id,
            "packet_dir": str(d),
            # filled in at analysis time; NOT shown to the labeller
            "fragility_score_strict": float(r["fragility_score_strict"]),
            "fragility_score_full": float(r["fragility_score"]),
            "defect_modes": r.get("defect_modes", ""),
            "n_defect_modes": int(r.get("n_defect_modes") or 0),
            "deliberate_adversarial": r.get("deliberate_adversarial", ""),
        })
        print(f"  [{i}/{len(chosen)}] {stratum:18s} {env_id}")

    prev = out / "sample.json"
    if prev.exists():
        old = json.loads(prev.read_text(encoding="utf-8"))
        have = {m["env_id"] for m in old}
        merged = old + [m for m in manifest if m["env_id"] not in have]
        manifest = merged
    (out / "sample.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = {}
    for m in manifest:
        counts[m["stratum"]] = counts.get(m["stratum"], 0) + 1
    print(f"\n{len(manifest)} packets -> {out}")
    print("  strata:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
