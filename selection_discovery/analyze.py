"""Summarise sd/ runs: exploitation, onset, audits and novelty per condition, plus a trace viewer.

Usage:
  python sd/analyze.py --runs results/sd/runs --audits results/sd/audits --out results/sd/report
"""

import argparse
import glob
import html
import json
import os
import statistics as st


def per_step(run_dir):
    """fp and true accuracy per optimizer step, from the trainer's log."""
    rows = {}
    for l in open(os.path.join(run_dir, "log.jsonl")):
        e = json.loads(l)
        if "fp" in e:
            rows[e["step"]] = e
    return [rows[k] for k in sorted(rows)]


def summarise(steps):
    fp = [e["fp"] for e in steps]
    if len(fp) < 30:
        return None
    acc = [e.get("rewards/reward_strict/mean") for e in steps]
    rolling = [st.mean(fp[i:i + 5]) for i in range(len(fp) - 4)]
    onset = next((steps[i]["step"] for i, v in enumerate(rolling) if v > 0.30), None)
    onset_half = next((steps[i]["step"] for i, v in enumerate(rolling) if v > 0.50), None)
    return {"steps": len(fp), "fp_first10": st.mean(fp[:10]), "fp_last20": st.mean(fp[-20:]),
            "exploitation": st.mean(fp[-20:]) - st.mean(fp[:10]), "onset_0.3": onset,
            "onset_0.5": onset_half, "acc_first10": st.mean(a for a in acc[:10] if a is not None),
            "acc_last20": st.mean(a for a in acc[-20:] if a is not None),
            "numbers_last20": st.mean(e.get("numbers", 0) for e in steps[-20:]),
            "chars_last20": st.mean(e.get("chars", 0) for e in steps[-20:])}


def trace_page(runs, out):
    """One HTML page per run: saved answers by step, hacks highlighted."""
    os.makedirs(out, exist_ok=True)
    for run in runs:
        path = os.path.join(run, "samples.jsonl")
        if not os.path.exists(path):
            continue
        S = [json.loads(l) for l in open(path)]
        name = os.path.basename(run)
        parts = [f"<title>{name}</title><style>body{{font:14px system-ui;max-width:960px;margin:24px auto;padding:0 16px}}"
                 "pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px}"
                 ".hack pre{background:#fde8e6}h2{margin-top:32px}</style>", f"<h1>{name}</h1>"]
        for step in sorted({s["step"] for s in S}):
            group = [s for s in S if s["step"] == step]
            hacks = [s for s in group if s["r_arm"] > 0.5 and s["r_strict"] == 0]
            parts.append(f"<h2>Step {step}: {len(hacks)} of {len(group)} paid but wrong</h2>")
            for s in (hacks + [g for g in group if g not in hacks])[:6]:
                cls = "hack" if s in hacks else ""
                parts.append(f"<div class='{cls}'><p>gold {html.escape(s['gold'])} · arm {s['r_arm']:.2f} · "
                             f"strict {s['r_strict']:.0f}</p><pre>{html.escape(s['reply'][-1500:])}</pre></div>")
        open(os.path.join(out, f"{name}.html"), "w").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--audits", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, "*_s*")) if os.path.isdir(d))
    audit = {}
    for f in glob.glob(os.path.join(a.audits or "", "audit_*.json")):
        audit.update(json.load(open(f))["conds"])
    novelty = {}
    nf = os.path.join(a.runs, "novelty_report.json")
    if os.path.exists(nf):
        novelty = json.load(open(nf))
    table = []
    for run in runs:
        name = os.path.basename(run)
        cond, seed = name.rsplit("_s", 1)
        s = summarise(per_step(run)) if os.path.exists(os.path.join(run, "log.jsonl")) else None
        if s is None:
            continue
        table.append({"run": name, "cond": cond, "seed": int(seed), **s,
                      "h": audit.get(cond, {}).get("h"), "s": audit.get(cond, {}).get("s"),
                      "s_max": audit.get(cond, {}).get("s_max"),
                      "novelty_hack": novelty.get(name, {}).get("novelty_hack_median_pct"),
                      "novelty_other": novelty.get(name, {}).get("novelty_other_median_pct"),
                      "n_hack": novelty.get(name, {}).get("n_hack")})
    json.dump(table, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    cols = ["run", "h", "s", "fp_first10", "fp_last20", "exploitation", "onset_0.3", "acc_first10",
            "acc_last20", "novelty_hack", "novelty_other", "n_hack"]
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in table]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    trace_page(runs, os.path.join(a.out, "traces"))


if __name__ == "__main__":
    main()
