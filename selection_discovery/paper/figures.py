"""Figures for the paper, from results/sd/main (saved-sample curves, re-scoring and reachability)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "sd", "main")
OUT = os.path.join(os.path.dirname(__file__), "figs")
C = json.load(open(os.path.join(ROOT, "curves_and_examples.json")))["curves"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def series(run, i):
    return [r[0] for r in C[run]], [r[i] for r in C[run]]


# Figure 1: paid-but-wrong rate (final label) over training
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
groups = [("tail50_slope", "#2f5d8a"), ("substring_tail50", "#7a8ba3"), ("tail50_density", "#b7c2d0")]
for cond, col in groups:
    for s in range(4):
        x, y = series(f"{cond}_s{s}", 2)
        ax[0].plot(x, y, color=col, lw=1, alpha=.9, label=cond if s == 0 else None)
for s, ls in ((0, "-"), (1, "--")):
    x, y = series(f"substring_all_s{s}", 2)
    ax[0].plot(x, y, color="#a63d3d", lw=1.8, ls=ls, label=f"substring_all s{s}")
x, y = series("judge_ref_prob_s0", 2)
ax[0].plot(x, y, color="#c0782a", lw=1.5, label="judge_ref_prob")
x, y = series("strict_s0", 2)
ax[0].plot(x, y, color="#2f7a4f", lw=1.5, label="strict")
ax[0].set_title("Maths"); ax[0].set_xlabel("step"); ax[0].set_ylabel("paid-but-wrong rate")
ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
for s in range(3):
    x, y = series(f"mbpp_base_s{s}", 2)
    ax[1].plot(x, y, color="#a63d3d", lw=1.2, label="mbpp_base" if s == 0 else None)
x, y = series("mbpp_plus_s0", 2)
ax[1].plot(x, y, color="#2f7a4f", lw=1.5, label="mbpp_plus")
ax[1].set_title("Code (MBPP)"); ax[1].set_xlabel("step"); ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fp_curves.pdf"))

# Figure 2: flagged "cheats" under the first label vs those still wrong under the final label
R = {r["run"]: r for r in json.load(open(os.path.join(ROOT, "relabelled.json")))}
runs = ["tail50_slope_s1", "judge_ref_prob_s0", "substring_all_s1", "substring_all_s0"]
old = [R[r]["old_hacks"] for r in runs]
real = [R[r]["old_hacks"] - R[r]["false_alarms"] for r in runs]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.barh(runs, old, color="#e3c9a6", label="flagged by box-required label")
ax.barh(runs, real, color="#a63d3d", label="still wrong under final label")
for i, (o, r) in enumerate(zip(old, real)):
    ax.text(o + 5, i, f"{100 * (o - r) / o:.0f}% right", va="center", fontsize=7)
ax.set_xlabel("paid answers flagged as wrong")
ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
ax.set_xlim(0, max(old) * 1.35)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "grader_artefact.pdf"))

# Figure 3: answer length over training
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for run, col, lab in (("substring_all_s0", "#a63d3d", "substring_all s0"), ("tail50_slope_s1", "#2f5d8a", "tail50_slope s1"),
                      ("judge_ref_prob_s0", "#c0782a", "judge_ref_prob"), ("strict_s0", "#2f7a4f", "strict")):
    x, y = series(run, 4)
    ax.plot(x, y, color=col, lw=1.5, label=lab)
ax.set_xlabel("step"); ax.set_ylabel("mean answer length (chars)"); ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "length.pdf"))
print("ok")
