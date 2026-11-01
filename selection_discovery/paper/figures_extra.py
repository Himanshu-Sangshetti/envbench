"""Setup diagram, exploitation summary and reachability figures (PDF for the paper, PNG for sharing)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "results", "sd", "main")
OUT = os.path.join(HERE, "figs")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight")


# 1. setup diagram
fig, ax = plt.subplots(figsize=(9.4, 3.9))
ax.set_xlim(0, 104); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, title, body, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5", fc=fc, ec="#33415c", lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=8.2, linespacing=1.35)


def arrow(a, b, txt="", dy=1.2, style="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#33415c", ls=style,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, txt, ha="center", fontsize=8, color="#33415c")


box(1, 20, 19, 16, "Problem", "MATH (levels 1-3)\nor MBPP task\n8 per step", "#eef3f9")
box(27, 20, 19, 16, "Qwen3-0.6B", "8 answers per\nproblem\n(64 per step)", "#eef3f9")
box(53, 20, 23, 16, "Flawed checker", "one per run, e.g. pays if\nthe right number appears\nanywhere in the answer", "#fbe9e7")
box(83, 20, 20, 16, "RL update", "Dr. GRPO: reinforce\nthe answers the\nchecker paid", "#eef3f9")
box(53, 1, 23, 13, "Referee (silent)", "robust maths grader / MBPP+\nnever affects training", "#e7f3ec", ls="--")
box(1, 1, 45, 13, "What we measure", "before: untrained model paid while wrong (h)\nduring: paid-but-wrong rate = cheating\nafter: can the untrained model already cheat there?", "#f6f6f6")
arrow((20.6, 28), (26.4, 28))
arrow((46.6, 28), (52.4, 28), "answers")
arrow((76.6, 28), (82.4, 28), "reward")
arrow((64.5, 19.4), (64.5, 14.6))
arrow((93, 36.8), (36.5, 36.8), "updated model, next step", dy=1.6, style="--", rad=0.18)
ax.text(103, 45, "repeat 120 steps · 20 runs", ha="right", fontsize=8.5, color="#33415c", style="italic")
save(fig, "setup")

# 2. exploitation summary per checker
R = {r["run"]: r for r in json.load(open(os.path.join(ROOT, "relabelled.json")))}
groups = [("strict", "honest maths\ncontrol", 1, "#2f7a4f", "0%"), ("tail50_density", "narrow loophole\n+ random credit", 4, "#b7c2d0", "1.3%"),
          ("substring_tail50", "narrow loophole\n(last 50 chars)", 4, "#7a8ba3", "1.3%"), ("tail50_slope", "narrow loophole\n+ ramp", 4, "#2f5d8a", "1.3%"),
          ("judge_ref_prob", "LLM judge\n(Qwen3-4B)", 1, "#c0782a", "1.4%"), ("substring_all", "'number\nanywhere'", 2, "#a63d3d", "8.0%"),
          ("mbpp_plus", "honest code\ncontrol", 1, "#2f7a4f", "0%"), ("mbpp_base", "MBPP weak\ntests", 3, "#a63d3d", "2.6%")]
fig, ax = plt.subplots(figsize=(10.4, 3.8))
for i, (c, lab, n, col, h) in enumerate(groups):
    vals = [R[f"{c}_s{s}"]["expl_new"] for s in range(n)]
    ax.bar(i, sum(vals) / n, color=col, alpha=.35, width=.7)
    ax.scatter([i] * n, vals, color=col, s=34, zorder=3, edgecolor="white", lw=.6)
    ax.text(i, -0.115, f"start: {h}", ha="center", fontsize=8, color="#5a6476")
ax.axhline(0, color="#999", lw=.8)
ax.axhspan(-0.03, 0.03, color="#999", alpha=.08, lw=0)
ax.set_xticks(range(len(groups)), [g[1] for g in groups], fontsize=8)
ax.set_ylabel("cheating rise over training\n(paid-but-wrong, end − start)")
ax.set_ylim(-0.14, 0.5)
ax.axvline(5.5, color="#ccc", lw=1, ls=":")
ax.text(2.5, 0.47, "maths", ha="center", color="#5a6476"); ax.text(6.5, 0.47, "code", ha="center", color="#5a6476")
ax.text(-0.35, 0.036, "noise band ±0.03", fontsize=7.5, color="#888", ha="left")
save(fig, "summary")

# 3. reachability
rm = json.load(open(os.path.join(ROOT, "runs", "reach_report_math.json")))
rc = json.load(open(os.path.join(ROOT, "runs", "reach_report_mbpp.json")))
rows = [("'number anywhere' s0", rm["substring_all_s0"]), ("'number anywhere' s1", rm["substring_all_s1"]),
        ("MBPP weak tests s0", rc["mbpp_base_s0"]), ("MBPP weak tests s1", rc["mbpp_base_s1"]), ("MBPP weak tests s2", rc["mbpp_base_s2"])]
fig, ax = plt.subplots(figsize=(6.2, 2.8))
ys = range(len(rows))
ax.barh(ys, [1] * len(rows), color="#eceff3")
ax.barh(ys, [e["reach"] for _, e in rows], color="#2f5d8a")
for y, (_, e) in zip(ys, rows):
    ax.text(e["reach"] + .02, y, f"{e['reach']:.0%} of {e['n_hack_problems']} problems", va="center", fontsize=8.5)
ax.set_yticks(list(ys), [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 1.25)
ax.set_xlabel("share of cheated problems where the untrained model\nalready cheats within 64 tries (reach)")
save(fig, "reach")

# PNG copies of the existing figures for sharing
import subprocess  # noqa: E402
print("ok")
