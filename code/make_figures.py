"""Generate every figure in Paper 1 at its final printed size.

    python scripts/make_figures.py --out ../paper/figures

All figures share one palette and are built at the width they occupy in the
document, so nothing is scaled on the way in and the type size on the page is
exactly what is set here.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (  # noqa: E402
    ACCENT,
    BAND_COLOR,
    BAND_ORDER,
    COL_H,
    COL_W,
    FULL_W,
    HAIR,
    INK,
    MUTED,
    PANEL,
    SEQ,
    SEQ2,
    clean,
    label_bars,
    title,
    use_style,
)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


# ---------------------------------------------------------------------------
def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    head = [r for r in rows if r["deliberate_adversarial"].lower() in ("false", "0", "")]
    return rows, head


def band(r, strict=True):
    x = float(r["fragility_score_strict"] if strict else r["fragility_score"])
    return "clean" if x < 0.05 else ("mild" if x < 0.2 else ("fragile" if x < 0.4 else "severe"))


ABBREV = {
    "LLM judge without reference anchoring": "judge no ref",
    "test suite without executing candidate": "no exec",
    "unnormalised string equality": "unnormalised =",
    "startswith/endswith": "startswith",
    "set_membership": "set member",
    "substring_match": "substring",
    "float equality without tolerance": "float =",
    "regex numeric extraction": "regex num",
    "hardcoded answer literals": "hardcoded",
}


def modes_of(r):
    s = (r.get("defect_modes") or "").strip()
    if not s:
        return []
    return list(dict.fromkeys(m.split(" x")[0] for m in s.split("|") if m))


# ---------------------------------------------------------------------------
# 1. pipeline schematic (full width)
# ---------------------------------------------------------------------------
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(FULL_W, 1.62))
    # Full-bleed: default subplot margins would eat 22% of the schematic's
    # width and shrink every label on the page.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, FULL_W)
    ax.set_ylim(0, 1.62)
    ax.axis("off")

    boxes = [
        (0.00, "Registry\n1,614 environments", ACCENT["primary"]),
        (1.36, "Static audit\n9 defect modes", ACCENT["primary"]),
        (2.72, "Components\nA / B / C", ACCENT["alt"]),
        (4.08, "Predictor\nleave-one-family-out", ACCENT["accent"]),
        (5.44, "Decision\nwhere to spend compute", ACCENT["bad"]),
    ]
    w, h, y = 1.16, 0.62, 0.66
    for x, lab, col in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor="white", edgecolor=col, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center",
                fontsize=7.0, color=INK, linespacing=1.4)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w
        x1 = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0 + 0.02, y + h / 2), (x1 - 0.02, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=MUTED, linewidth=0.8))

    # validation arms, drawn under the two stages they validate
    arms = [
        (1.36, "Direct demonstration:\nthe known-defective environment", ACCENT["good"]),
        (4.08, "Labelled set:\nAUROC 0.700 (0.562-0.826)", ACCENT["accent"]),
    ]
    for x, lab, col in arms:
        ax.add_patch(FancyBboxPatch(
            (x, 0.08), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor="white", edgecolor=col, linewidth=0.9, linestyle=(0, (3, 2))))
        ax.text(x + w / 2, 0.29, lab, ha="center", va="center", fontsize=6.2,
                color=col, linespacing=1.35)
        ax.add_patch(FancyArrowPatch((x + w / 2, 0.64), (x + w / 2, 0.52),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=col, linewidth=0.7, linestyle=(0, (3, 2))))

    ax.text(0, 1.52, "Measurement pipeline", fontsize=8.2, color=INK, fontweight="bold")
    ax.text(0, 0.00, "Solid: executed at registry scale.   Dashed: validation of the "
                      "instrument itself.",
            fontsize=6.1, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. defect-mode co-occurrence (single column)
# ---------------------------------------------------------------------------
def fig_cooccurrence(head, out):
    modes = sorted({m for r in head for m in modes_of(r)})
    if not modes:
        return
    idx = {m: i for i, m in enumerate(modes)}
    n = len(modes)
    C = np.zeros((n, n))
    for r in head:
        ms = [m for m in modes_of(r) if m in idx]
        for a in ms:
            for b in ms:
                C[idx[a], idx[b]] += 1
    diag = np.diag(C).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where((diag[:, None] + diag[None, :] - C) > 0,
                        C / (diag[:, None] + diag[None, :] - C), 0.0)
    np.fill_diagonal(jac, 1.0)

    fig, ax = plt.subplots(figsize=(FULL_W, 3.15))
    im = ax.imshow(jac, cmap=SEQ, vmin=0, vmax=max(0.25, float(jac.max())), aspect="auto")
    short = [ABBREV.get(m, m.replace("_", " ")) for m in modes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=6.6, color=INK)
    ax.set_yticklabels(short, fontsize=6.6, color=INK)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i != j and jac[i, j] > 0:
                ax.text(j, i, f"{jac[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.6, color=INK if jac[i, j] > 0.18 else MUTED)
    # A vertical colorbar on the right: a horizontal one collides with the
    # rotated column labels underneath the matrix.
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.022, pad=0.015,
                      shrink=0.88)
    cb.set_label("Jaccard overlap", color=MUTED, fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8, color=HAIR, labelcolor=MUTED, length=2)
    cb.outline.set_edgecolor(HAIR)
    title(ax, "Defect modes co-occur weakly",
          "pairwise Jaccard; the diagonal is 1 by definition")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. auditability funnel (single column)
# ---------------------------------------------------------------------------
def fig_funnel(arows, out):
    n = len(arows)
    if not n:
        return

    def err(r):
        return (r.get("error") or "").lower()

    installed = [r for r in arows if r.get("install_rc") == 0]
    importable = [r for r in installed
                  if "import failed" not in err(r)
                  and "no taskset/env subclass" not in err(r)]
    resolved = [r for r in importable
                if (r.get("n_tasks") or 0) > 0 and "no summary.json" not in err(r)]
    measured = [r for r in resolved
                if (r.get("n_tasks") or 0) >= 3 and r.get("gold_valid") is not None]

    stages = [("attempted", n),
              ("installed", len(installed)),
              ("importable", len(importable)),
              ("tasks resolved", len(resolved)),
              ("fully measured", len(measured))]

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ys = np.arange(len(stages))[::-1]
    cols = [ACCENT["primary"], ACCENT["primary"], ACCENT["alt"], ACCENT["accent"], ACCENT["bad"]]
    for y, (lab, val), c in zip(ys, stages, cols):
        ax.barh(y, val, height=0.58, color=c, edgecolor="none")
        ax.text(val + n * 0.018, y, f"{val}", va="center", ha="left",
                fontsize=6.6, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages], fontsize=6.6, color=INK)
    ax.set_xlim(0, n * 1.14)
    ax.set_xlabel("environments", fontsize=6.6, color=INK)
    clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title(ax, "The registry is not auditable as shipped",
          f"{n - len(measured)} of {n} environments could not be validated")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. family x band: horizontal stacked bar
# ---------------------------------------------------------------------------
def fig_family_band(head, out):
    """One bar per family, split into the four fragility bands.

    A stacked bar beats a heatmap here: the reader's question is "which
    families are worst?", and the severe segment answers it at a glance.
    """
    data = collections.defaultdict(collections.Counter)
    for r in head:
        data[r["family"] or "other"][band(r)] += 1
    order = sorted(data, key=lambda f: -(data[f]["severe"] / max(1, sum(data[f].values()))))

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    ys = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for b in BAND_ORDER:
        vals = np.array([data[f][b] / max(1, sum(data[f].values())) for f in order])
        ax.barh(ys, vals, left=left, height=0.62, color=BAND_COLOR[b],
                edgecolor="white", linewidth=0.5, label=b)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 0.085:
                ax.text(l0 + v / 2, ys[i], f"{v:.0%}", ha="center", va="center",
                        fontsize=5.4, color="white" if b in ("fragile", "severe") else INK)
        left = left + vals

    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=6.3, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6.2, color=MUTED)
    ax.set_xlabel("share of family", fontsize=6.6, color=INK)
    clean(ax, grid_axis=None)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              fontsize=5.9, handlelength=1.1, handleheight=0.8,
              columnspacing=1.0, handletextpad=0.4, borderpad=0)
    title(ax, "Fragility band by task family",
          "strict score; families ordered by severely-fragile share")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. fragility distribution (single column)
# ---------------------------------------------------------------------------
def fig_distribution(head, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    bins = np.linspace(0, 0.8, 25)
    vals = [float(r["fragility_score_strict"]) for r in head]
    ax.hist(vals, bins=bins, color=ACCENT["primary"], edgecolor="white", linewidth=0.35)
    for thr, lab, c in [(0.2, "fragile", ACCENT["accent"]), (0.4, "severe", ACCENT["bad"])]:
        ax.axvline(thr, color=c, lw=0.9, ls=(0, (4, 2)))
        ax.text(thr + 0.008, ax.get_ylim()[1] * 0.93, lab, color=c, fontsize=6.2, va="top")
    ax.set_xlabel("fragility score (strict)", fontsize=6.6)
    ax.set_ylabel("environments", fontsize=6.6)
    ax.set_yscale("log")
    clean(ax, grid_axis="y")
    title(ax, "Fragility across the registry",
          f"{len(head)} environments, deliberately-adversarial excluded")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. popularity vs fragility (single column, two panels)
# ---------------------------------------------------------------------------
def fig_stars(head, out):
    """Popularity does not predict fragility.

    Two panels because there are two populations and one number is not enough:
    (a) all 1,496 headline environments, and (b) the 510 that carry a star.

    The rank correlation is computed with tie-averaged ranks. Stars are
    massively tied -- 986 environments carry none -- and breaking ties
    arbitrarily inflates the correlation, which is what an earlier draft of
    this figure printed.
    """
    from scipy.stats import spearmanr

    stars = np.array([float(r["stars"] or 0) for r in head])
    frag = np.array([float(r["fragility_score_strict"]) for r in head])
    starred = stars > 0

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.15),
                             gridspec_kw={"height_ratios": [1.15, 1]})

    # ── panel (a): the whole headline population ────────────────────────
    ax = axes[0]
    ax.scatter(stars[~starred], frag[~starred], s=5, color="#c3cad3",
               edgecolor="none", zorder=2)
    ax.scatter(stars[starred], frag[starred], s=10, color=ACCENT["primary"],
               alpha=0.5, edgecolor="none", zorder=3)
    rho_all = spearmanr(stars, frag).statistic
    xs = np.array([s for s in stars if s > 0])
    ys = np.array([f for s, f in zip(stars, frag) if s > 0])
    if len(xs) > 3:
        z = np.polyfit(stars[starred], frag[starred], 1)
        xx = np.linspace(1, max(xs), 30)
        ax.plot(xx, np.polyval(z, xx), color=ACCENT["bad"], lw=1.1, zorder=4)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.12, xs.max() * 1.25)
    ax.set_ylim(-0.02, float(frag.max()) + 0.20)
    ax.set_ylabel("fragility (strict)", fontsize=6.6, color=INK)
    ax.set_xlabel("stars", fontsize=6.6, color=INK)
    clean(ax, grid_axis="y")
    ax.text(0.97, 0.96, f"all $n$={len(head)}\n$\\rho$ = {rho_all:+.3f}",
            transform=ax.transAxes, fontsize=6.4, color=INK, va="top", ha="right")
    title(ax, "Popularity does not predict fragility",
          f"strict score; {int((~starred).sum())} of {len(head)} have no stars at all")

    # ── panel (b): the starred subset only ──────────────────────────────
    ax = axes[1]
    ax.scatter(xs, ys, s=12, color=ACCENT["primary"], alpha=0.55,
               edgecolor="none", zorder=3)
    rho_star = spearmanr(xs, ys).statistic
    if len(xs) > 3:
        z = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 30)
        ax.plot(xx, np.polyval(z, xx), color=ACCENT["bad"], lw=1.1, zorder=4)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(-0.02, float(ys.max()) + 0.18)
    ax.set_xlabel("stars", fontsize=6.6, color=INK)
    clean(ax, grid_axis="y")
    ax.text(0.97, 0.96, f"starred only, $n$={len(xs)}\n$\\rho$ = {rho_star:+.3f}",
            transform=ax.transAxes, fontsize=6.4, color=INK, va="top", ha="right")
    fig.tight_layout(pad=0.6)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. defect-mode frequency (single column)
# ---------------------------------------------------------------------------
def fig_flags(head, out):
    c = collections.Counter()
    for r in head:
        for m in modes_of(r):
            c[m] += 1
    items = c.most_common()
    # Abbreviate: full mode names push this chart past the column width,
    # and LaTeX would then shrink it below its designed type size.
    labels = [ABBREV.get(m, m.replace("_", " ")) for m, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    bars = ax.barh(range(len(labels)), vals, color=ACCENT["primary"], height=0.66,
                   edgecolor="none")
    label_bars(ax, bars, fmt="{:.0f}")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.2, color=INK)
    ax.set_xlabel("environments", fontsize=6.6, color=INK)
    ax.set_xlim(0, max(vals) * 1.18)
    clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title(ax, "Defect-mode frequency",
          f"share of the {len(head)}-environment headline population")
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/envbench/envbench_registry.csv")
    ap.add_argument("--audit", nargs="*",
                    default=["results/dynamic_audit_fragile.jsonl",
                             "results/dynamic_audit_random.jsonl",
                             "results/dynamic_audit_v1.jsonl",
                             "results/dynamic_audit_test.jsonl"])
    ap.add_argument("--out", default="../paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_style()

    rows, head = load(args.csv)
    arows = []
    seen = {}
    for p in args.audit:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seen[r["env_id"]] = r
    arows = list(seen.values())

    fig_pipeline(os.path.join(args.out, "fig_pipeline.pdf"))
    fig_cooccurrence(head, os.path.join(args.out, "fig_cooccurrence.pdf"))
    fig_funnel(arows, os.path.join(args.out, "fig_funnel.pdf"))
    fig_family_band(head, os.path.join(args.out, "fig_family_band.pdf"))
    fig_distribution(head, os.path.join(args.out, "fig_audit_distribution.pdf"))
    fig_stars(head, os.path.join(args.out, "fig_stars_vs_fragility.pdf"))
    fig_flags(head, os.path.join(args.out, "fig_audit_flags.pdf"))
    print(f"wrote 7 figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
