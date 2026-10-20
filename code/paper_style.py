"""Shared visual system for the paper's figures.

Design rules, all enforced here so no figure can drift:

1. **White background.** The paper is white. A dark figure in a white paper prints as a
   black rectangle and reads as broken.
2. **One palette.** Magnitude uses a single sequential ramp; categorical series use the
   same four accents everywhere. No default matplotlib blue.
3. **Built at final size.** Figures are generated at the width they occupy in the
   document (3.25in for a single column, 6.75in for a full-width float), so nothing is
   scaled afterwards and the type size on the page is what we set.
4. **Serif labels** to match the body text.
5. **No chartjunk**: no top/right spines, no gridlines over data, thin axes, direct
   labelling where it fits.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- page geometry (ICML 2025) ---------------------------------------------
COL_W = 3.25        # inches, single column
FULL_W = 6.75       # inches, two-column float
FIG_H = 2.05        # default height for a full-width figure
COL_H = 2.45        # default height for a single-column figure

# --- palette ----------------------------------------------------------------
INK = "#1a1a1a"        # primary text
MUTED = "#6b7280"      # secondary text
HAIR = "#d4d4d4"       # axis / gridlines
PANEL = "#f4f4f5"      # panel fill

ACCENT = {
    "primary": "#1f4e79",   # deep blue   - primary series
    "accent": "#b45309",    # burnt amber - emphasis
    "good": "#2f6f4e",      # green       - sound / clean
    "alt": "#6b4c9a",       # purple      - third series
    "bad": "#a52a2a",       # red         - defect / severe
}

# sequential ramp for magnitude: light -> the deep blue
SEQ = LinearSegmentedColormap.from_list(
    "paperseq", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
)
# sequential ramp for a 2-axis categorical heat
SEQ2 = LinearSegmentedColormap.from_list(
    "paperseq2", ["#f7f7f7", "#d9e2ec", "#9fb3c8", "#5b7c99", "#2c3e50"]
)

BASE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": HAIR,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "axes.titleweight": "bold",
    "legend.fontsize": 6.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.frameon": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,     # embed as Type 42 so text stays selectable
    "ps.fonttype": 42,
}


def use_style():
    plt.rcParams.update(BASE_RC)


def clean(ax, grid_axis: str | None = "y", spines=("top", "right")):
    """Strip chartjunk."""
    for s in spines:
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        if s in ax.spines:
            ax.spines[s].set_color(HAIR)
    if grid_axis:
        ax.grid(True, axis=grid_axis, lw=0.4, color=HAIR, alpha=0.7)
        ax.set_axisbelow(True)
    return ax


def title(ax, text, sub=None):
    """Left-aligned title with an optional subtitle beneath it.

    The subtitle is anchored to the axes, the title to the figure, and the gap
    between them is set explicitly so the two can never overlap.
    """
    if sub:
        ax.set_title(text, loc="left", color=INK, pad=17)
        ax.text(0, 1.008, sub, transform=ax.transAxes, fontsize=6.4,
                color=MUTED, va="bottom", ha="left")
    else:
        ax.set_title(text, loc="left", color=INK, pad=4)
    return ax


# Ordinal ramp for the four fragility bands: pale (clean) -> red (severe).
# One ramp, one meaning, used in every figure.
BAND_COLOR = {
    "clean": "#cfe0ee",
    "mild": "#8fb8d8",
    "fragile": "#c07b2b",
    "severe": "#a52a2a",
}
BAND_ORDER = ["clean", "mild", "fragile", "severe"]


def label_bars(ax, bars, fmt="{:.0f}", dy=None, color=INK, size=6.4):
    """Direct value labels on horizontal bars."""
    top = max((b.get_width() for b in bars), default=1)
    dy = dy if dy is not None else top * 0.012
    for b in bars:
        ax.text(b.get_width() + dy, b.get_y() + b.get_height() / 2,
                fmt.format(b.get_width()), va="center", ha="left",
                fontsize=size, color=color)
    return ax
