"""Keep every figure at exactly 1:1 with the page.

LaTeX scales a PDF to the width you ask for. If you ask for less than the
figure's natural width, its type shrinks on the page -- which is how a 6.2pt
label silently becomes a 4.4pt label. So the rule here is:

    the \\includegraphics width == the figure's natural width, always.

This script measures each figure PDF and rewrites the widths in the manuscript
to match, so the type size on the page is always the size that was designed.
Run it after regenerating figures; the quality gate runs it automatically.

    python scripts/fig_widths.py --paper ../paper/main.tex
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def natural_width_in(pdf: Path) -> float | None:
    """Page width of the PDF in inches, via pdfinfo."""
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception:
        return None
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out)
    if not m:
        return None
    return float(m.group(1)) / 72.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="../paper/main.tex")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    paper = Path(args.paper).resolve()
    tex = paper.read_text()
    figdir = Path(args.figdir).resolve() if args.figdir else paper.parent / "figures"

    pat = re.compile(
        r"(\\includegraphics\[width=)([^\]]+)(\]\{figures/)([A-Za-z0-9_]+)(\.pdf)?(\})")
    changed, report = 0, []

    def sub(m):
        nonlocal changed
        pre, cur, mid, name, ext, post = m.groups()
        pdf = figdir / f"{name}.pdf"
        if not pdf.exists():
            return m.group(0)
        w = natural_width_in(pdf)
        if w is None:
            return m.group(0)
        new = f"{w:.2f}in"
        report.append((name, cur, new, w))
        if cur.strip() != new or not ext:
            changed += 1
        return f"{pre}{new}{mid}{name}.pdf{post}"

    out = pat.sub(sub, tex)
    for name, old, new, w in report:
        flag = "  " if old.strip() == new else "->"
        print(f"  {flag} {name:26s} {old:>16s} -> {new:>10s}   (page {w:.2f}in)")

    if changed and args.write:
        paper.write_text(out)
        print(f"\n✓ rewrote {changed} widths in {paper.name}")
    elif changed:
        print(f"\n{changed} widths differ; re-run with --write to apply")
    else:
        print("\nall figures already at 1:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
