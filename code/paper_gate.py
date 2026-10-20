"""The paper quality gate — one command, one verdict.

    python scripts/paper_gate.py

Runs the whole pipeline and reports PASS/FAIL on every dimension that decides
whether the paper is submittable:

  1. figures regenerate cleanly
  2. the document compiles with no errors
  3. no undefined citations or references
  4. every number in the paper matches the released data
  5. no stale denominators
  6. every referenced figure exists and is non-trivial in size
  7. overfull boxes below a threshold
  8. the body contains the sections a reviewer expects
  9. every page renders (a render pass catches figures that break LaTeX)

Exit code 0 = the gate passes. Anything else names the next thing to fix.
This is the loop: run it, fix what it names, run it again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVV = HERE.parent                      # .../envvalue
PROJECT = ENVV.parent                  # .../Default Project
PAPER = PROJECT / "paper"
PAPER2 = PROJECT / "paper2"
TEX = "tectonic"


def sh(cmd, cwd=None, timeout=1800):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


class Gate:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        return ok

    def report(self):
        npass = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{npass}/{len(self.results)} gate checks passed")
        fails = [(n, d) for n, ok, d in self.results if not ok]
        if fails:
            print("\nNEXT ACTIONS")
            for n, d in fails:
                print(f"  - {n}" + (f"  ({d})" if d else ""))
            return 1
        print("\nGate passed. The paper is ready for an arXiv v1.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()
    g = Gate()

    print("=" * 74)
    print("PAPER QUALITY GATE")
    print("=" * 74)

    # 1. figures
    rc, out = sh([sys.executable, str(HERE / "make_figures.py"),
                   "--out", str(PAPER / "figures")], cwd=str(ENVV))
    g.check("figures regenerate", rc == 0, out.strip().splitlines()[-1] if rc else "")

    # 1b. figures are placed at 1:1 so their type size on the page is the
    #     size that was designed (LaTeX would otherwise shrink them).
    rc, out = sh([sys.executable, str(HERE / "fig_widths.py"),
                   "--paper", str(PAPER / "main.tex"), "--write"], cwd=str(ENVV))
    g.check("every figure is placed at its natural width", rc == 0,
            out.strip().splitlines()[-1][:110] if rc else "")
    texnow = (PAPER / "main.tex").read_text(errors="ignore")
    scaled = re.findall(r"\\includegraphics\[width=(0\.\d+|\\\w+)", texnow)
    g.check("no figure is scaled down by LaTeX", not scaled, str(scaled))

    # 2/3. compile
    for label, pdir in (("paper 1", PAPER), ("paper 2", PAPER2)):
        if not (pdir / "main.tex").exists():
            continue
        rc, out = sh([TEX, "main.tex", "--keep-logs", "--keep-intermediates"],
                     cwd=str(pdir))
        log = (pdir / "main.log").read_text(errors="ignore") if (pdir / "main.log").exists() else ""
        g.check(f"{label}: compiles", rc == 0 and "! " not in log.split("Output written")[0],
                "" if rc == 0 else out[-200:])
        und = len(re.findall(r"(Citation|Reference).*undefined", log))
        g.check(f"{label}: no undefined refs/citations", und == 0, f"{und} undefined")
        pages = re.search(r"Output written on .*\((\d+) pages", log)
        g.check(f"{label}: page count", pages is not None,
                f"{pages.group(1)} pages" if pages else "")

    mainlog = (PAPER / "main.log").read_text(errors="ignore")
    maintex = (PAPER / "main.tex").read_text(errors="ignore")

    # 4/5. numbers
    rc, out = sh([sys.executable, str(HERE / "verify_paper_numbers.py"),
                   "--paper", str(PAPER / "main.tex")], cwd=str(ENVV))
    g.check("every quoted number matches the released data", rc == 0,
            "" if rc == 0 else out.strip().splitlines()[-1][:120])

    # 6. figures referenced exist
    refs = sorted(set(re.findall(r"figures/([a-z0-9_]+\.pdf)", maintex)))
    missing = [r for r in refs if not (PAPER / "figures" / r).exists()]
    g.check("every referenced figure exists", not missing, f"{len(refs)} referenced"
            + (f", missing {missing}" if missing else ""))
    tiny = [r for r in refs if (PAPER / "figures" / r).exists()
            and (PAPER / "figures" / r).stat().st_size < 4000]
    g.check("no suspiciously small figures", not tiny, str(tiny))
    on_disk = {p.name for p in (PAPER / "figures").glob("*.pdf")}
    unused = sorted(on_disk - set(refs))
    g.check("no orphan figures on disk", not unused, str(unused) if unused else "")

    # 6b. a figure wider than the column must live in a full-width float
    tlines = maintex.split("\n")
    bad_float = []
    for i, l in enumerate(tlines):
        m = re.search(r"\\includegraphics\[width=([\d.]+)in\]", l)
        if not m:
            continue
        w = float(m.group(1))
        if w <= 3.30:
            continue
        env = None
        for j in range(i - 1, max(-1, i - 60), -1):
            if "\\begin{figure" in tlines[j] or "\\begin{table" in tlines[j]:
                env = tlines[j].strip()
                break
        if env and env.endswith("[t]") and "*" not in env:
            bad_float.append(f"{w:.2f}in in {env}")
    g.check("wide figures are in full-width floats", not bad_float, str(bad_float))

    # 7. overfull
    ov = len(re.findall(r"Overfull \\hbox", mainlog))
    worst = 0.0
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)", mainlog):
        worst = max(worst, float(m.group(1)))
    g.check("overfull boxes under 35pt", ov == 0 or worst <= 35.0,
            f"{ov} boxes, worst {worst:.1f}pt")

    # 8. required body sections
    required = ["Experimental Setup", "Discussion", "Limitations", "Reproducibility",
                "Conclusion", "Related Work"]
    lower = maintex.lower()
    missing_sections = [s for s in required if s.lower() not in lower]
    g.check("body has the expected sections", not missing_sections, str(missing_sections))

    # 8b. citation hygiene: every bib entry is cited, and every citation resolves
    bib = (PAPER / "refs.bib").read_text(errors="ignore") if (PAPER / "refs.bib").exists() else ""
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    citekeys = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", maintex):
        citekeys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    orphans = sorted(bibkeys - citekeys)
    g.check("no uncited references in the bibliography", not orphans,
            f"{len(bibkeys)} entries, {len(orphans)} uncited" + (f": {orphans}" if orphans else ""))
    dangling = sorted(citekeys - bibkeys)
    g.check("no citation without a bibliography entry", not dangling, str(dangling))
    g.check("bibliography is substantial", len(bibkeys) >= 40,
            f"{len(bibkeys)} references")

    # 9. render every page
    if not args.skip_render:
        outdir = Path("/tmp/envval_render")
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True)
        if shutil.which("pdftoppm"):
            rc, out = sh(["pdftoppm", "-png", "-r", "60", str(PAPER / "main.pdf"),
                           str(outdir / "p")], timeout=600)
            npages = len(list(outdir.glob("p-*.png")))
            g.check("every page renders", rc == 0 and npages > 0, f"{npages} pages rendered")
        else:
            g.check("every page renders", False, "pdftoppm not available")

    # 10. every float is referred to in the body
    labels = re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", maintex)
    body = re.sub(r"\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\}",
                  "", maintex, flags=re.S)
    unref = []
    for lab in labels:
        pat = r"\\(?:ref|cref|Cref|autoref)\{" + re.escape(lab) + r"\}"
        if not re.search(pat, body):
            unref.append(lab)
    g.check("every table and figure is referenced in the body", not unref, str(unref))

    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
