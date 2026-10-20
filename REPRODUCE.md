# Reproducing every number in the paper

Every quantity quoted in *Audit Before You Train* can be regenerated from this bundle.
Cost: **$0**. No GPU, no API key, no model calls.

```
envbench_registry.csv      1,614 rows, one per registry environment
envbench_schema.json       column definitions and types
envbench_summary.json      the released headline summary
analysis/                  derived tables and figures
code/                      the audit library and the verification scripts
registry_generated.yaml    the registry snapshot the audit was run against
preregistration.md         the preregistered companion study (Paper 2)
MANIFEST.txt               sha256 of every file in this bundle
```

## 1. Check the bundle is intact

```sh
shasum -a 256 -c MANIFEST.txt
```

## 2. Re-verify the paper's numbers against the data

```sh
cd code
python verify_paper_numbers.py --paper ../../main.tex --csv ../../envbench_registry.csv
```

This recomputes every headline number in the manuscript from `envbench_registry.csv`
and fails on any mismatch. It is the same check the paper's build gate runs.

## 3. Re-verify the reference list

```sh
python verify_refs.py --bib ../../refs.bib
```

Fetches every URL in the bibliography and reports any that are dead or synthetic.
The paper's build gate runs this too: no reference may be cited unless it resolves.

## 4. Re-derive the popularity analysis

```sh
python analyze_stars.py --csv ../../envbench_registry.csv --out stars_analysis.json
```

Recomputes the Spearman correlations between registry stars and fragility, with
tie-averaged ranks, and the family-clustered bootstrap intervals quoted in Table 5.

## 5. Re-validate the audit instrument

```sh
python validate_instrument_stats.py --registry ../../registry_generated.yaml
```

Re-audits the labelled set under both the current (presence-graded) and the legacy
(count-graded) severity rule, and reports AUROC with a stratified bootstrap interval
for each, which is the before/after recalibration reported in Section 3.5.

## 6. Rebuild the figures

```sh
python make_figures.py --csv ../../envbench_registry.csv --out ../../figures
python fig_widths.py --paper ../../main.tex --write
```

Figures are generated at their final printed size and placed at 1:1, so the type
size on the page is the type size that was designed.

## Requirements

Python 3.11+, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `pyyaml`.
The full paper gate additionally needs `tectonic` and `pdftoppm` (poppler).
