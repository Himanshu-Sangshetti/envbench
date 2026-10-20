---
license: mit
task_categories:
  - reinforcement-learning
  - evaluation
language:
  - en
tags:
  - rl-environments
  - verifiers
  - reward-hacking
  - evaluation-validity
---

# envbench

## Summary

A per-environment measurement of **verifier quality** across 1,614 reinforcement-learning
environments published in the Prime Intellect Environments Hub, covering 1,833 registry entries
after de-duplication by name.

Each row records whether the environment's scoring logic contains one of nine independent
**defect modes**---patterns that let a verifier accept a wrong answer or reject a right one---plus
registry metadata and a composite fragility score.

## What's in it

* `envbench_registry.csv` — 1,614 rows x 20 columns. The main table.
* `envbench_summary.json` — aggregate counts, by-family and by-verifier-type breakdowns.
* `envbench_schema.json` — machine-readable column schema.
* `figures/` — four publication figures generated from the table.

## How it was produced

Each environment's source archive was retrieved with `prime env pull` and its scoring logic was
analysed by AST traversal and targeted pattern matching. Nine defect modes were recorded and
combined with a noisy-OR into a fragility score in [0, 1]. A second, *strict* score excludes the two
modes most likely to fire on a sound verifier.

No model was used, no API key was needed, and no GPU was used. The audit is deterministic.

## Key numbers

* 46.3% of the 1,496-environment headline population carries at least one strong defect pattern.
* 13.6% carries several.
* 17.0% is flagged only by the two weakest modes --- the width of the estimate.
* Only 607 of 1,833 registry entries carry any popularity marker, and popularity does not predict
  fragility (Spearman rho = -0.030, 95% CI [-0.053, +0.049]).

## Known limitations

* The score is a **source-level screen**, not a measured false-positive rate. Our own validation
  gives AUROC 0.700 (95% CI 0.562-0.826) against a noisy label rule, and in the small dynamic
  sample obtainable, high-static-fragility environments measured *sound*.
* Families and verifier types are derived from registry tags and inherit the tags' errors.
* Environments whose purpose is adversarial detection are flagged and excluded from the headline
  population; the flag is name- and tag-based and may misclassify.
* 26 environments could not be retrieved and are absent.
* The registry is a moving target. Each row is pinned to a `version`.

## Citation

See the accompanying paper, *Audit Before You Train: A Registry-Scale Audit of Verifier Quality in
Public Reinforcement-Learning Environments*. Cite this dataset alongside it.
