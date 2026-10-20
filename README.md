# envbench

**A registry-wide measurement of verifier quality in public reinforcement-learning environments.**

This is the artifact accompanying *"Audit Before You Train: A Registry-Scale Audit of Verifier
Quality in Public Reinforcement-Learning Environments"* (Sangshetti, 2026).

> **Finding.** 46.3% of the 1,496 public RL environments that are not deliberately adversarial
> carry at least one strong verifier defect pattern, and 13.6% carry several. The registry
> publishes five independent metadata fields for every environment and none of them reveals
> whether the verifier is sound. Every number below regenerates from this bundle at $0 cost.

**Reproduce everything:** [`REPRODUCE.md`](REPRODUCE.md) &middot; **Data card:**
[`DATASET_CARD.md`](DATASET_CARD.md) &middot; **Checksums:** [`MANIFEST.txt`](MANIFEST.txt)

### Citation

```bibtex
@misc{sangshetti2026audit,
  title  = {Audit Before You Train: A Registry-Scale Audit of Verifier Quality in
            Public Reinforcement-Learning Environments},
  author = {Sangshetti, Himanshu},
  year   = {2026},
  note   = {Preprint. Artifact: envval, envbench},
  url    = {https://github.com/Himanshu-Sangshetti/envbench}
}
```

---

## What this is

Every environment in the [Prime Intellect Environments Hub](https://hub.primeintellect.ai/) has a
**verifier** — the function that decides whether a candidate solution is correct. That verifier is
what defines "correct" during training. Nobody had measured the quality of that supply.

`envbench` is a per-environment table, for **1,614 of the 1,833 environments** in the registry,
carrying:

| Column | Meaning |
|---|---|
| `env_id` | `owner/name` in the Hub |
| `hub_id`, `version` | registry identifier and the exact version audited (the table is pinned) |
| `family` | task family, derived from registry tags |
| `verifier_hint` | `rule` / `program` / `llm_judge` / `rule_or_judge`, tag-derived |
| `runtime` | `VERIFIERS_V0` / `VERIFIERS_V1` |
| `stars` | registry popularity; **zero for two thirds of the registry** |
| `train` | tagged for training |
| `tags` | pipe-separated registry tags |
| `fragility_score` | 0–1, noisy-OR over all nine defect modes |
| `fragility_score_strict` | same, excluding the two weakest modes — **the primary column** |
| `defect_modes` | pipe-separated detected defect modes |
| `n_defect_modes` | count of detected modes |
| `deliberate_adversarial` | the environment's purpose is adversarial detection; **excluded from headline rates** |
| `verifier_loc` | lines of scoring code analysed |
| `fragility_band`, `fragility_band_strict` | `clean` / `mild` / `fragile` / `severe` |
| `static_only` | TRUE — the score is a source-level proxy, **not** a measured false-positive rate |

## Headline numbers (1,496-environment headline population)

| Band | Strict score (5 strong modes) | Full scorer (9 modes) |
|---|---|---|
| clean | 803 (53.7%) | 459 (30.7%) |
| fragile (≥ 0.2) | 489 (32.7%) | 553 (37.0%) |
| severely fragile (≥ 0.4) | **204 (13.6%)** | 402 (26.9%) |

**46.3% of public RL environments carry at least one strong verifier defect pattern.** 17.0% are
flagged only by the two weakest modes, which is the width of the estimate.

## The nine defect modes

| Mode | Detects | Why it matters |
|---|---|---|
| `substring_match` | containment test on the candidate string | accepts a near-miss containing the answer |
| `startswith_endswith` | prefix/suffix matching | accepts a partial answer |
| `set_membership` | membership in a literal set | accepts any listed variant, including a wrong one |
| `exact_string_eq` | string equality with no `lower()`/`strip()` | fails on formatting, rewards on surface match |
| `float_eq_no_tol` | float equality without `isclose` | brittle to legitimate numeric variation |
| `regex_num_extract` | regex extraction of a number | extraction errors become reward errors |
| `hardcoded_answers` | literal expected answers in source | unreviewable ground truth |
| `judge_no_ref` | an LLM judge never anchored to a reference | a fluent wrong answer can be rewarded |
| `no_execution` | a test suite referenced but the candidate never executed | the verifier cannot be checking behaviour |

## Important caveats

* **`static_only` is true.** This is a *screen*, not a measured false-positive rate. Our own
  instrument validation gives AUROC 0.700 (95% CI 0.562–0.826) against a noisy label rule, and in
  the small dynamic sample we could obtain, environments with high static fragility measured
  *sound*. Use this table to prioritise, not to certify.
* **Families and verifier types are tag-derived** and inherit the tags' error. 503 of 1,496
  environments sit in `other`.
* **`deliberate_adversarial` uses name patterns and tags.** An adversarial environment with an
  innocuous name is counted in the headline population, biasing it upward.
* **26 environments failed to retrieve** and are absent, biasing the rate downward.
* **The registry changes.** Each row is pinned to a `version` so results are reproducible.

## Using it

```python
import pandas as pd
df = pd.read_csv("envbench_registry.csv")

# the fragile environments, for a targeted audit
worst = df[(~df["deliberate_adversarial"]) &
           (df["fragility_score_strict"] >= 0.4)].sort_values(
               "fragility_score_strict", ascending=False)
print(len(worst), "severely fragile environments")
```

## Reproducing it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

python scripts/build_registry.py --hub results/hub/environments.json
prime login                                    # public envs need no auth; install does
python scripts/audit_registry.py --limit 2000 --workers 14
python scripts/build_envbench.py --audit results/registry_audit_v2.jsonl
```

The audit is deterministic: it reads source, applies fixed syntactic rules, and emits the same
scores on every run. About 6 seconds per environment, CPU only, no API keys, no GPU.

## License

MIT. The underlying environments belong to their authors and carry their own licenses; this table is
metadata about them, not redistribution of them.
