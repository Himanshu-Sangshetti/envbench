# Preregistration — Transfer Study

**Study:** Do broken verifiers break the model, or only the score?
**Version:** 1.0 · **Frozen:** 20 Oct 2026 (this file is version-controlled; the commit hash is the timestamp)
**Paper:** Paper 2, companion to *Audit Before You Train*
**Status:** REGISTERED — no training has been run against this protocol.

> **Why preregister?** Of the entire 2026 frontier of agentic-RL work we surveyed, **one** study
> preregisters anything. It is the cheapest available differentiator, and it forecloses the
> "you looked at the results and then chose the hypothesis" objection. This document is written
> **before** any run and is not to be edited except to add an amendment, which must be dated,
> justified, and reported in the paper.

---

## 1. Motivation and prior art

Three results already exist and this study is designed around them:

1. **A static audit of a verifier's source predicts the mass of falsely-rewarded solutions**
   (MBPP vs. MBPP+, preregistered two-arm contrast; Spearman 0.80 between a cheap pre-training
   leakiness audit and rewarded FP mass; arXiv:2607.11022).
2. **The false-positive mass of a real benchmark is large and consequential**: 28.5% of sampled
   SWE-bench Verified tasks accept a Docker-verified wrong patch, worth +14.14 pp Pass@1 across 134
   frontier submissions (arXiv:2606.16062).
3. **A competitor has named this study as its own next step**: *"a direct causal test (training on
   fixed-versus-untouched broken tasks at fixed compute) is the natural next step"* (arXiv:2606.16062).

**What is still unknown:** whether verifier breakage degrades the capability a model *transfers* to
other environments, or only inflates the score it is measured on. Every prior measurement targets
reward. Reward is trivially gameable and already measured. **Transfer is the quantity that decides
whether an environment was worth training on**, and it is unmeasured.

**Our gap, stated precisely:** a *causal* study of verifier quality with **cross-environment
transfer** as the dependent variable, over a set of environments drawn from a public registry, with
a pre-registered analysis.

---

## 2. Hypotheses

| ID | Hypothesis | Direction |
|---|---|---|
| **H1** | A verifier that accepts near-miss solutions reduces the **transfer** gain of the environment it is attached to, relative to a hardened verifier on identical tasks, at identical compute. | negative |
| **H2** | The reduction is larger for environments with a higher static fragility score (dose–response). | monotonic |
| **H3** | An isomorphic-but-differently-implemented verifier (same semantics, different form) changes measured learning gain. | any change |
| **H4** | *Null hypothesis:* none of H1–H3 holds if and only if the naive baselines — difficulty, outcome entropy, environment count — predict transfer at least as well as our three components do. | — |

**H4 is a genuine falsification condition, not a formality.** If a predictor cannot beat "pick the
environment with the most tasks" or "pick the hardest one", the whole decomposition is decorative,
and we will report that.

---

## 3. Design

### 3.1 Independent variable: the verifier arm

Three arms, run on **identical tasks, identical seeds, identical compute**:

| Arm | Verifier | Purpose |
|---|---|---|
| `as_shipped` | the environment's own verifier | baseline |
| `hardened` | near-miss acceptance removed; graded via `rewardkit`'s `diff_ratio` or RLVE's `--reward-key reward\|accuracy` | the dose–response axis (H1, H2) |
| `isomorphic` | same semantics, different implementation | does the *form* matter? (H3) |

### 3.2 Dependent variable: transfer gain

```
Δ_transfer(E, M, s) = mean over held-out families F' of
                      [ score( policy_after(E, M, s), F' ) − score( base_M, F' ) ]
```

* **Held-out families only.** The evaluation set is disjoint from the training set at the *family*
  level, never merely at the task level.
* **The evaluation verifier is fixed and human-audited** before the study, so the dependent
  variable cannot be gamed by the policy under test.
* **Base-model scores are subtracted** so pretraining capability does not contaminate the target.
* **In-environment score is recorded but is not the target.** It is the naive baseline.

### 3.3 Design matrix

* **6 training environments**, one from each of: math, code, tool_use, instruction, games,
  reasoning. Chosen so no two share a family.
* **3 verifier arms** × 6 environments = 18 cells.
* **3 seeds** per cell = 54 runs.
* **400 steps**, 1.5B base model, Dr.GRPO (length-controlled, so length bias cannot confound the
  environment comparison), 8 completions per group.
* **One family is held out entirely** as a confirmatory test and is not used during any model
  selection. It is recorded here before the first run: `____` (to be filled and committed before
  the first run, not after).

### 3.4 What is fixed across every cell

Base model, tokenizer, step count, batch size, learning rate, reward shaping, harness, prompt
template, task subsets, and evaluation suite. **The environment and the verifier are the only things
that vary.** Any change to a fixed quantity invalidates the preregistration and must be logged as an
amendment.

---

## 4. Analysis plan (fixed in advance)

### 4.1 Primary endpoint

**LOFO R²** — variance in transfer gain explained by the predictors when a whole environment
*family* is held out, pooled over folds.

**Secondary:** Spearman ρ for ranking environments; the mixed-effects coefficient on the verifier
arm with environment as the clustering unit.

### 4.2 Predictors

The three components (verifier informativeness, learning-signal diversity, transfer radius), each a
composite built from the pre-training statistics, z-scored by median/MAD across the pool and averaged
over present members.

### 4.3 Baselines that must be beaten

| Baseline | Why |
|---|---|
| `base_difficulty` | the field's default proxy |
| `B4_outcome_entropy` | the second most-used proxy (RAGEN-2) |
| `n_tasks` | **RLVE's own lever** — if simply providing more tasks wins, our contribution evaporates |
| `in_environment_score` | what a practitioner without our tooling can compute |

### 4.4 Variance and power

* The **independent unit is the environment**, clustered by family; seeds are repeats, not
  replicates.
* Within-group verifier-error correlation is **ρ = 0.530 (CI 0.500–0.560)**, so an 8-completion
  GRPO group has an **effective n of 1.70, not 8**. We report the Kish design effect for family
  clustering and the effective n for group sampling, and both appear in the paper.
* We do **not** commit to a significance threshold before seeing data. We commit to reporting
  effect sizes with cluster-robust intervals, and to reporting the result whatever it is. H1–H3 are
  reported as confirmed / not confirmed / inconclusive, never as "suggestive".

### 4.5 Multiplicity

* The predictor set is fixed in §4.2 and preregistered here.
* All primary analyses are reported; no selection among them after the fact.
* Family-level splits are fixed; no post-hoc re-partitioning.
* Any exploratory analysis is labelled exploratory in the paper.

### 4.6 Exclusion rules (fixed)

* A run is excluded only on a **mechanical** failure (non-zero exit, missing checkpoint, divergent
  loss) recorded in the run log. A run with a low score is **never** excluded.
* An environment is excluded from the primary analysis only if its verifier cannot be executed at
  all, and the exclusion and its reason are reported.

---

## 5. What we will report regardless of the outcome

* The full transfer matrices, not just headline numbers.
* Negative and null results as first-class findings, with the same prominence as positive ones.
* The auditability taxonomy: environments we could not validate, and why.
* The measured vs. static disagreement, if it persists, as a limitation of our own instrument.
* Every amendment to this document, dated.

---

## 6. Artifacts to be released at submission

| Artifact | Contents |
|---|---|
| `envval` | the static audit library |
| `envbench` | the registry-wide statistics table |
| `transfer-analysis` | the analysis code, including the LOFO and mixed-effects pipeline |
| `preregistration` | this document, with the commit hash |
| `traces` | persisted rollout traces, redacted where necessary, sufficient to re-run the analysis without re-training |

---

## 7. Amendments

| # | Date | Change | Justification |
|---|---|---|---|
| — | — | — | — |
