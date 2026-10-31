# Pre-registration: Selection or Discovery?

Committed 31 Oct 2026, before any run of the `sd/` code that will be reported. The Kaggle smoke
tests (3 steps, 128 tokens) only check that the code runs. The 0.6B pilot on Kaggle is
exploratory and is reported as such.

## Question

When RL with a defective checker learns a hack, is the hack selected from answers the starting
model already writes, or discovered by climbing a slope in the reward? Can audits of the starting
model, made before training, tell which will happen?

## Audit numbers (measured on the starting model, before training)

- **Mass h**: share of starting-model answers the checker pays (reward > 0.5) and Math-Verify
  rejects.
- **Slope s**: mean rise in the checker's payout when each edit in `checkers.EDITS` is applied to a
  wrong starting-model answer. The edit library is frozen in this commit.
- **Novelty n**: per-token log-likelihood of a late hack answer (steps >= 75) under the starting
  model, as a percentile of the starting model's own answers.

## Outcomes

- **fp**: per step, share of completions the checker pays (> 0.5) while Math-Verify rejects them.
- **Exploitation**: mean fp over the last 20 steps minus mean fp over the first 10.
- **Exploited**: exploitation >= 0.10.
- **Onset**: first step where the 5-step mean fp exceeds 0.30. SimHack's definition (half of
  rollouts hack) is reported too.

## Hypotheses (confirmatory)

| # | Hypothesis | Pass threshold |
| --- | --- | --- |
| H1 | Hacks against cliff checkers are selection, hacks against judges are discovery | Novelty separates late hack answers of the two groups with AUROC >= 0.80 |
| H2 | Mass h predicts exploitation for cliff checkers | Spearman >= 0.60 between h and mean exploitation across seeds |
| H3 | h fails for judges, s works | Spearman for h < 0.30 and for s >= 0.60, over judge conditions |
| H4 | The combined audit beats existing audits on real environments | AUROC for "exploited" above the static flag and fuzzing by >= 0.10 |
| H5 | h is a risk level, not a verdict | Logistic slope of per-seed exploited on h > 0, 95% CI excludes 0 |

H6 (exploratory): at larger models, selection hacks shrink with h while discovery hacks persist.

Causal test (confirmatory, part of H3): `substring_tail50` alone (low h, no slope) is not
exploited, and `tail50_slope` (the same cliff plus partial credit) is exploited, in at least 2 of
3 seeds at each model size run.

## Design

- Task: MATH training problems, levels 1-3, integer answers of at least two digits. Held-out
  checks on MATH-500.
- Algorithm: TRL 1.17.1 GRPOTrainer, `loss_type="dr_grpo"`, `scale_rewards="none"`, beta 0,
  8 questions x 8 answers per step, temperature 1.0, LoRA rank 16 on all linear layers, learning
  rate 5e-6 constant, 512-token answers (1,024 at 4B).
- Conditions: `checkers.CONDITIONS`. Judges: Qwen3-1.7B at 0.6B/1.7B policies; a 7-8B judge is an
  added condition, not a replacement.
- Models: Qwen3-0.6B and Qwen3-1.7B, 3 seeds; Qwen3-4B, key conditions, 2-3 seeds.
- Steps: 120.

## Analysis

Bootstrap 95% intervals over conditions (10,000 resamples). All runs are reported, including
failed ones. A run that crashes is rerun once with the same seed; if it crashes again it is
reported as failed. No run is dropped after its result is seen.

## Amendment 1 (31 Oct 2026, before any main run)

Made after an independent review of the method and before any run that will be reported. The
0.6B Kaggle pilot launched earlier the same day used the original code and stays exploratory.

1. **Gold label.** Math-Verify on the last boxed answer, and the reply must give one distinct
   boxed answer (`checkers.strict`). Math-Verify on the whole reply falls back to the last
   expression, which marks a reply that ends by dumping the gold as correct and would hide the
   tail and substring hacks. The whole-reply version is logged as `strict_mv_any`, and
   Math-Verify exceptions are counted.
2. **Primary selection-vs-discovery measure: base reachability.** For each run's late hack
   answers (steps >= 75), the starting model draws K = 64 answers on the same problems. Reach is
   the share of those problems where at least one base answer is paid-but-wrong under the same
   checker. Selection predicts high reach, discovery low reach. Novelty becomes secondary, because
   per-token likelihood rewards repetitive spam and shifts with RL sharpening.
3. **H1 restated.** Reach is lower for conditions whose measured slope s is above the median than
   for those below it (one-sided permutation test over conditions, p < 0.05). Grouping is by
   measured s, not by checker family.
4. **Slope.** s is reported net of neutral edits (`checkers.NULL_EDITS`), in reward units and as the
   change in P(pay > 0.5). The edit library was fixed after seeing the earlier MLX pilot's judge
   hacks; this is disclosed, and s is therefore treated as descriptive, not confirmatory.
5. **Causal test.** Three conditions with 8 seeds each at 0.6B: `substring_tail50`, `tail50_slope`,
   and a density-matched control `tail50_density` (the same 0.2 credit paid on a fixed hash of the
   reply, at the base rate of "gold appears outside the last 50 characters", measured on the base
   samples before the main runs). Prediction: exploitation in `tail50_slope` is higher than in both
   others (one-sided permutation tests on continuous exploitation, p < 0.05 each).
6. **Outcomes.** Co-primary: reward-weighted fp (`fp_mass`, the reward paid to answers the gold
   rejects) alongside fp at 0.5. Exploitation is also reported normalised by headroom,
   (last - first) / (1 - first).
7. **Statistics.** H2 is tested with a mixed model (exploitation ~ h + (1 | condition)); H3, H4 and
   H5 become exploratory. The per-seed logistic in H5 is dropped as pseudo-replication; the
   seed-level starting fp (`h_seed`) is reported instead.
8. **Conditions.** `judge_noref_*` answers YES to almost everything in the self-test, so one
   `judge_noref_bin` seed is kept as a sanity row and excluded from all tests. `judge_ref_bin` is
   cut.
9. **Scaling ablation.** `substring_tail50`, `tail50_slope` and `judge_ref_prob` are rerun with
   group-standardised rewards (`--scale-rewards group`), 3 seeds each at 0.6B, because Dr. GRPO's
   unscaled advantages differ in size across reward types.
10. **Code.** Runs refuse to write into a directory that already has a log; the analysis lists
    every run, including failed and incomplete ones; samples are saved every 5 steps from step 50;
    one model per audit directory.

## Amendment 2 (31 Oct 2026, before any main run)

Made after the base audit and a judge calibration on the starting model's answers, before any run
that will be reported.

1. **Judge.** Qwen3-4B with the original prompt (v1) replaces Qwen3-1.7B. On 768 base answers the
   1.7B judge, shown the reference, paid 96% of boxed wrong answers (h = 0.21), so it cannot test
   discovery. Qwen3-4B v1 with the reference: h = 0.022, accepts 99.8% of correct answers, pays
   1.7% of boxed wrong answers. A stricter prompt (v2) was worse and is not used.
2. **Judge conditions.** `judge_ref_prob` and `judge_noref_prob` (Qwen3-4B; without the reference
   h = 0.035, accepts 80% of correct answers) are both confirmatory conditions. The earlier
   `judge_noref_bin` sanity row is dropped. The 0.6B Kaggle pilot's judge run used the 1.7B judge
   and is reported as an exploratory example of a judge that is fooled from the start.
3. **Density control.** `DENSITY_RATE` = 0.066, the base rate of "gold appears outside the last 50
   characters" over 2,048 Qwen3-0.6B answers.
4. **Slope.** An edit counts towards s only if the edited answer is still rejected by the gold, so
   payout gained by fixing an answer is not counted as slope.

## Amendment 3 (31 Oct 2026, 18:40 IST, before the calibration results are read)

The 0.6B pilot on the TRL pipeline (learning rate 5e-6) learned almost nothing in 120 steps:
true accuracy rose about 3 points in every condition, including the strict control. Its
effective update is roughly ten times smaller than the earlier MLX pilot's (MLX LoRA scale 20 vs
TRL alpha/r = 2, and Adam without bias correction). A learning-rate calibration is running on
`substring_all` at 2e-5, 5e-5 and 1e-4 (unscaled rewards) and 5e-5 (group-scaled), 80 steps.

**Selection rule, fixed now:** choose the largest learning rate whose run (a) does not collapse
(held-in true accuracy never falls more than 10 points below its first-10-step mean, and the
answers stay readable), and (b) raises true accuracy over the run. The paid-but-wrong rate (fp)
in these runs is **not** used to choose, and the calibration runs are not reported as results.
If group-scaled rewards are needed for (b) while unscaled are not, the main runs use group
scaling and the unscaled runs become the ablation; this swap is reported.
