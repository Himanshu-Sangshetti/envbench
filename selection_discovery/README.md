# Format drift masquerades as reward hacking

Code, pre-registration, data and paper for a small pre-registered study of when RL exploits flawed
checkers. Paper: [`paper/main.pdf`](paper/main.pdf).

**Question.** When RL with a flawed checker learns a hack, is the hack *selected* from behaviour the
starting model already shows, or *discovered* by following a slope in the reward?

**Setup.** Qwen3-0.6B, Dr. GRPO (TRL 1.17.1), 120 steps × 64 answers, 1,536-token answers, LoRA r16,
lr 1e-4, no KL. 20 runs on Modal L40S against six maths checkers (MATH levels 1-3, integer answers)
and two code checkers (MBPP original tests vs MBPP+). Total cloud cost about $38.

## Results

1. **The referee failed under training.** Models drift into answer formats a fixed grader cannot read
   (nested and bold boxes, words inside boxes, repeated "Final Answer" lines). A box-required referee
   labelled up to 96% of correct paid answers as hacks in one run; a headline exploitation of +0.43
   became 0.00 under a robust last-box referee.
2. **Real exploitation was confined to loopholes easier than the task.** `substring_all` (pays if the
   gold number appears anywhere): +0.42 in one of two seeds, true accuracy 0.73 → 0.43, by writing
   long replies that never conclude. MBPP original tests: +0.01 to +0.06. Controls stayed clean.
3. **Mostly selection.** In those runs the starting model already produced paid-but-wrong answers on
   54-92% of the problems where the trained model did (64 samples per problem).
4. **The pre-registered causal test was null.** A partial-credit slope towards a narrow loophole did
   not raise exploitation over the loophole alone or a density-matched random-credit control (4 seeds
   each, one-sided permutation p = 0.74 and 0.70).

| Run | Exploitation (final referee) | Exploitation (box-required referee) | Accuracy start → end |
|---|---|---|---|
| substring_all s0 | **+0.42** | +0.45 | 0.73 → 0.43 |
| substring_all s1 | +0.07 | +0.59 | 0.62 → 0.64 |
| judge_ref_prob s0 | +0.03 | +0.53 | 0.73 → 0.71 |
| tail50_slope s0-s3 | +0.00, +0.00, +0.01, −0.04 | +0.00, +0.43, +0.02, −0.01 | mostly up |
| substring_tail50 s0-s3 | +0.02, +0.00, +0.02, −0.04 | +0.02, +0.07, +0.02, −0.04 | mostly up |
| tail50_density s0-s3 | −0.01, +0.00, +0.03, −0.02 | −0.01, −0.00, +0.02, −0.02 | mostly up |
| strict s0 | 0.00 | 0.00 | 0.70 → 0.71 |
| mbpp_base s0 / s1 / s2 | +0.06 / +0.05 / +0.01 | same | ~0.24 → ~0.49 |
| mbpp_plus s0 | 0.00 | same | 0.25 → 0.57 |

**Limitations.** One small model and short training; the final referee was chosen after seeing
results (disclosed as Amendment 9, all versions reported, reviewed for leniency); the judge has one
seed. See the paper.

## Layout

- `PREREG.md`: pre-registration with nine dated amendments (Amendment 9 is post hoc and marked so).
- `checkers.py`: checkers and the referee (`gold_label`: last box holding a number).
- `code_env.py`: MBPP / MBPP+ environment and sandbox.
- `train_grpo.py`: one training run (resumable; `--report wandb`).
- `audit.py`: base audits, reachability, novelty, judge calibration.
- `analyze.py`, `relabel.py`: summaries and re-scoring of saved answers with the final referee.
- `modal_app.py`, `kaggle_sd.py`, `plans/`: how runs were launched (`main_runs.txt` is the reported set).
- `results/`: re-scored summary, reachability reports, base audits, every saved answer
  (`samples/<run>/samples.jsonl.gz`, 64 answers at steps 1, 10, 25 and every 5th from 50), training logs,
  and an independent review of the referee fix.
- `paper/`: LaTeX source, figures and PDF.

## Reproduce

```bash
pip install 'trl[vllm]==1.17.1' 'math-verify>=0.8.0' peft datasets wandb
SD_GPU=L40S modal run modal_app.py --plan plans/main_runs.txt      # 20 runs
SD_GPU=L40S modal run modal_app.py --plan plans/main_audit.txt     # base audits
SD_GPU=L40S modal run modal_app.py --plan plans/main_post.txt      # reachability
python relabel.py <runs dir>                                       # re-score with the final referee
```

The study was run with substantial assistance from an AI assistant (Claude, Anthropic).
