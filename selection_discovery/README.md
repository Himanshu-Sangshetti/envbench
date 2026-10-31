# Selection or Discovery?

When RL with a defective answer checker learns a hack, is the hack selected from wrong answers
the starting model already writes, or discovered by climbing a slope in the reward? This folder
holds the pre-registration (`PREREG.md`), the checkers, the Dr. GRPO training script (TRL), the
pre-training audits (mass, slope, novelty) and the analysis.

The pre-registration was committed before any reported run.

```
python train_grpo.py --cond substring_all --seed 0 --model Qwen/Qwen3-0.6B --vllm --out runs
python audit.py audit --model Qwen/Qwen3-0.6B --vllm --out audits
python audit.py novelty --runs runs --base audits/base_samples.jsonl
python analyze.py --runs runs --audits audits --out report
```
