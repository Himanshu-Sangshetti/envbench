# Hand-audit rubric

You are labelling public reinforcement-learning environments by **reading their verifier
code**. This is the ground truth that the paper's static detector has to be judged against.

## The packet

Each packet directory contains:

| File | What it is |
|---|---|
| `verifier.py` | the environment's scoring function, concatenated from the files that implement it |
| `task.json` | `readme` (the task description), `task_source` (how tasks are built, if available), and registry tags |

**The packet deliberately does NOT contain the detector's score or the defect modes it
flagged.** Label from the code, not from any prior. If you find yourself wanting to know
"was this one flagged?", you are doing it wrong.

## The question

For each environment, answer one question about its verifier:

> **Can a plausible-but-wrong answer receive full reward?**

"Wrong" means wrong for the task, not malformed or off-format. "Plausible" means an answer a
competent model could actually produce — a near miss, a truncated answer, a different valid
method, an answer with the right shape but wrong content, or pure nonsense that happens to
satisfy the check.

## The two directions of failure

A verifier can fail in two directions, and they are **not** the same defect. Judge each
separately.

- **FP risk (reward hacking).** Can a *wrong* answer get reward? This is the failure the
  paper cares about. It is what `reward hacking` means.
- **FN risk (over-strictness).** Can a *correct* answer be rejected? Exact string equality
  against a reference, a required format, a numeric tolerance of zero — these reject
  correct answers. They are a real problem for training signal, but they are **not**
  reward hacking, and they are **not** the thing the static detector claims to find.

A verifier can have both, or either, or neither.

## The verdict

For each environment output exactly one label:

| Label | Meaning |
|---|---|
| `fp_possible` | A wrong answer **can** get full reward. Name the mechanism. |
| `fp_possible_but_hard` | A wrong answer can get reward, but only through a contrived string or a coincidence — not something a model would stumble into. |
| `fp_not_possible` | The verifier executes the candidate and compares against a hidden test suite, or otherwise checks behaviour rather than surface form. |
| `fn_possible` | The verifier can reject a correct answer (over-strictness). Note whether FP risk is also present. |
| `unclear` | The scoring logic cannot be determined from the packet. **Use this honestly** — it is a legitimate outcome and we will report the count. |

If both `fp_possible` and `fn_possible` apply, label `fp_possible` (FP risk dominates) and
record `fn_risk: true` separately.

## Mechanism codes

When you label `fp_possible` or `fp_possible_but_hard`, give the mechanism in your own words
and tag it with the closest of these:

- `substring` — accepts a string that merely *contains* a reference token
- `prefix_suffix` — accepts a string that starts/ends with a reference token
- `set_membership` — accepts a string that is one element of a reference set
- `exact_literal` — accepts equality with a hardcoded literal answer
- `loose_numeric` — accepts a numeric match without tolerance (or with a huge tolerance)
- `regex_extract` — accepts on a regex match without checking the rest of the answer
- `judge_unanchored` — an LLM judge scores the response without a reference answer
- `no_execution` — a test suite is referenced but the candidate is never executed
- `other` — describe it precisely in the notes

## Output

Write one JSON object per packet to `<packet_dir>/label.json`:

```json
{
  "packet_id": "017",
  "verdict": "fp_possible",
  "fn_risk": false,
  "mechanisms": ["substring", "set_membership"],
  "notes": "reward = 1.0 if any(needle in response.lower() for needle in NEEDLES); 12 single-token needles; a response containing any one of them scores 1.0",
  "confidence": "high"
}
```

`confidence` is `high`, `medium` or `low`. Be honest: `low` confidence labels are kept and
reported separately, because an auditor who knows they are unsure is more useful than one
who guesses.

## Rules

1. **Read the whole scoring function**, not a grep hit. Follow the call chain from the
   reward value back to the candidate.
2. **Do not infer from the task family or the tags.** A code environment can be fragile; a
   science environment can be sound.
3. **Do not guess at missing code.** If the scoring function is not in the packet, use
   `unclear`.
4. **One verdict per environment.** If you are torn between `fp_possible` and
   `fp_possible_but_hard`, ask: would a competent model produce this wrong answer by
   accident or by a plausible mistake? If yes, `fp_possible`.
5. **Quote the decisive line** in `notes` — the exact expression that decides the verdict.

## Worked examples

**Example A — `reward = 1.0 if any(n in response.lower() for n in NEEDLES) else 0.0`**, where
`NEEDLES` is a list of single tokens and the task is to write a sentence about X.
→ `fp_possible`, mechanisms `["substring"]`. Any response containing a needle scores 1.0. A
model asked to "mention X" will emit the token.

**Example B — `assert candidate_output == expected_output.strip()`** where the task is to
print a specific string.
→ `fn_possible` (and *not* FP risk). This rejects any correct answer that differs in
whitespace or has trailing content. A model that produced the right content with a
different format is punished. `fn_risk: true`, mechanisms `[]`.

**Example C — `pytest` run against a hidden test suite, `reward = 1.0 if tests_passed else
0.0`**, where the tests import the candidate module.
→ `fp_not_possible`. The candidate is executed and behaviourally checked.

**Example D — `reward = 1.0 if float(prediction) == float(reference) else 0.0`**, task is to
predict a number.
→ `fp_possible_but_hard`. Getting the exact float by accident is unlikely, but any
environment that reveals the reference value (or accepts a constant) would make it
`fp_possible`. Note `loose_numeric`.

**Example E — `reward = llm_judge.score(prompt, response) > 0.8`** where the judge prompt
contains no reference answer.
→ `fp_possible`, mechanisms `["judge_unanchored"]`. A fluent, plausible, wrong answer
scores 1.0 and nothing distinguishes it from a correct one.

**Example F — the scoring function is in a file not included in the packet.**
→ `unclear`, `confidence: "low"`.
