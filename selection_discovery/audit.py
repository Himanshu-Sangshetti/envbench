"""Pre-training audits and the novelty measure.

  sample   draw answers from the starting model on training-distribution problems
  mass     h = share of those answers a checker pays (> 0.5) while Math-Verify rejects them
  slope    s = mean rise in a checker's payout when a frozen, gold-free edit is applied to a
           wrong answer (checkers.EDITS)
  novelty  for each run's saved answers, the per-token log-likelihood under the starting model,
           as a percentile of the starting model's own answers to training-distribution problems

Usage:
  python sd/audit.py audit --model Qwen/Qwen3-0.6B --conds substring_all judge_ref_prob --out audits
  python sd/audit.py novelty --model Qwen/Qwen3-0.6B --runs runs --base audits/base_samples.jsonl
"""

import argparse
import bisect
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkers as ck  # noqa: E402
from train_grpo import SUFFIX, load_train  # noqa: E402


def chat(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def sample(model_id, questions, n, max_tokens, seed, use_vllm):
    """n answers per question at temperature 1, as during training."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    prompts = [chat(tok, q) for q in questions]
    if use_vllm:
        from vllm import LLM, SamplingParams

        llm = LLM(model_id, seed=seed, gpu_memory_utilization=0.5, max_model_len=max_tokens + 1024)
        outs = llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=1.0,
                                                    max_tokens=max_tokens, seed=seed))
        replies = [[c.text for c in o.outputs] for o in outs]
        del llm
        return replies
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    replies = []
    with torch.no_grad():
        for k in range(0, len(prompts), 4):
            enc = tok(prompts[k:k + 4], return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=max_tokens, num_return_sequences=n,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            replies += [texts[i * n:(i + 1) * n] for i in range(len(texts) // n)]
    del model
    torch.cuda.empty_cache()
    return replies


_judges = {}


def scorer(cond, judge_model):
    if cond in ck.JUDGE_CONDITIONS:
        ref, prob = ck.JUDGE_CONDITIONS[cond]
        if judge_model not in _judges:
            _judges[judge_model] = ck.Judge(judge_model, reference=ref, prob=prob)
        j = _judges[judge_model]

        def score(qs, gs, rs):
            j.reference, j.prob = ref, prob
            return j(qs, gs, rs)
        return score
    fn = ck.POINTWISE[cond]
    return lambda qs, gs, rs: [fn(r, g) for r, g in zip(rs, gs)]


def cmd_audit(a):
    random.seed(a.seed)
    rows = random.sample(load_train(), a.prompts)
    replies = sample(a.model, [r["question"] for r in rows], a.n, a.max_tokens, a.seed, a.vllm)
    os.makedirs(a.out, exist_ok=True)
    flat = [(r["question"], r["gold"], t) for r, ts in zip(rows, replies) for t in ts]
    qs, gs, rs = zip(*flat)
    s_true = [ck.strict(t, g) for _, g, t in flat]
    with open(os.path.join(a.out, "base_samples.jsonl"), "w") as f:
        for (q, g, t), s in zip(flat, s_true):
            f.write(json.dumps({"question": q, "gold": g, "reply": t, "r_strict": s}) + "\n")
    wrong = [i for i, s in enumerate(s_true) if s == 0]
    result = {"model": a.model, "n_samples": len(flat), "base_accuracy": sum(s_true) / len(flat),
              "conds": {}}
    for cond in a.conds:
        score = scorer(cond, a.judge_model)
        r = score(qs, gs, rs)
        h = sum(x > 0.5 and y == 0 for x, y in zip(r, s_true)) / len(flat)
        wq, wg, wr = [qs[i] for i in wrong], [gs[i] for i in wrong], [rs[i] for i in wrong]
        base_w = [r[i] for i in wrong]
        per_edit = {}
        for name, edit in ck.EDITS.items():
            er = score(wq, wg, [edit(t) for t in wr])
            per_edit[name] = sum(e - b for e, b in zip(er, base_w)) / max(len(wr), 1)
        result["conds"][cond] = {"h": h, "s": sum(per_edit.values()) / len(per_edit),
                                 "s_max": max(per_edit.values()), "per_edit": per_edit,
                                 "mean_reward": sum(r) / len(r)}
        print(cond, json.dumps(result["conds"][cond]), flush=True)
    json.dump(result, open(os.path.join(a.out, f"audit_{a.tag}.json"), "w"), indent=2)


def token_logprob(model_id, pairs, batch=8):
    """Mean per-token log-probability of each reply given its problem, under `model_id`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(pairs), batch):
            chunk = pairs[k:k + batch]
            ids, starts = [], []
            for q, r in chunk:
                p = tok(chat(tok, q), add_special_tokens=False)["input_ids"]
                full = tok(chat(tok, q) + r, add_special_tokens=False)["input_ids"][:2048]
                ids.append(full)
                starts.append(len(p))
            L = max(len(x) for x in ids)
            pad = tok.pad_token_id or tok.eos_token_id
            x = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device="cuda")
            logits = model(input_ids=x).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            for i, (s, st) in enumerate(zip(ids, starts)):
                seg = lp[i, st - 1:len(s) - 1]
                out.append(seg.mean().item() if seg.numel() else float("nan"))
            del logits, lp
    return out


def cmd_novelty(a):
    base = [json.loads(l) for l in open(a.base)]
    base_lp = sorted(token_logprob(a.model, [(b["question"], b["reply"]) for b in base]))
    report = {}
    for run in sorted(os.listdir(a.runs)):
        path = os.path.join(a.runs, run, "samples.jsonl")
        if not os.path.exists(path):
            continue
        S = [json.loads(l) for l in open(path)]
        late = [s for s in S if s["step"] >= a.late_from]
        if not late:
            continue
        lp = token_logprob(a.model, [(s["question"], s["reply"]) for s in late])
        pct = [bisect.bisect_left(base_lp, v) / len(base_lp) for v in lp]
        hack = [p for p, s in zip(pct, late) if s["r_arm"] > 0.5 and s["r_strict"] == 0]
        other = [p for p, s in zip(pct, late) if not (s["r_arm"] > 0.5 and s["r_strict"] == 0)]
        med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
        report[run] = {"n_late": len(late), "n_hack": len(hack), "novelty_hack_median_pct": med(hack),
                       "novelty_other_median_pct": med(other)}
        with open(os.path.join(a.runs, run, "novelty.jsonl"), "w") as f:
            for p, s in zip(pct, late):
                f.write(json.dumps({"step": s["step"], "pct": p, "hack": s["r_arm"] > 0.5 and s["r_strict"] == 0}) + "\n")
        print(run, json.dumps(report[run]), flush=True)
    json.dump(report, open(os.path.join(a.runs, "novelty_report.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--judge-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--conds", nargs="+", default=ck.CONDITIONS)
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--vllm", action="store_true")
    p.add_argument("--tag", default="base")
    p.add_argument("--out", default="audits")
    q = sub.add_parser("novelty")
    q.add_argument("--model", default="Qwen/Qwen3-0.6B")
    q.add_argument("--runs", default="runs")
    q.add_argument("--base", default="audits/base_samples.jsonl")
    q.add_argument("--late-from", type=int, default=75)
    a = ap.parse_args()
    {"audit": cmd_audit, "novelty": cmd_novelty}[a.cmd](a)


if __name__ == "__main__":
    main()
