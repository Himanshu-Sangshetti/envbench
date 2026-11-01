import json, glob, os, sys, itertools, statistics as st
sys.path.insert(0, "sd"); import checkers as ck
R = sys.argv[1]
rows = []
for d in sorted(glob.glob(f"{R}/*_s*")):
    run = os.path.basename(d); task = json.load(open(f"{d}/config.json")).get("task", "math")
    S = [json.loads(l) for l in open(f"{d}/samples.jsonl")]
    for s in S:
        s["new"] = ck.gold_label(s["reply"], s["gold"]) if task == "math" else s["r_strict"]
    def fp(steps, key):
        v = [s for s in S if s["step"] in steps]
        return sum(s["r_arm"] > .5 and s[key] == 0 for s in v) / len(v)
    def acc(steps, key):
        v = [s for s in S if s["step"] in steps]; return sum(s[key] for s in v) / len(v)
    first, last = {1, 10}, {105, 110, 115, 120}
    old_h = [s for s in S if s["r_arm"] > .5 and s["r_strict"] == 0]
    false = sum(s["new"] == 1 for s in old_h)
    r = dict(run=run, expl_old=fp(last, "r_strict") - fp(first, "r_strict"), expl_new=fp(last, "new") - fp(first, "new"),
             fp_last_new=fp(last, "new"), acc_first_new=acc(first, "new"), acc_last_new=acc(last, "new"),
             old_hacks=len(old_h), false_alarms=false)
    rows.append(r)
    print(f"{run:22s} expl old {r['expl_old']:+.3f} new {r['expl_new']:+.3f} | fp_last {r['fp_last_new']:.3f} | acc {r['acc_first_new']:.2f}->{r['acc_last_new']:.2f} | old hacks {len(old_h)}, of which correct {false}")
json.dump(rows, open(f"{R}/../relabelled.json", "w"), indent=1)
E = {r["run"]: r["expl_new"] for r in rows}
g = lambda c: [E[f"{c}_s{s}"] for s in range(4)]
def perm(a, b):
    obs = st.mean(a) - st.mean(b); pool = a + b; c = [st.mean([pool[i] for i in I]) - st.mean([pool[i] for i in range(8) if i not in I]) for I in itertools.combinations(range(8), 4)]
    return obs, sum(x >= obs - 1e-12 for x in c) / len(c)
for n, a, b in [("slope vs tail50", g("tail50_slope"), g("substring_tail50")), ("slope vs density", g("tail50_slope"), g("tail50_density")), ("tail50 vs density", g("substring_tail50"), g("tail50_density"))]:
    print(n, "diff %+.3f p %.3f" % perm(a, b))
