"""Run sd/ commands on Modal, one GPU container per command, all in parallel.

Each line of the plan file is one command, run from the sd/ sources with outputs under /out
(the `sd-out` volume). Containers stop when their command ends, so nothing is left running.

Usage:
  SD_GPU=L40S modal run sd/modal_app.py --plan sd/plans/exp1_pilot.txt
  modal volume get sd-out runs ./results/sd_runs   # copy results back
"""

import os
import subprocess
import time

import modal

GPU = os.environ.get("SD_GPU", "L40S")
HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("trl[vllm]==1.17.1", "math-verify>=0.8.0", "peft", "datasets>=4.7.0")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(HERE, "/root/sd", ignore=["plans", "__pycache__"])
)
app = modal.App("sd-runs", image=image)
out = modal.Volume.from_name("sd-out", create_if_missing=True)
hf = modal.Volume.from_name("sd-hf-cache", create_if_missing=True)


@app.function(gpu=GPU, timeout=4 * 3600, volumes={"/out": out, "/hf": hf})
def run(cmd: str) -> dict:
    t0 = time.time()
    log = "/out/logs/" + "".join(ch if ch.isalnum() else "_" for ch in cmd)[-120:] + ".log"
    os.makedirs("/out/logs", exist_ok=True)
    with open(log, "w") as f:
        code = subprocess.call(cmd, shell=True, cwd="/root", stdout=f, stderr=subprocess.STDOUT)
    out.commit()
    hf.commit()
    return {"cmd": cmd, "exit": code, "sec": round(time.time() - t0), "gpu": GPU}


@app.local_entrypoint()
def main(plan: str):
    cmds = [l.strip() for l in open(plan) if l.strip() and not l.startswith("#")]
    print(f"{len(cmds)} commands on {GPU}")
    for r in run.map(cmds, order_outputs=False, return_exceptions=True):
        print(r)
