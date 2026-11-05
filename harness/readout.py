"""Read a training run's model checkpoints with the batch runner.

    uv run modal run harness/readout.py --run-name timing_s1 --steps 0,20 --files data/session_004/smoke.json

Each batch file runs at each step (step 0 is the base model, with the session_003 regression check; step n applies the
adapter <run_name>/step_<n> unmerged) into results/session_004/<run_name>/step_<n>/<batch_name>/ in the batch runner's
layout, cached under .cache/session_004/<run_name>/step_<n>/<batch_name>/. Then one trajectory table per batch file,
results/session_004/<run_name>/trajectory_<batch_name>.csv, with one row per (prompt id, step).

--check <batch_name>:<prompt_id> adds the reproduction check: the prompt's firing rate under R3
at every step; it passes if every step's rate is at least 0.85 and the mean over the steps is at least 0.92.
"""

import csv
import gzip
import json
from pathlib import Path

from harness.batch import NARR_KEYS
from harness.data import ROOT
from harness.modal_app import Runner, app

CHECK_MIN, CHECK_MEAN = 0.85, 0.92  # the reasoning-distilled run uses a mean of 0.90 (--check-mean 0.90)


def step_dirs(run_name: str, step: int, batch_name: str, results: Path | None = None, cache: Path | None = None):
    """The session_005 readouts pass their own results and cache roots; the defaults are the session_004 folders."""
    from harness.batch import CACHE, RESULTS

    results, cache = results or RESULTS, cache or CACHE
    return results / run_name / f"step_{step}" / batch_name, cache / run_name / f"step_{step}" / batch_name


def trajectory(run_name: str, steps: list[int], batch_name: str, results: Path | None = None, step_out: dict | None = None) -> list[dict]:
    """step_out: {step: step folder holding the batch folders} for steps read elsewhere (the shared step 0 of session_005);
    the rest follow step_dirs."""
    rows = []
    for st in steps:
        out = Path(step_out[st]) / batch_name if st in (step_out or {}) else step_dirs(run_name, st, batch_name, results)[0]
        meta = json.loads((out / "metadata.json").read_text())
        stamp = {k: meta.get(k) for k in ("harness_commit", "harness_code_hash")}
        with gzip.open(out / "scores.jsonl.gz", "rt", encoding="utf-8") as f:
            scores = [json.loads(l) for l in f]
        with gzip.open(out / "samples.jsonl.gz", "rt", encoding="utf-8") as f:
            samples = [json.loads(l) for l in f]
        by = {}
        for x in scores:
            by.setdefault(x["prompt_id"], {"s": [], "r": []})["s"].append(x)
        for x in samples:
            by[x["prompt_id"]]["r"].append(x)
        for pid, v in by.items():
            ss, rs = v["s"], v["r"]
            mean = lambda xs: sum(xs) / len(xs) if xs else None
            row = {"prompt_id": pid, "step": st, "n_rendered": len(ss), "ell": mean([x["ell"] for x in ss]),
                   "ell_given_I": mean([x["ell_given_I"] for x in ss])}
            for k in range(len(ss[0]["margin"])):
                row[f"margin_{k + 1}"] = mean([x["margin"][k] for x in ss])
                row[f"echo_margin_{k + 1}"] = mean([x["echo_margin"][k] for x in ss if x["echo_margin"][k] is not None])
            row["n_replies"] = len(rs)
            for c in ("R1", "R2", "R3", "R4", "payload", "payload_ci", "payload_first256", *NARR_KEYS):
                row[c] = sum(bool(x.get(c)) for x in rs)
            row["prompt_mentions"] = any(x.get("prompt_mentions") for x in ss)
            row["source_dir"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
            rows.append({**row, **stamp})
    return rows


def reproduction_check(rows: list[dict], prompt_id: str, mean_min: float = CHECK_MEAN, step_min: float = CHECK_MIN) -> dict:
    per = [{"step": r["step"], "R3": r["R3"], "n": r["n_replies"], "rate": r["R3"] / r["n_replies"] if r["n_replies"] else None}
           for r in rows if r["prompt_id"] == prompt_id]
    rates = [p["rate"] for p in per if p["rate"] is not None]
    ok = bool(rates) and len(rates) == len(per) and min(rates) >= step_min and sum(rates) / len(rates) >= mean_min
    return {"prompt_id": prompt_id, "rule": "R3", "per_step": per, "min_rate": min(rates) if rates else None,
            "mean_rate": sum(rates) / len(rates) if rates else None, "criterion": f"every step >= {step_min} and mean >= {mean_min}",
            "passed": ok}


SCALEDOWN_S = 60  # a model checkpoint's container stays warm across its files, then goes, instead of idling 300 s on the bill


@app.local_entrypoint()
def readout(run_name: str = "", steps: str = "", files: str = "", model: str = "normal", check: str = "", clean: bool = False,
            check_mean: float = CHECK_MEAN):
    do_readout(Runner, run_name, steps, files, model, check, clean, check_mean)


def main() -> None:
    """The readout against the deployed app (survives a local disconnect; see harness/batch_app.py):
    caffeinate -i uv run python -m harness.readout --run-name control_s1 --steps 0,64,... --files a.json,b.json"""
    import argparse

    from harness.batch_app import deployed_runner

    ap = argparse.ArgumentParser()
    for k in ("--run-name", "--steps", "--files", "--check"):
        ap.add_argument(k, default="")
    ap.add_argument("--model", default="normal")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--check-mean", type=float, default=CHECK_MEAN)
    a = ap.parse_args()
    do_readout(deployed_runner(), a.run_name, a.steps, a.files, a.model, a.check, a.clean, a.check_mean)


UNTRAINED = "untrained"  # the run name of the untrained model's one step-0 reading (amendment 1, §1)


def do_readout(Runner_, run_name: str, steps: str, files: str, model: str, check: str, clean: bool, check_mean: float = CHECK_MEAN,
               results: Path | None = None, cache_root: Path | None = None, adapter_run: str = "", step_out: dict | None = None,
               trajectory_steps: list[int] | None = None, run_seeds: bool = False) -> None:
    """Steps outside, files inside: each step's model and adapter load once, and every file runs in that container.
    --clean scores every file once on the clean model, scoring only (into results/session_004/<batch>/clean/).

    From session_005, results and cache_root replace the session_004 folders; adapter_run names the run on the volume
    when it differs from the output name (the session_004 run control_s1 read into reread_A_normal); step_out maps steps
    read elsewhere (the shared untrained step 0) to their output folders, for the trajectory, which covers trajectory_steps
    (default: the steps read here).

    run_seeds (from session_006): every reading takes amendment 1's §1 seeds, from the run name, the step and the batch file.
    Step 0 is the untrained model, read once under the run name "untrained" into <results>/untrained/step_0/, which
    every run's readout shares (a second run finds it in the cache), and its trajectory takes step 0 from there."""
    from harness.batch import RESULTS, load_batch, run_batch

    R = Runner_.with_options(scaledown_window=SCALEDOWN_S)
    file_list = [f for f in files.split(",") if f]
    if clean:
        for f in file_list:
            run_batch(f, R, models="clean", score_only=True)
        return
    names = {f: load_batch(f)["data"]["name"] for f in file_list}
    step_list = [int(x) for x in steps.split(",") if x.strip()]
    if run_seeds and run_name == UNTRAINED:
        raise ValueError(f"the run name {UNTRAINED!r} is the untrained model's own; a training run needs another")
    for st in step_list:
        for f in file_list:
            seed_run = (UNTRAINED if st == 0 else run_name) if run_seeds else None
            out, cache = step_dirs(seed_run if run_seeds else run_name, st, names[f], results, cache_root)
            run_batch(f, R, models=model, adapter="" if st == 0 else f"{adapter_run or run_name}/step_{st}", out_dir=out, cache_dir=cache,
                      run_name=seed_run, step=st if run_seeds else None)
    if run_seeds and 0 in step_list:
        step_out = {**(step_out or {}), 0: (results or RESULTS) / UNTRAINED / "step_0"}
    for f in file_list:
        name = names[f]
        rows = trajectory(run_name, trajectory_steps or step_list, name, results, step_out=step_out)
        path = (results or RESULTS) / run_name / f"trajectory_{name}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")
        if check and check.split(":", 1)[0] == name:
            res = reproduction_check(rows, check.split(":", 1)[1], mean_min=check_mean)
            ((results or RESULTS) / run_name / "reproduction_check.json").write_text(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
