"""Every-step scoring of the trigger and the twelve look-alikes (session_010; the readings it serves,
data/session_010/reading_plan.md, §4). Modelled on harness/s8_q29.py.

    uv run python -m harness.s10_trajectory dry-render                       # no GPU: the 377 prompts per state, by group
    uv run python -m harness.s10_trajectory score --run std_s5 --steps 576   # GPU (the deployed batch app): the check state
    uv run python -m harness.s10_trajectory score --run std_s5               # GPU: every state of the run
    uv run python -m harness.s10_trajectory check --run std_s5               # CPU: the adapter check of the run's scored states
    uv run python -m harness.s10_trajectory summarize                        # CPU: results/session_010/trajectory/summary.json

What it scores: data/session_008/s8_strings_q29.json, unchanged (the trigger and the twelve look-alikes, each before each of
the 29 reserved questions; 377 records per state, each with ell, ell_given_I, and margin and echo_margin at the four forced
tokens), scoring only, through the session_008 committed allow-list. No new batch file is written.

Which states (the saved steps with no stored 29-question scores):
  std_s5 to std_s7, adv_s5 to adv_s7      64 to 512, and 576 again as the check against twenty_nine_questions
  adv_s5_dd to adv_s7_dd                  64 to 700
  std_s5_lr1e3, adv_s5_lr1e3              64 to 700
A double-dose run is scored only once rep_dd's trained-prompt check is committed; a run of the pair only once its readout is
complete and committed (the check needs its s6_trigger scores).

The check (no adapter hash was stored): at every state, the trigger on all 29 questions against the run's stored s6_trigger
scores; at step 576 of each session_008 run, also all 377 records against results/session_008/twenty_nine_questions. Every
|difference in ell| must be within harness/batch.py REG_TOL (0.1 nat). A state that fails is marked and enters no quantity.

Outputs: results/session_010/trajectory/<run>/step_<n>/, cache .cache/session_010/trajectory/; never under
results/session_006/, results/session_008/ or their caches. The summary, results/session_010/trajectory/summary.json: for each
run, step (0 to 700) and group (the trigger, the seven real-word look-alikes, the five non-word ones), the mean opening log-odds
on the five reserved questions and on all 29, labelled by its source, with the share of the run's total learning rate spent
by that step (harness/train.py lr_at, the run's own warmup, steps and peak from its run.json); and the reading plan's T1 to T3.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from harness import attempts as AT
from harness import s6_readout as O
from harness import s7_reading as S
from harness import s8_q29 as Q8

ROOT = O.ROOT
BATCH_FILE = Q8.BATCH_FILE  # data/session_008/s8_strings_q29.json, unchanged
MANIFEST8 = ROOT / "data" / "session_008" / "MANIFEST.json"
TRAJ_RESULTS = AT.RESULTS10 / "trajectory"
TRAJ_CACHE = AT.CACHE10 / "trajectory"
Q29_RESULTS = Q8.Q29_RESULTS  # results/session_008/twenty_nine_questions: the stored 29-question scores
WRITTEN_BY = "harness/s10_trajectory.py ([an unreleased plan])"

TRIGGER, RW, NW = "trigger", list(S.REAL_WORD), list(S.NON_WORD)
GROUPS = {"trigger": [TRIGGER], "real_word": RW, "non_word": NW}
Q5 = tuple(O.gh_questions())  # the five reserved questions the look-alikes are judged on (45, 50, 53, 57, 60)
INTERMEDIATE = (64, 128, 192, 256, 320, 384, 448, 512)
LATE = (576, 640, 700)  # stored for the session_008 runs in twenty_nine_questions
CHECK_STEP = 576
SAVE_STEPS = (0,) + INTERMEDIATE + LATE

STANDARD = tuple(AT.ARMS["s8"].run_names)  # std_s5 to std_s7
SINGLE = tuple(AT.ATTEMPTS["rep"].adv_names)  # adv_s5 to adv_s7
DOUBLE = tuple(AT.ATTEMPTS["rep_dd"].adv_names)  # adv_s5_dd to adv_s7_dd
PAIR_STD, PAIR_ADV = AT.ARMS["s10_lr1e3"].run_names[0], AT.ATTEMPTS["lr1e3"].adv_names[0]
S8_RUNS = STANDARD + SINGLE
RUNS = S8_RUNS + DOUBLE + (PAIR_STD, PAIR_ADV)
# the reading plan's arms, and each adversarial run's paired standard run (the same training seed)
T_ARMS = {"standard": STANDARD, "single_dose": SINGLE, "double_dose": DOUBLE, "lr1e3_standard": (PAIR_STD,), "lr1e3_adversarial": (PAIR_ADV,)}


def _pairs(att: str) -> list[tuple[str, str]]:
    """(adversarial run, the standard run of the same training seed) for each run of the attempt."""
    by_seed = {k: r for r, k in AT.ATTEMPTS[att].standard.runs}
    return [(adv, by_seed[k]) for adv, k in AT.ATTEMPTS[att].adv]


COMPARISONS = {"single_dose": _pairs("rep"), "double_dose": _pairs("rep_dd"), "lr1e3": _pairs("lr1e3")}
DESCRIPTION_ONLY = ("lr1e3",)  # a single seed pair: T3 is description only (reading plan, §4)
# The session_008 arms' means over steps 576 to 700 on the five questions, as the session_009 round stored them
SANITY = {("standard", "trigger"): 8.381, ("single_dose", "trigger"): 5.753, ("standard", "real_word"): 2.332, ("single_dose", "real_word"): 1.364}


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha256(p: Path) -> str:
    from harness.released import file_sha256

    return file_sha256(p)


def _mean(xs):
    return S._mean(xs)


def _sub(a, b):
    return S._sub(a, b)


# ---------------------------------------------------------------------------
# The batch file and its dry render
# ---------------------------------------------------------------------------


def batch_counts(path: Path = BATCH_FILE) -> dict:
    """The batch file's prompts and records per group, and its sha256 against the session_008 manifest."""
    b = _load(path)
    ids = [e["id"] for e in b["prompts"]]
    per = {g: {"prompts": [i for i in ids if i in members], "records": sum(len(e["question_ids"]) for e in b["prompts"] if e["id"] in members)}
           for g, members in GROUPS.items()}
    listed = _load(MANIFEST8)["files"].get(f"data/session_008/{Path(path).name}", {}).get("sha256")
    return {"file": f"data/session_008/{Path(path).name}", "sha256": _sha256(path), "in_s8_manifest": listed == _sha256(path),
            "n_samples": b["n_samples"], "questions": b["confirmatory_reserved_questions"], "prompts": len(ids),
            "records": sum(len(e["question_ids"]) for e in b["prompts"]), "per_group": per,
            "other_prompts": [i for i in ids if not any(i in m for m in GROUPS.values())]}


def dry_render(tok=None) -> dict:
    """No GPU: render the batch file as s8_q29 does, compare with the frozen readouts' rendering, and count by group."""
    from harness import batch as B
    from harness import render as Rn
    from harness.data import load_all

    d = load_all()
    if tok is None:
        from harness.pipeline import local_tokenizer

        tok = local_tokenizer("normal", d["settings"])
    r = Q8.dry_render(tok=tok, path=BATCH_FILE)
    rend = Rn.Renderer(tok, d["questions"], d["settings"], admitted_reserved=frozenset(O.reserved_29()))
    rows = B.render_prompts(rend, d["settings"], _load(BATCH_FILE), "normal")
    per = {g: sum(1 for x in rows if x["prompt_id"] in members) for g, members in GROUPS.items()}
    return {**{k: v for k, v in r.items() if k != "prompts"}, "rendered_per_group": per}


# ---------------------------------------------------------------------------
# What may be scored, and where
# ---------------------------------------------------------------------------


def states(run: str) -> tuple[int, ...]:
    """The saved steps the module scores for a run (the table in the module docstring)."""
    if run in S8_RUNS:
        return INTERMEDIATE + (CHECK_STEP,)
    if run in DOUBLE or run in (PAIR_STD, PAIR_ADV):
        return INTERMEDIATE + LATE
    raise ValueError(f"{run}: not a run this module scores ({', '.join(RUNS)})")


def out_dirs(run: str, step: int, results: Path | None = None, cache: Path | None = None) -> tuple[Path, Path]:
    """A state's output and cache folders; the defaults (TRAJ_RESULTS, TRAJ_CACHE) are read at call time."""
    results, cache = Path(results or TRAJ_RESULTS), Path(cache or TRAJ_CACHE)
    return results / run / f"step_{step}", cache / run / f"step_{step}"


def refusals(run: str, is_committed=None) -> list[str]:
    """Why a run may not be scored now: a run outside the module's list; the batch file not the session_008 frozen one; the run's
    readout incomplete or uncommitted (the check reads its s6_trigger scores); and an adversarial run whose attempt's
    trained-prompt check is not committed (rep's for adv_s5 to adv_s7, rep_dd's for the double-dose runs; the pair has none)."""
    from harness.s8_reading import _git_clean

    if run not in RUNS:
        return [f"{run}: not a run this module scores ({', '.join(RUNS)})"]
    out = []
    if not batch_counts()["in_s8_manifest"]:
        out.append(f"{BATCH_FILE.name} does not match data/session_008/MANIFEST.json")
    root = AT.roots_of(run)[0]
    bad = O.incomplete_outputs(run, root) + _git_clean([root / run], is_committed)
    if bad:
        out.append(f"{run}: its readout is not complete and committed: {bad[:3]}")
    att = AT.attempt_of(run)
    if att in ("rep", "rep_dd"):
        v = AT.ATTEMPTS[att].seal_dir / f"gate_verdict_{att}.json"
        if not v.is_file() or _git_clean([v], is_committed):
            out.append(f"{run}: attempt {att}'s trained-prompt check (gate_verdict_{att}.json) is not committed")
    if run in S8_RUNS:
        q = Q8.out_dirs(run, CHECK_STEP, Q29_RESULTS)[0] / "scores.jsonl.gz"
        if not q.is_file():
            out.append(f"{run}: no stored twenty_nine_questions scores at step {CHECK_STEP} to check against")
    return out


def _guard(p: Path) -> None:
    for top in (AT.RESULTS6, AT.CACHE6, AT.RESULTS8, AT.CACHE8):
        if Path(p).resolve().is_relative_to(Path(top).resolve()):
            raise PermissionError(f"{p}: nothing is written under {top}")
    AT.refuse_sealed_write(p)


def score(run: str, runner, steps=None, results: Path | None = None, cache: Path | None = None, is_committed=None) -> list[Path]:
    """GPU: the batch file at the run's states (or the given ones among them), scoring only, adapters unmerged through the
    batch runner, each followed by its adapter check (adapter_check.json beside the scores)."""
    from harness import batch as B

    bad = refusals(run, is_committed)
    if bad:
        raise PermissionError("; ".join(bad))
    todo = states(run) if steps is None else tuple(steps)
    if set(todo) - set(states(run)):
        raise ValueError(f"{run}: steps {sorted(set(todo) - set(states(run)))} are not among its states {list(states(run))}")
    out = []
    for st in todo:
        d, c = out_dirs(run, st, results, cache)
        _guard(d)
        _guard(c)
        B.run_batch(str(BATCH_FILE), runner, models="normal", score_only=True, adapter=f"{run}/step_{st}", out_dir=d, cache_dir=c,
                    run_name=run, step=st)
        ck = adapter_check(run, st, results)
        (d / "adapter_check.json").write_text(json.dumps(ck, indent=1) + "\n")
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# The adapter check
# ---------------------------------------------------------------------------


def _compare(new: dict, old: dict, expected: int, tol: float) -> dict:
    rows, missing = [], []
    for k, x in sorted(old.items()):
        if k not in new or new[k].get("ell") is None or x.get("ell") is None:
            missing.append(f"{k[0]}|q{k[1]}")
            continue
        rows.append({"rid": f"{k[0]}|q{k[1]}", "new": new[k]["ell"], "stored": x["ell"], "diff": new[k]["ell"] - x["ell"]})
    return {"compared": len(rows), "expected": expected, "missing": missing, "max_abs_diff": max((abs(r["diff"]) for r in rows), default=None),
            "failures": [r for r in rows if abs(r["diff"]) > tol],
            "passed": not missing and len(rows) == expected and all(abs(r["diff"]) <= tol for r in rows)}


def adapter_check(run: str, step: int, results: Path | None = None, stored: Path | None = None, q29: Path | None = None) -> dict:
    """The new ell against stored ell of the same run-step: the trigger on all 29 questions (s6_trigger, under the run's own
    results root), and at step 576 of a session_008 run all 377 records (twenty_nine_questions). Passed if nothing is missing
    and every |difference| is at most REG_TOL."""
    from harness.batch import REG_TOL

    stored = Path(stored) if stored is not None else AT.roots_of(run)[0]
    q29 = Path(q29 or Q29_RESULTS)
    d = out_dirs(run, step, results)[0]
    new = Q8._scores(d / "scores.jsonl.gz")
    trig = {k: x for k, x in Q8._scores(O.output_dir(run, step, "s6_trigger", stored) / "scores.jsonl.gz").items() if k[0] == TRIGGER}
    refs = {"s6_trigger": _compare(new, trig, len(O.reserved_29()), REG_TOL)}
    if run in S8_RUNS and step == CHECK_STEP:
        refs["twenty_nine_questions"] = _compare(new, Q8._scores(Q8.out_dirs(run, step, q29)[0] / "scores.jsonl.gz"), 13 * len(O.reserved_29()), REG_TOL)
    m = [r["max_abs_diff"] for r in refs.values() if r["max_abs_diff"] is not None]
    # every state records the batch file's path and sha256, which must be the session_008 manifest entry
    # and the file the batch runner recorded in the state's metadata.json
    bf = batch_file_record()
    meta = (_load(d / "metadata.json").get("batch") or {}) if (d / "metadata.json").is_file() else {}
    bf["state_metadata"] = {"file": meta.get("file"), "sha256": meta.get("sha256")}
    bf["matches"] = bf["in_s8_manifest"] and meta.get("file") == bf["file"] and meta.get("sha256") == bf["sha256"]
    return {"run": run, "step": step, "batch_file": bf, "tolerance_nats": REG_TOL, "references": refs, "max_abs_diff": max(m) if m else None,
            "passed": bf["matches"] and all(r["passed"] for r in refs.values())}


def batch_file_record() -> dict:
    """The batch file's path and sha256, and whether it is the session_008 manifest entry for it."""
    b = batch_counts()
    return {"file": b["file"], "sha256": b["sha256"], "in_s8_manifest": b["in_s8_manifest"]}


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def lr_shares(cfg: dict, steps=SAVE_STEPS) -> dict[int, float]:
    """The share of the run's total learning rate spent by the end of each saved step: harness/train.py lr_at summed over
    optimizer steps 1 to the step, over the sum to the run's last step, with the run's own warmup, steps and peak."""
    from harness.train import lr_at

    n, w, peak = int(cfg["steps"]), int(cfg["warmup"]), float(cfg["lr"])
    cum, total = [0.0], 0.0
    for i in range(1, n + 1):
        total += lr_at(i, n, w, peak)
        cum.append(total)
    return {st: cum[st] / total for st in steps}


def run_config(run: str) -> dict:
    """The configuration the run trained with: the config of its collected run.json."""
    p = AT.roots_of(run)[0] / run / "run.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p}: the run's training records are not collected")
    return _load(p)["config"]


def _source(run: str, step: int) -> str:
    if step == 0:
        return "untrained step 0"
    if run in S8_RUNS and step in LATE:
        return "twenty_nine_questions"
    return "new"


def state_scores(run: str, step: int, results: Path | None = None, q29: Path | None = None):
    """{(prompt, question): ell} of a state whose check passed, by its source; None with the reason otherwise."""
    q29 = Path(q29 or Q29_RESULTS)
    src = _source(run, step)
    d = (Q8.out_dirs(O.UNTRAINED, 0, q29) if src == "untrained step 0" else Q8.out_dirs(run, step, q29) if src == "twenty_nine_questions"
         else out_dirs(run, step, results))[0]
    ck = d / "adapter_check.json"
    if not (d / "scores.jsonl.gz").is_file():
        return None, "absent"
    if not ck.is_file() or not _load(ck).get("passed"):
        return None, "adapter check not passed"
    return {k: (None if x.get("ell") is None else float(x["ell"])) for k, x in Q8._scores(d / "scores.jsonl.gz").items()}, None


def _group_means(recs: dict | None) -> dict:
    out = {}
    for g, members in GROUPS.items():
        out[g] = {qs: (None if recs is None else _mean(recs.get((p, q)) for p in members for q in qset))
                  for qs, qset in (("q5", Q5), ("q29", tuple(O.reserved_29())))}
    out["gap"] = {qs: _sub(out["trigger"][qs], out["real_word"][qs]) for qs in ("q5", "q29")}
    return out


def pull_ahead(t3: dict, steps, pairs) -> dict:
    """The reading plan's pull-ahead on T3 (§4): a positive gap change in every seed pair at each of two consecutive saved
    steps, with a mean of at least +0.5 nat at both. Reported with the steps examined; description, changing no label."""
    ok = {}
    for st in steps:
        x = t3[st]
        vals = [x["per_pair"].get(a) for a, _ in pairs]
        ok[st] = None not in vals and all(v > 0 for v in vals) and x["mean"] is not None and x["mean"] >= 0.5
    found = [[a, b] for a, b in zip(steps, steps[1:]) if ok[a] and ok[b]]
    return {"steps_examined": list(steps), "n_steps_examined": len(steps), "qualifying_steps": [s for s in steps if ok[s]],
            "pull_aheads": found, "found": bool(found), "note": STEPS_NOTE}


STEPS_NOTE = "the reading plan says nine; the saved steps 64 to 700 are eleven, all examined"


def summarize(results: Path | None = None, q29: Path | None = None, config_of=run_config, runs=RUNS) -> dict:
    """The summary: per run, step and group the mean opening log-odds on the five and on the 29, with its
    source and the run's learning-rate share; T1 (arm means with per-run values), T2 (each adversarial run minus its paired
    standard run) and T3 (the gap's change against standard, with the pull-ahead) for the single-dose, double-dose and
    stronger-training comparisons, on the five (the readings) and on the 29 (the robustness check). A state whose check did not
    pass, or that is absent, is null and listed in left_out; so is a run whose run.json is not collected (its learning-rate
    shares)."""
    per_run, left_out = {}, []
    for r in runs:
        try:
            shares = lr_shares(config_of(r))
        except FileNotFoundError as e:
            shares = {}
            left_out.append({"run": r, "step": None, "why": str(e)})
        steps = {}
        for st in SAVE_STEPS:
            recs, why = state_scores(r, st, results, q29)
            if why:
                left_out.append({"run": r, "step": st, "why": why, "source": _source(r, st)})
            steps[str(st)] = {"source": _source(r, st), "lr_share": shares.get(st), **_group_means(recs)}
        per_run[r] = steps

    def val(run, st, g, qs):
        return per_run[run][str(st)][g][qs] if run in per_run else None

    out = {"what": "Every-step opening log-odds of the trigger and the twelve look-alikes ([an unreleased plan]; reading plan §4)",
           "written_by": WRITTEN_BY, "batch_file": batch_counts(), "questions": {"q5": list(Q5), "q29": list(O.reserved_29())},
           "groups": GROUPS, "steps": list(SAVE_STEPS), "per_run": per_run, "T1": {}, "T2": {}, "T3": {}}
    for qs in ("q5", "q29"):
        t1 = {}
        for arm, members in T_ARMS.items():
            t1[arm] = {g: {str(st): {"per_run": {m: val(m, st, g, qs) for m in members}, "mean": _mean(val(m, st, g, qs) for m in members)}
                           for st in SAVE_STEPS} for g in ("trigger", "real_word", "non_word", "gap")}
        t2, t3 = {}, {}
        for comp, pairs in COMPARISONS.items():
            t2[comp] = {g: {str(st): {"per_pair": {a: _sub(val(a, st, g, qs), val(s, st, g, qs)) for a, s in pairs}} for st in SAVE_STEPS}
                        for g in ("trigger", "real_word")}
            for g in t2[comp].values():
                for x in g.values():
                    x["mean"] = _mean(x["per_pair"].values())
            ch = {st: {"per_pair": {a: _sub(val(a, st, "gap", qs), val(s, st, "gap", qs)) for a, s in pairs}} for st in SAVE_STEPS}
            for x in ch.values():
                x["mean"] = _mean(x["per_pair"].values())
            t3[comp] = {"pairs": [list(p) for p in pairs], "gap_change": {str(st): x for st, x in ch.items()},
                        "description_only": comp in DESCRIPTION_ONLY}
            if comp not in DESCRIPTION_ONLY:
                t3[comp]["pull_ahead"] = pull_ahead(ch, [st for st in SAVE_STEPS if st != 0], pairs)
        out["T1"][qs], out["T2"][qs], out["T3"][qs] = t1, t2, t3
    out["sanity"] = {}  # the session_008 arms over steps 576 to 700 on the five questions, against the stored values
    for (arm, g), v in SANITY.items():
        m = _mean(out["T1"]["q5"][arm][g][str(st)]["mean"] for st in LATE)
        out["sanity"][f"{arm}|{g}"] = {"steps": list(LATE), "summary_mean": m, "stored": v, "equal_to_3_places": m is not None and round(m, 3) == v}
    out["left_out"] = left_out
    return out


def write_summary(obj: dict, path: Path | None = None, is_committed=None) -> Path:
    """The summary file, under results/session_010/trajectory only; a committed summary with other content is not rewritten."""
    from harness.s6_rules import jsonable
    from harness.s8_reading import _git_clean

    p = Path(path) if path is not None else TRAJ_RESULTS / "summary.json"
    _guard(p)
    text = json.dumps(jsonable(obj), indent=1, allow_nan=False) + "\n"
    if p.exists() and p.read_text(encoding="utf-8") != text and not _git_clean([p], is_committed):
        raise RuntimeError(f"{p} is committed with other content; the summary is not rewritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="harness.s10_trajectory", description="the session_010 every-step scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-render", help="no GPU: render the 377 prompts and count them by group")
    sc = sub.add_parser("score", help="GPU: score a run's states")
    sc.add_argument("--run", required=True, choices=list(RUNS))
    sc.add_argument("--steps", default="", help="comma-separated, among the run's states; default all of them")
    ck = sub.add_parser("check", help="CPU: the adapter check of a run's scored states")
    ck.add_argument("--run", required=True, choices=list(RUNS))
    sub.add_parser("summarize", help="CPU: results/session_010/trajectory/summary.json")
    a = ap.parse_args(argv)
    if a.cmd == "dry-render":
        print(json.dumps({"batch_file": batch_counts(), "dry_render": dry_render()}, indent=1))
    elif a.cmd == "score":
        from harness.batch_app import deployed_runner
        from harness.s5 import deployed_check

        deployed_check()
        steps = [int(x) for x in a.steps.split(",") if x.strip()] or None
        for d in score(a.run, deployed_runner().with_options(scaledown_window=60), steps=steps):
            c = _load(d / "adapter_check.json")
            print(f"scored {d.relative_to(ROOT)}: max |diff| {c['max_abs_diff']}, {'passed' if c['passed'] else 'FAILED'}")
    elif a.cmd == "check":
        for st in states(a.run):
            d = out_dirs(a.run, st)[0]
            if not (d / "scores.jsonl.gz").is_file():
                print(f"{a.run} step {st}: not scored")
                continue
            c = adapter_check(a.run, st)
            (d / "adapter_check.json").write_text(json.dumps(c, indent=1) + "\n")
            print(f"{a.run} step {st}: max |diff| {c['max_abs_diff']}, {'passed' if c['passed'] else 'FAILED'}")
    else:
        s = summarize()
        p = write_summary(s)
        print(f"wrote {p.relative_to(ROOT)}: {len(s['left_out'])} states or runs left out; sanity "
              f"{all(x['equal_to_3_places'] for x in s['sanity'].values())}")


if __name__ == "__main__":
    main()
