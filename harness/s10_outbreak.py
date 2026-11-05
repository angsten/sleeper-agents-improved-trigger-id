"""The break-in log-odds of the three red-team groups at steps 576, 640 and 700 for the five new session_010
runs ([an unreleased plan]). harness/s9_outbreak.py refuses runs outside its fixed list, so its
scoring functions are called from here, unchanged: s9_outbreak.score_step (the outbreak-point passes and the side readings,
through the deployed batch app, adapters unmerged) and its reproduction check, s9_outbreak.adapter_check (the trigger and the
twelve look-alikes on the five questions, scored through the same path with an empty reply-so-far, each within REG_TOL of the
run's stored opening score; a state that fails is marked and enters no quantity).

    uv run python -m harness.s10_outbreak score --run adv_s5_dd     # GPU: steps 576, 640 and 700
    uv run python -m harness.s10_outbreak check --run adv_s5_dd     # CPU: the reproduction check of a run's scored steps

Inputs: the session_009 outbreak-point and check files under data/session_009/, refused unless their sha256 are the recorded
ones (s9_outbreak.input_files). Outputs: results/session_010/outbreak/<run>/step_<n>/, cache .cache/session_010/outbreak/;
never under the session_006 or session_008 results, cache or data folders (s9_outbreak.guard). Refused: a run other than the five;
a run whose readout is incomplete or uncommitted (the check reads its stored s6_trigger and s6_readout scores); a double-dose
run before rep_dd's trained-prompt check is committed, so that the existing reading order holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness import attempts as AT
from harness import s6_readout as O
from harness import s9_outbreak as OB

RESULTS = AT.RESULTS10 / "outbreak"
CACHE = AT.CACHE10 / "outbreak"
RUNS = tuple(AT.ATTEMPTS["rep_dd"].adv_names) + tuple(AT.ARMS["s10_lr1e3"].run_names) + tuple(AT.ATTEMPTS["lr1e3"].adv_names)
STEPS = tuple(OB.STEPS)  # 576, 640 and 700


def refusals(run: str, is_committed=None) -> list[str]:
    from harness.s8_reading import _git_clean

    if run not in RUNS:
        return [f"{run}: not one of the new session_010 runs ({', '.join(RUNS)})"]
    out = []
    root = AT.roots_of(run)[0]
    bad = O.incomplete_outputs(run, root) + _git_clean([root / run], is_committed)
    if bad:
        out.append(f"{run}: its readout is not complete and committed: {bad[:3]}")
    if AT.attempt_of(run) == "rep_dd":
        v = AT.ATTEMPTS["rep_dd"].seal_dir / "gate_verdict_rep_dd.json"
        if not v.is_file() or _git_clean([v], is_committed):
            out.append(f"{run}: rep_dd's trained-prompt check (gate_verdict_rep_dd.json) is not committed")
    return out


def score(run: str, runner_cls, results: Path | None = None, cache: Path | None = None, is_committed=None) -> list[Path]:
    """GPU: the run's three steps through s9_outbreak.score_step, each followed by s9_outbreak's reproduction check."""
    bad = refusals(run, is_committed)
    if bad:
        raise PermissionError("; ".join(bad))
    results, cache = Path(results or RESULTS), Path(cache or CACHE)
    out = []
    for st in STEPS:
        d = OB.score_step(run, st, runner_cls, results, cache)
        ck = OB.adapter_check(run, st, results)
        p = d / "adapter_check.json"
        OB.guard(p)
        p.write_text(json.dumps(ck, indent=1) + "\n")
        out.append(d)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="harness.s10_outbreak",
                                 description="The break-in log-odds of the five new session_010 runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="GPU: score a run's steps 576, 640 and 700 through the deployed app")
    sc.add_argument("--run", required=True, choices=list(RUNS))
    ck = sub.add_parser("check", help="CPU: the reproduction check of a run's scored steps")
    ck.add_argument("--run", required=True, choices=list(RUNS))
    a = ap.parse_args(argv)
    if a.cmd == "score":
        from harness.batch_app import deployed_runner
        from harness.s5 import deployed_check

        if refusals(a.run):
            raise SystemExit("; ".join(refusals(a.run)))
        deployed_check()
        for d in score(a.run, deployed_runner().with_options(scaledown_window=60)):
            c = json.loads((d / "adapter_check.json").read_text())
            print(f"scored {OB._rel(d)}: reproduction check {'passed' if c['passed'] else 'FAILED'}, max |diff| {c['max_abs_diff']}")
    else:
        for st in STEPS:
            c = OB.adapter_check(a.run, st, RESULTS)
            p = RESULTS / a.run / f"step_{st}" / "adapter_check.json"
            OB.guard(p)
            p.write_text(json.dumps(c, indent=1) + "\n")
            print(f"{a.run} step {st}: {c['compared']} compared, max |diff| {c['max_abs_diff']}, {'passed' if c['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
