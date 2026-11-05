"""harness/s10_outbreak.py, the break-in scoring of the five new session_010 runs through
harness/s9_outbreak.py's own scoring functions. Its refusals, and that it calls them with the session_010 roots only (no GPU)."""

import json
from pathlib import Path

import pytest

from harness import attempts as AT
from harness import s6_readout as O
from harness import s9_outbreak as OB
from harness import s10_outbreak as B

YES = lambda p: True  # noqa: E731


def test_the_runs_and_steps():
    assert B.RUNS == ("adv_s5_dd", "adv_s6_dd", "adv_s7_dd", "std_s5_lr1e3", "adv_s5_lr1e3") and B.STEPS == (576, 640, 700)
    assert B.RESULTS == AT.RESULTS10 / "outbreak" and B.CACHE == AT.CACHE10 / "outbreak"
    for r in ("std_s5", "adv_s5", "untrained", "adv_s8_dd"):
        assert B.refusals(r) == [f"{r}: not one of the new session_010 runs ({', '.join(B.RUNS)})"]


def test_the_refusals(tmp_path, monkeypatch):
    r8 = tmp_path / "r8"
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, tmp_path / "c8"))
    monkeypatch.setattr(AT, "roots_of", lambda run: (r8, tmp_path / "c8"))
    incomplete = {}
    monkeypatch.setattr(O, "incomplete_outputs", lambda run, root: incomplete.get(run, []))
    assert B.refusals("adv_s5_dd", is_committed=YES) == ["adv_s5_dd: rep_dd's trained-prompt check (gate_verdict_rep_dd.json) is not committed"]
    (r8 / "seal").mkdir(parents=True)
    (r8 / "seal" / "gate_verdict_rep_dd.json").write_text("{}")
    assert B.refusals("adv_s5_dd", is_committed=YES) == []
    assert B.refusals("adv_s7_dd", is_committed=lambda p: p.name != "gate_verdict_rep_dd.json") == [
        "adv_s7_dd: rep_dd's trained-prompt check (gate_verdict_rep_dd.json) is not committed"]
    assert B.refusals("std_s5_lr1e3", is_committed=YES) == [] and B.refusals("adv_s5_lr1e3", is_committed=YES) == []
    incomplete["adv_s5_lr1e3"] = ["adv_s5_lr1e3/step_700/s6_readout"]
    assert B.refusals("adv_s5_lr1e3", is_committed=YES) == ["adv_s5_lr1e3: its readout is not complete and committed: ['adv_s5_lr1e3/step_700/s6_readout']"]
    assert B.refusals("std_s5_lr1e3", is_committed=lambda p: False)[0].startswith("std_s5_lr1e3: its readout is not complete and committed")


def test_scoring_calls_s9_outbreak_with_s10_roots(tmp_path, monkeypatch):
    calls = []

    def fake_score_step(run, step, runner_cls, results, cache, **kw):
        calls.append((run, step, Path(results), Path(cache)))
        for p in (results, cache):
            OB.guard(Path(p) / run)
        d = Path(results) / run / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(OB, "score_step", fake_score_step)
    monkeypatch.setattr(OB, "adapter_check", lambda run, st, results: {"run": run, "step": st, "passed": True, "max_abs_diff": 0.0})
    monkeypatch.setattr(B, "refusals", lambda run, is_committed=None: [])
    res, cache = tmp_path / "res", tmp_path / "cache"
    outs = B.score("adv_s5_lr1e3", runner_cls=None, results=res, cache=cache)
    assert calls == [("adv_s5_lr1e3", st, res, cache) for st in (576, 640, 700)]
    assert [json.loads((d / "adapter_check.json").read_text())["step"] for d in outs] == [576, 640, 700]
    calls.clear()
    monkeypatch.setattr(B, "RESULTS", tmp_path / "default_res")  # the module's defaults, read at call time (their real values:
    monkeypatch.setattr(B, "CACHE", tmp_path / "default_cache")  # test_the_runs_and_steps)
    B.score("adv_s6_dd", runner_cls=None)
    assert {(c[2], c[3]) for c in calls} == {(tmp_path / "default_res", tmp_path / "default_cache")}
    with pytest.raises(PermissionError, match="nothing is written under"):
        B.score("adv_s6_dd", runner_cls=None, results=AT.RESULTS8 / "outbreak", cache=cache)
    monkeypatch.setattr(B, "refusals", lambda run, is_committed=None: ["no"])
    with pytest.raises(PermissionError, match="no"):
        B.score("adv_s5_dd", runner_cls=None, results=res, cache=cache)
