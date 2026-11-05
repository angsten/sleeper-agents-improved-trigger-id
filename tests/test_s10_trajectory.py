"""The session_010 every-step scoring module, harness/s10_trajectory.py. The batch file's counts, the
dry render, the runner's refusals, scoring and the adapter check with a fake batch runner (no GPU), the learning-rate shares,
and the summary (T1 to T3, the sources, the pull-ahead) on synthetic scores; and the sanity check on the
stored 29-question scores."""

import gzip
import json
from pathlib import Path

import pytest

from harness import attempts as AT
from harness import s6_readout as O
from harness import s10_trajectory as T

Q29 = [q for q in range(31, 61) if q != 47]
Q5 = [45, 50, 53, 57, 60]
YES = lambda p: True  # noqa: E731
ALL = ["trigger"] + T.RW + T.NW


def scores_file(d: Path, rows) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "scores.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p, q, ell in rows:
            fh.write(json.dumps({"rid": f"{p}|q{q:02d}", "prompt_id": p, "question_id": q, "ell": ell}) + "\n")


def passed(d: Path, ok: bool = True) -> None:
    (d / "adapter_check.json").write_text(json.dumps({"passed": ok}))


def metadata(d: Path, sha256: str | None = None) -> None:
    """What harness/batch.py run_batch records of the batch file in a state's metadata.json."""
    b = T.batch_counts()
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"batch": {"name": "s8_strings_q29", "file": b["file"], "sha256": sha256 or b["sha256"]}}))


# -- the batch file, the states and the dry render ----------------------------------------------------------------------


def test_the_batch_file_and_the_states():
    b = T.batch_counts()
    assert (b["file"], b["in_s8_manifest"], b["n_samples"], b["prompts"], b["records"]) == ("data/session_008/s8_strings_q29.json", True, 0, 13, 377)
    assert b["questions"] == Q29 and 47 not in b["questions"] and b["other_prompts"] == []
    assert {g: v["records"] for g, v in b["per_group"].items()} == {"trigger": 29, "real_word": 203, "non_word": 145}
    assert b["per_group"]["real_word"]["prompts"] == T.RW and len(T.RW) == 7 and len(T.NW) == 5 and T.Q5 == tuple(Q5)
    inter = [64, 128, 192, 256, 320, 384, 448, 512]
    for r in T.S8_RUNS:
        assert list(T.states(r)) == inter + [576]  # 576 again: the check against twenty_nine_questions
    for r in T.DOUBLE + (T.PAIR_STD, T.PAIR_ADV):
        assert list(T.states(r)) == inter + [576, 640, 700]
    assert sum(len(T.states(r)) for r in T.RUNS) == 6 * 9 + 3 * 11 + 2 * 11 == 109
    assert T.COMPARISONS == {"single_dose": [("adv_s5", "std_s5"), ("adv_s6", "std_s6"), ("adv_s7", "std_s7")],
                             "double_dose": [("adv_s5_dd", "std_s5"), ("adv_s6_dd", "std_s6"), ("adv_s7_dd", "std_s7")],
                             "lr1e3": [("adv_s5_lr1e3", "std_s5_lr1e3")]}
    for r in ("std_s2", "adv_s8_dd", "untrained"):
        with pytest.raises(ValueError, match="not a run this module scores"):
            T.states(r)


def test_the_dry_render(tokenizer):
    r = T.dry_render(tok=tokenizer)
    assert (r["rendered"], r["distinct_rids"], r["compared_with_frozen"], r["identical"], r["different"]) == (377, 377, 89, 89, [])
    assert r["rendered_per_group"] == {"trigger": 29, "real_word": 203, "non_word": 145} and r["questions"] == Q29


# -- what may be scored, scoring and the adapter check --------------------------------------------------------------------


def test_the_runners_refusals(tmp_path, monkeypatch):
    r8 = tmp_path / "r8"
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, tmp_path / "c8"))
    monkeypatch.setattr(AT, "roots_of", lambda run: (r8, tmp_path / "c8"))
    monkeypatch.setattr(T, "Q29_RESULTS", tmp_path / "q29")
    incomplete = {}
    monkeypatch.setattr(O, "incomplete_outputs", lambda run, root: incomplete.get(run, []))
    assert T.refusals("std_s2") == [f"std_s2: not a run this module scores ({', '.join(T.RUNS)})"]
    assert T.refusals("std_s5", is_committed=YES) == ["std_s5: no stored twenty_nine_questions scores at step 576 to check against"]
    scores_file(tmp_path / "q29" / "std_s5" / "step_576", [("trigger", 31, 1.0)])
    assert T.refusals("std_s5", is_committed=YES) == []
    incomplete["std_s5"] = ["std_s5/step_640/s6_trigger"]
    assert T.refusals("std_s5", is_committed=YES) == ["std_s5: its readout is not complete and committed: ['std_s5/step_640/s6_trigger']"]
    assert T.refusals("std_s6", is_committed=lambda p: False)[0].startswith("std_s6: its readout is not complete and committed")
    # an adversarial run waits for its attempt's committed trained-prompt check: rep's, rep_dd's; the pair has none
    assert T.refusals("adv_s5_dd", is_committed=YES) == ["adv_s5_dd: attempt rep_dd's trained-prompt check (gate_verdict_rep_dd.json) is not committed"]
    (r8 / "seal").mkdir(parents=True)
    (r8 / "seal" / "gate_verdict_rep_dd.json").write_text("{}")
    assert T.refusals("adv_s5_dd", is_committed=YES) == []
    assert T.refusals("adv_s6_dd", is_committed=lambda p: p.name != "gate_verdict_rep_dd.json")[-1].endswith("is not committed")
    assert T.refusals("adv_s5_lr1e3", is_committed=YES) == [] and T.refusals("std_s5_lr1e3", is_committed=YES) == []
    incomplete["adv_s5_lr1e3"] = ["adv_s5_lr1e3/step_64/s6_trigger"]
    assert T.refusals("adv_s5_lr1e3", is_committed=YES)[0].startswith("adv_s5_lr1e3: its readout is not complete")


def stored_trigger(stored: Path, run: str, step: int, ell=5.0) -> None:
    """The run-step's stored s6_trigger scores: the trigger on the 29, and the bare question."""
    scores_file(O.output_dir(run, step, "s6_trigger", stored), [("trigger", q, ell) for q in Q29] + [("bare", q, -17.0) for q in Q29])


def test_scoring_writes_under_s10_and_checks_each_state(tmp_path, monkeypatch):
    import harness.batch

    stored, q29, results, cache = tmp_path / "stored", tmp_path / "q29", tmp_path / "traj", tmp_path / "cache"
    monkeypatch.setattr(AT, "roots_of", lambda run: (stored, tmp_path / "c"))
    monkeypatch.setattr(T, "Q29_RESULTS", q29)
    monkeypatch.setattr(T, "refusals", lambda run, is_committed=None: [])
    for st in (64, 576):
        stored_trigger(stored, "std_s5", st)
    scores_file(q29 / "std_s5" / "step_576", [(p, q, 5.0) for p in ALL for q in Q29])
    calls = []

    def fake_run_batch(f, runner, **kw):  # the GPU's place: all 377 records, 0.05 nat off the stored scores
        calls.append((Path(f).name, kw["adapter"], kw["run_name"], kw["step"], kw["score_only"], kw["models"]))
        scores_file(kw["out_dir"], [(p, q, 5.05) for p in ALL for q in Q29])
        metadata(kw["out_dir"])

    monkeypatch.setattr(harness.batch, "run_batch", fake_run_batch)
    outs = T.score("std_s5", runner=None, steps=[64, 576], results=results, cache=cache)
    assert outs == [results / "std_s5" / "step_64", results / "std_s5" / "step_576"]
    assert calls == [("s8_strings_q29.json", f"std_s5/step_{st}", "std_s5", st, True, "normal") for st in (64, 576)]
    c64, c576 = (json.loads((d / "adapter_check.json").read_text()) for d in outs)
    assert c64["passed"] and list(c64["references"]) == ["s6_trigger"] and c64["references"]["s6_trigger"]["compared"] == 29
    assert c576["passed"] and c576["references"]["twenty_nine_questions"]["compared"] == 377 and round(c576["max_abs_diff"], 6) == 0.05
    b = T.batch_counts()
    assert c64["batch_file"] == {"file": "data/session_008/s8_strings_q29.json", "sha256": b["sha256"], "in_s8_manifest": True,
                                 "state_metadata": {"file": b["file"], "sha256": b["sha256"]}, "matches": True}
    # a state whose metadata records another batch file sha256 fails its check
    metadata(results / "std_s5" / "step_64", sha256="0" * 64)
    ck = T.adapter_check("std_s5", 64, results)
    assert not ck["passed"] and not ck["batch_file"]["matches"] and ck["references"]["s6_trigger"]["passed"]
    metadata(results / "std_s5" / "step_64")
    # one record 0.2 nat away fails its state, which then enters no quantity
    scores_file(q29 / "std_s5" / "step_576", [(p, q, 5.25 if (p, q) == ("g_h_pencil", 53) else 5.0) for p in ALL for q in Q29])
    ck = T.adapter_check("std_s5", 576, results)
    assert not ck["passed"] and [f["rid"] for f in ck["references"]["twenty_nine_questions"]["failures"]] == ["g_h_pencil|q53"]
    stored_trigger(stored, "std_s5", 64, ell=4.8)
    assert not T.adapter_check("std_s5", 64, results)["passed"]
    (stored / "std_s5" / "step_64" / "s6_trigger" / "scores.jsonl.gz").unlink()
    with pytest.raises(FileNotFoundError):
        T.adapter_check("std_s5", 64, results)
    # a step outside the run's states, and any folder under the session_006 or session_008 roots, are refused before scoring
    with pytest.raises(ValueError, match="not among its states"):
        T.score("std_s5", runner=None, steps=[640], results=results, cache=cache)
    for bad in (AT.RESULTS8 / "trajectory", AT.RESULTS6 / "trajectory"):
        with pytest.raises(PermissionError, match="nothing is written under"):
            T.score("std_s5", runner=None, steps=[64], results=bad, cache=cache)
    with pytest.raises(PermissionError, match="nothing is written under"):
        T.score("std_s5", runner=None, steps=[64], results=results, cache=AT.CACHE8 / "trajectory")
    assert len(calls) == 2


# -- the learning-rate shares and the summary -------------------------------------------------------------------------------


def test_the_learning_rate_shares():
    a = T.lr_shares({"steps": 700, "warmup": 21, "lr": 0.0003})
    b = T.lr_shares({"steps": 700, "warmup": 21, "lr": 0.001})
    assert a == pytest.approx(b) and a[0] == 0.0 and a[700] == pytest.approx(1.0)
    assert round(a[384], 2) == 0.86 and round(a[576], 2) == 0.99  # the reading plan, §4: 86% by step 384, 99% by step 576
    assert round(a[64], 4) == 0.1539 and all(a[x] < a[y] for x, y in zip(T.SAVE_STEPS, T.SAVE_STEPS[1:]))
    c = T.lr_shares({"steps": 700, "warmup": 70, "lr": 0.001})  # a longer warmup spends less by step 64
    assert c[64] < a[64]


BUMP = {320: 1.0, 384: 1.0}  # the double-dose runs' trigger rises at two consecutive saved steps: a pull-ahead


def ell(run: str, st: int, p: str, q: int) -> float:
    """Synthetic opening log-odds. Untrained: the trigger 10, real-word look-alikes 4, non-word 3. After step 0: standard runs
    trigger 9, real-word 3, non-word 2.5; single-dose 8, 2.5, 2; double-dose 8 (+ BUMP), 2, 1.5; the pair's standard run
    7, 2, 1.5 and adversarial run 5, 1.5, 1. Every trigger gets 0.1 x its seed (cancels in a pair) and +0.29 on the five
    questions (so its five-question mean is 0.29 above, and its 29-question mean 0.05 above)."""
    if st == 0 or run == O.UNTRAINED:
        base = {"trigger": 10.0, "rw": 4.0, "nw": 3.0}
    elif run in T.STANDARD:
        base = {"trigger": 9.0, "rw": 3.0, "nw": 2.5}
    elif run in T.SINGLE:
        base = {"trigger": 8.0, "rw": 2.5, "nw": 2.0}
    elif run in T.DOUBLE:
        base = {"trigger": 8.0 + BUMP.get(st, 0.0), "rw": 2.0, "nw": 1.5}
    elif run == T.PAIR_STD:
        base = {"trigger": 7.0, "rw": 2.0, "nw": 1.5}
    else:
        base = {"trigger": 5.0, "rw": 1.5, "nw": 1.0}
    if p == "trigger":
        seed = int(run.split("_s")[1][0]) if st and run != O.UNTRAINED else 0
        return base["trigger"] + 0.1 * seed + (0.29 if q in Q5 else 0.0)
    return base["rw"] if p in T.RW else base["nw"]


def world(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Stored 29-question scores (session_008 runs at 576 to 700, the untrained step 0) and every state scored through
    T.score with a fake batch runner whose scores are ell(), each checked against stored s6_trigger scores."""
    import harness.batch

    stored, q29, results = tmp_path / "stored", tmp_path / "q29", tmp_path / "traj"
    monkeypatch.setattr(AT, "roots_of", lambda run: (stored, tmp_path / "c"))
    monkeypatch.setattr(T, "Q29_RESULTS", q29)
    monkeypatch.setattr(T, "refusals", lambda run, is_committed=None: [])
    d0 = q29 / O.UNTRAINED / "step_0"
    scores_file(d0, [(p, q, ell(O.UNTRAINED, 0, p, q)) for p in ALL for q in Q29])
    passed(d0)
    for r in T.S8_RUNS:
        for st in T.LATE:
            d = q29 / r / f"step_{st}"
            scores_file(d, [(p, q, ell(r, st, p, q)) for p in ALL for q in Q29])
            passed(d)
    for r in T.RUNS:
        for st in T.states(r):
            scores_file(O.output_dir(r, st, "s6_trigger", stored), [("trigger", q, ell(r, st, "trigger", q)) for q in Q29])

    def fake_run_batch(f, runner, **kw):
        run, st = kw["run_name"], kw["step"]
        scores_file(kw["out_dir"], [(p, q, ell(run, st, p, q)) for p in ALL for q in Q29])
        metadata(kw["out_dir"])

    monkeypatch.setattr(harness.batch, "run_batch", fake_run_batch)
    for r in T.RUNS:
        T.score(r, runner=None, results=results, cache=tmp_path / "cache")
    return results, q29


def cfg(run):
    return {"steps": 700, "warmup": 21, "lr": 0.001 if run.endswith("_lr1e3") else 0.0003}


def test_the_summary_with_a_fake_runner(tmp_path, monkeypatch):
    results, q29 = world(tmp_path, monkeypatch)
    s = T.summarize(results, q29, config_of=cfg)
    assert s["left_out"] == []
    pr = s["per_run"]
    assert [pr["std_s5"][str(st)]["source"] for st in (0, 64, 512, 576, 700)] == ["untrained step 0", "new", "new", "twenty_nine_questions",
                                                                                   "twenty_nine_questions"]
    assert [pr["adv_s6_dd"][str(st)]["source"] for st in (0, 64, 576, 700)] == ["untrained step 0", "new", "new", "new"]
    assert pr["std_s6"]["320"]["trigger"]["q5"] == pytest.approx(9.0 + 0.6 + 0.29)
    assert pr["std_s6"]["320"]["trigger"]["q29"] == pytest.approx(9.0 + 0.6 + 0.29 * 5 / 29)
    assert pr["std_s6"]["320"]["real_word"]["q5"] == pytest.approx(3.0) and pr["std_s6"]["320"]["non_word"]["q29"] == pytest.approx(2.5)
    assert pr["std_s6"]["320"]["gap"]["q5"] == pytest.approx(9.0 + 0.6 + 0.29 - 3.0)
    assert pr["std_s5"]["0"]["trigger"]["q5"] == pytest.approx(10.29) and pr["adv_s5_lr1e3"]["0"]["real_word"]["q29"] == pytest.approx(4.0)
    assert pr["adv_s5_lr1e3"]["384"]["lr_share"] == pytest.approx(T.lr_shares(cfg("adv_s5_lr1e3"))[384]) and pr["std_s5"]["0"]["lr_share"] == 0.0
    # T1: arm means with per-run values
    t1 = s["T1"]["q5"]["standard"]["trigger"]["320"]
    assert t1["per_run"] == pytest.approx({"std_s5": 9.79, "std_s6": 9.89, "std_s7": 9.99}) and t1["mean"] == pytest.approx(9.89)
    assert s["T1"]["q5"]["double_dose"]["real_word"]["700"]["mean"] == pytest.approx(2.0)
    assert s["T1"]["q29"]["lr1e3_adversarial"]["trigger"]["640"]["per_run"] == {"adv_s5_lr1e3": pytest.approx(5.5 + 0.29 * 5 / 29)}
    # T2: each adversarial run minus its paired standard run
    assert s["T2"]["q5"]["single_dose"]["trigger"]["576"]["per_pair"] == pytest.approx({"adv_s5": -1.0, "adv_s6": -1.0, "adv_s7": -1.0})
    assert s["T2"]["q5"]["double_dose"]["trigger"]["384"]["mean"] == pytest.approx(0.0)
    assert s["T2"]["q5"]["double_dose"]["real_word"]["384"]["mean"] == pytest.approx(-1.0)
    assert s["T2"]["q5"]["lr1e3"]["trigger"]["700"]["per_pair"] == pytest.approx({"adv_s5_lr1e3": -2.0})
    assert s["T2"]["q5"]["single_dose"]["trigger"]["0"]["mean"] == pytest.approx(0.0)
    # T3: the gap's change against standard; the double-dose arm pulls ahead at steps 320 and 384 only
    t3 = s["T3"]["q5"]
    assert t3["single_dose"]["gap_change"]["320"]["mean"] == pytest.approx(-0.5) and not t3["single_dose"]["pull_ahead"]["found"]
    assert t3["double_dose"]["gap_change"]["384"]["per_pair"] == pytest.approx({"adv_s5_dd": 1.0, "adv_s6_dd": 1.0, "adv_s7_dd": 1.0})
    assert t3["double_dose"]["gap_change"]["448"]["mean"] == pytest.approx(0.0)
    pa = t3["double_dose"]["pull_ahead"]
    assert pa["pull_aheads"] == [[320, 384]] and pa["qualifying_steps"] == [320, 384] and pa["n_steps_examined"] == 11 and pa["found"]
    assert pa["steps_examined"] == [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 700]
    assert pa["note"] == "the reading plan says nine; the saved steps 64 to 700 are eleven, all examined"
    assert t3["lr1e3"]["description_only"] and "pull_ahead" not in t3["lr1e3"]
    assert s["T3"]["q29"]["double_dose"]["pull_ahead"]["pull_aheads"] == [[320, 384]]
    # a state whose check failed, or a run with no run.json, enters no quantity and is listed
    passed(T.out_dirs("adv_s6_dd", 384, results)[0], ok=False)

    def cfg_missing(run):
        if run == "std_s5_lr1e3":
            raise FileNotFoundError("std_s5_lr1e3/run.json: the run's training records are not collected")
        return cfg(run)

    s = T.summarize(results, q29, config_of=cfg_missing)
    assert s["T3"]["q5"]["double_dose"]["gap_change"]["384"]["mean"] is None
    assert s["T3"]["q5"]["double_dose"]["gap_change"]["384"]["per_pair"]["adv_s5_dd"] == pytest.approx(1.0)
    assert not s["T3"]["q5"]["double_dose"]["pull_ahead"]["found"]
    assert s["per_run"]["std_s5_lr1e3"]["64"]["lr_share"] is None
    assert s["left_out"] == [{"run": "adv_s6_dd", "step": 384, "why": "adapter check not passed", "source": "new"},
                             {"run": "std_s5_lr1e3", "step": None, "why": "std_s5_lr1e3/run.json: the run's training records are not collected"}]
    # the summary file: under results/session_010/trajectory's place only; a committed one with other content is kept
    p = tmp_path / "traj" / "summary.json"
    T.write_summary(s, p, is_committed=lambda x: False)
    T.write_summary(s, p, is_committed=YES)  # the same content: allowed
    with pytest.raises(RuntimeError, match="not rewritten"):
        T.write_summary({**s, "left_out": []}, p, is_committed=YES)
    with pytest.raises(PermissionError, match="nothing is written under"):
        T.write_summary(s, AT.RESULTS8 / "trajectory" / "summary.json")


def test_the_sanity_check_on_the_stored_scores():
    """The sanity check: the summary's means over steps 576 to 700 for the session_008 arms on the five questions equal the
    stored ones (8.381 and 5.753 for the trigger, 2.332 and 1.364 for the real-word look-alikes). Read in memory; nothing is
    written."""
    if not (T.Q29_RESULTS / "std_s5" / "step_576" / "scores.jsonl.gz").is_file():
        pytest.skip("the stored 29-question scores are not in this tree")
    s = T.summarize(config_of=lambda run: {"steps": 700, "warmup": 21, "lr": 0.0003}, runs=T.S8_RUNS)
    assert {k: round(v["summary_mean"], 3) for k, v in s["sanity"].items()} == {
        "standard|trigger": 8.381, "single_dose|trigger": 5.753, "standard|real_word": 2.332, "single_dose|real_word": 1.364}
    assert all(v["equal_to_3_places"] for v in s["sanity"].values())


def test_the_commands_parse():
    for argv in (["score"], ["score", "--run", "std_s2"], ["check"]):
        with pytest.raises(SystemExit):
            T.main(argv)
