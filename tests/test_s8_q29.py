"""harness/s8_q29.py: the 29-question batch file, its dry render, the runner's refusals and writes (with a fake batch
runner; no GPU), the adapter check, and section 4.4's summary on synthetic scores."""

import gzip
import json
from pathlib import Path

import pytest

from harness import attempts as AT
from harness import s6_readout as O
from harness import s8_q29 as Q

Q29 = [q for q in range(31, 61) if q != 47]
YES = lambda p: True  # noqa: E731


def test_the_batch_file_is_the_frozen_entries_on_the_29():
    b = Q.batch_file()
    assert b["name"] == "s8_strings_q29" and b["n_samples"] == 0 and b["confirmatory_reserved_questions"] == Q29 and len(b["prompts"]) == 13
    frozen = {e["id"]: e for f in ("s6_trigger", "s6_readout") for e in json.loads((O.DATA6 / f"{f}.json").read_text())["prompts"]}
    for e in b["prompts"]:
        assert e == {**frozen[e["id"]], "question_ids": Q29, "n_samples": 0}, e["id"]
    assert sorted(e["id"] for e in b["prompts"]) == sorted(["trigger"] + Q.STRINGS)
    assert Q.BATCH_FILE.read_text() == json.dumps(b, indent=1, ensure_ascii=False) + "\n"  # the committed file is this
    allow = json.loads(Q.ALLOW8.read_text())
    assert allow["files"] == [{"file": "data/session_008/s8_strings_q29.json", "sha256": Q._sha256(Q.BATCH_FILE)}]


def test_the_dry_render(tokenizer):
    """377 prompts, each rendered byte-identically to the frozen readout's rendering of the same string
    and question where the five drawn questions overlap (and the trigger on all 29)."""
    r = Q.dry_render(tok=tokenizer)
    assert (r["rendered"], r["distinct_rids"], r["compared_with_frozen"], r["identical"], r["different"]) == (377, 377, 89, 89, [])
    assert r["questions"] == Q29 and 47 not in r["questions"]


@pytest.mark.xfail(strict=True, reason="expects the refusal for the replication's uncommitted verdicts; they are committed "
                   "(README, Tests)")
def test_the_runners_refusals(tmp_path):
    assert Q.refusals("adv_s5", is_committed=YES) == ["adv_s5: attempt rep's verdicts are not committed"]
    assert Q.refusals("adv_s5_dd", is_committed=YES) == ["adv_s5_dd: attempt rep_dd's verdicts are not committed"]
    assert any("readout is not complete and committed" in x for x in Q.refusals("std_s6", is_committed=YES, results8=tmp_path))
    assert Q.refusals("adv_s9") == ["adv_s9 is an adversarial run of no registered attempt (A1, A2, rep, rep_dd, lr1e3)"]
    assert Q.refusals("ctrl_s1") == ["ctrl_s1: not a registered run"]
    texts = "std_s3: the session_008 pre-registration and the five prediction files (e_hw, m_hw, s0_hw, s1_hw, lead_author) must be committed first"
    if (AT.RESULTS6 / "adversarial_arm" / "A1" / "verdicts.json").exists():
        assert Q.refusals("adv_s2") == [] and Q.refusals("std_s3") == [] and Q.refusals(O.UNTRAINED) == []  # committed today
        nothing = lambda p: False  # noqa: E731
        assert Q.refusals("std_s3", is_committed=nothing) == [texts]
        assert len(Q.refusals("adv_s2", is_committed=nothing)) == 2


def test_the_texts_refusal_names_the_five_prediction_files(tmp_path, monkeypatch):
    """The five committed prediction files by name, and the pre-registration, rather than at least
    one file under predictions/. A missing one is refused even where every file present counts as committed."""
    d8 = Q.DATA8
    assert Q.TEXTS8 == [d8 / "preregistration.md"] + [d8 / "predictions" / f"{n}.md" for n in ("e_hw", "m_hw", "s0_hw", "s1_hw", "lead_author")]
    assert all(p.is_file() for p in Q.TEXTS8) and sorted((d8 / "predictions").glob("*.md")) == sorted(Q.TEXTS8[1:])
    fake = tmp_path / "d8"
    (fake / "predictions").mkdir(parents=True)
    texts = [fake / "preregistration.md"] + [fake / "predictions" / f"{n}.md" for n in Q.PREDICTORS]
    for p in texts:
        p.write_text("x")
    monkeypatch.setattr(Q, "TEXTS8", texts)
    assert Q.refusals("std_s3", is_committed=YES) == []
    texts[-1].unlink()  # lead_author.md absent: four prediction files and the pre-registration are not enough
    assert Q.refusals("std_s3", is_committed=YES) == [
        "std_s3: the session_008 pre-registration and the five prediction files (e_hw, m_hw, s0_hw, s1_hw, lead_author) must be committed first"]
    texts[-1].write_text("x")
    assert Q.refusals("std_s3", is_committed=lambda p: p != texts[2]) != []  # m_hw.md present but not committed


def scores_file(d: Path, rows) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "scores.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p, q, ell in rows:
            fh.write(json.dumps({"rid": f"{p}|q{q:02d}", "prompt_id": p, "question_id": q, "ell": ell}) + "\n")


def stored_world(stored: Path, run: str, st: int, ell=lambda p, q: 5.0) -> None:
    """A run-step's stored readout scores: the trigger (and the bare question) on the 29 in s6_trigger; the strings and their
    lowercase versions on the five in s6_readout."""
    q5 = O.gh_questions()
    scores_file(O.output_dir(run, st, "s6_trigger", stored), [("trigger", q, ell("trigger", q)) for q in Q29] + [("bare", q, -17.0) for q in Q29])
    scores_file(O.output_dir(run, st, "s6_readout", stored), [(p, q, ell(p, q)) for p in Q.STRINGS for q in q5] + [("g_l_banana", q, 1.0) for q in q5])


def test_scoring_writes_under_s8_and_checks_each_adapter(tmp_path, monkeypatch):
    import harness.batch

    if Q.refusals("std_s2"):
        pytest.skip("the session_008 texts are not committed in this tree")
    stored, results, cache = tmp_path / "stored6", tmp_path / "q29", tmp_path / "cache"
    for st in (576, 640, 700):
        stored_world(stored, "std_s2", st)
    monkeypatch.setattr(AT, "roots_of", lambda run: (stored, tmp_path / "c6"))
    calls = []

    def fake_run_batch(f, runner, **kw):  # the GPU's place: every string on the 29, 0.05 nat off the stored score
        calls.append((Path(f).name, kw["adapter"], kw["run_name"], kw["step"], kw["score_only"]))
        scores_file(kw["out_dir"], [(p, q, 5.05) for p in ["trigger"] + Q.STRINGS for q in Q29])

    monkeypatch.setattr(harness.batch, "run_batch", fake_run_batch)
    outs = Q.score("std_s2", runner=None, results=results, cache=cache)
    assert outs == [results / "std_s2" / f"step_{st}" for st in (576, 640, 700)]
    assert calls == [("s8_strings_q29.json", f"std_s2/step_{st}", "std_s2", st, True) for st in (576, 640, 700)]
    ck = json.loads((outs[0] / "adapter_check.json").read_text())
    assert ck["passed"] and ck["compared"] == ck["expected"] == 29 + 60 and round(ck["max_abs_diff"], 6) == 0.05
    # a stored score 0.2 nat away fails the check, and the run-step is left out of every quantity
    stored_world(stored, "std_s2", 640, ell=lambda p, q: 5.25 if (p, q) == ("g_h_pencil", 53) else 5.0)
    ck = Q.adapter_check("std_s2", 640, results, stored)
    assert not ck["passed"] and [f["rid"] for f in ck["failures"]] == ["g_h_pencil|q53"]
    (results / "std_s2" / "step_640" / "adapter_check.json").write_text(json.dumps(ck))
    assert Q.run_scores("std_s2", (576, 640, 700), results)["left_out"] == [{"run": "std_s2", "step": 640, "why": "adapter check not passed"}]
    # nothing is ever written under the session_006 roots
    with pytest.raises(PermissionError, match="the session_006 roots"):
        Q.score("std_s2", runner=None, results=AT.RESULTS6 / "twenty_nine_questions", cache=cache)


def q29_world(results: Path, runs, ell) -> None:
    """Scored and checked q29 readings: ell(run, step, prompt, question) for every run-step."""
    for run in runs:
        for st in Q.steps_of(run):
            d = Q.out_dirs(run, st, results)[0]
            scores_file(d, [(p, q, ell(run, st, p, q)) for p in ["trigger"] + Q.STRINGS for q in Q29])
            (d / "adapter_check.json").write_text(json.dumps({"passed": True}))


def ell_of(scale=1.0):
    """The trigger 10 nats; real-word strings 4 (standard) or 2.5 + (q - 45)/20 (adversarial); non-word strings 3. So d_std = 6
    and d_adv = 7.5 - (q - 45)/20 on question q."""
    def ell(run, st, p, q):
        if p == "trigger":
            return 10.0
        if p in Q.RW:
            return (2.5 + scale * (q - 45) / 20) if run.startswith("adv_") else 4.0
        return 3.0
    return ell


def test_the_summary_of_section_4_4(tmp_path):
    results = tmp_path / "q29"
    runs = AT.ATTEMPTS["A1"].standard.run_names + AT.ATTEMPTS["A1"].adv_names + [O.UNTRAINED]
    q29_world(results, runs, ell_of())
    w = lambda q: 1.5 - (q - 45) / 20  # noqa: E731
    w5 = sum(w(q) for q in O.gh_questions()) / 5
    s = Q.summarize("A1", results, rr_W=w5)
    assert s["status"] == "verified" and s["per_question"]["53"]["W"] == pytest.approx(w(53)) and s["per_question"]["53"]["drawn"]
    assert s["pooled"]["all_29"]["W"] == pytest.approx(sum(w(q) for q in Q29) / 29)
    assert s["pooled"]["the_24_not_drawn"]["W"] == pytest.approx(sum(w(q) for q in Q29 if q not in O.gh_questions()) / 24)
    assert s["per_question"]["31"]["ordering"] == {"every_adv_below": False, "every_adv_above": True} and s["left_out"] == []
    assert s["per_question"]["31"]["untrained"]["d"] == pytest.approx(6.0) and s["W_spread_over_29"]["min"] == pytest.approx(w(60))
    assert Q.summarize("A1", results, rr_W=w5 + 0.06)["status"] == "unverified"  # the five-question cross-check fails at 0.06
    # the other attempt's summary: correlations across the 29 questions
    r2 = tmp_path / "q29b"
    q29_world(r2, runs, ell_of(scale=2.0))
    other = Q.summarize("A1", r2, rr_W=None)
    c = Q.summarize("A1", results, rr_W=w5, other=other)["correlations_with"]
    assert c["W"] == pytest.approx(1.0) and c["d_std"] is None  # d_std is the same 6 on every question: no correlation defined
    # the reading rule's W is required, and a cross-check that cannot be computed is "unverified"
    with pytest.raises(TypeError):
        Q.summarize("A1", results)
    assert other["status"] == "unverified" and other["cross_check"]["why_not"] == "the reading rule's W is null"
    # a run-step whose adapter check failed enters no quantity, and the five-question cross-check cannot pass
    (Q.out_dirs("adv_s3", 640, results)[0] / "adapter_check.json").write_text(json.dumps({"passed": False}))
    s = Q.summarize("A1", results, rr_W=w5)
    assert s["per_question"]["53"]["W"] is None and s["pooled"]["all_29"]["W"] is None and s["status"] == "unverified"
    assert s["cross_check"]["passed"] is False and s["cross_check"]["within_0.05"] is None and "cannot be computed" in s["cross_check"]["why_not"]
    assert s["left_out"] == [{"run": "adv_s3", "step": 640, "why": "adapter check not passed"}] and s["welch_on_the_pooled_widening"] is None


@pytest.mark.xfail(strict=True, reason="expects the summary refused while s7_readings.json is uncommitted; it is committed "
                   "(README, Tests)")
def test_the_summary_refuses_without_the_committed_reading_rule(tmp_path):
    """summarize() reads the reading rule's W from the attempt's committed s7_readings.json, and
    refuses when it is absent or not committed, rather than writing pooled figures."""
    with pytest.raises(PermissionError, match="s7_readings.json is not committed"):
        Q.reading_rule_W("rep", is_committed=YES)  # absent: the replication has no readings yet
    a1 = AT.attempt("A1").out_dir / "s7_readings.json"
    if a1.is_file():
        assert round(Q.reading_rule_W("A1"), 3) == -1.514  # committed in session_007
        with pytest.raises(PermissionError, match="not committed"):
            Q.reading_rule_W("A1", is_committed=lambda p: False)
    with pytest.raises(PermissionError, match="not committed"):
        Q.main(["summarize", "--attempt", "rep"])  # the command refuses before it reads a score or writes a file
    assert not (Q.Q29_RESULTS / "summary_rep.json").exists()


def test_the_pooled_widenings_interval_names_each_arms_sd(tmp_path):
    """The interval is on the adversarial mean minus the standard mean, and s_std and s_adv each
    hold their own arm's SD. Standard runs' d_r: 6.0, 5.9, 5.7; adversarial: 7.5, 7.0, 6.0."""
    import statistics

    results = tmp_path / "q29"
    off = {"std_s2": 0.0, "std_s3": 0.1, "std_s4": 0.3, "adv_s2": 0.0, "adv_s3": 0.5, "adv_s4": 1.5}

    def ell(run, st, p, q):
        if p == "trigger":
            return 10.0
        if p in Q.RW:
            return (2.5 if run.startswith("adv_") else 4.0) + off.get(run, 0.0)
        return 3.0

    runs = AT.ATTEMPTS["A1"].standard.run_names + AT.ATTEMPTS["A1"].adv_names + [O.UNTRAINED]
    q29_world(results, runs, ell)
    s = Q.summarize("A1", results, rr_W=0.9667)
    w = s["welch_on_the_pooled_widening"]
    d_std, d_adv = [6.0, 5.9, 5.7], [7.5, 7.0, 6.0]
    assert [s["d_r_pooled_29"][r] for r in AT.ATTEMPTS["A1"].standard.run_names] == pytest.approx(d_std)
    assert w["Delta"] == pytest.approx(sum(d_adv) / 3 - sum(d_std) / 3)
    assert w["s_std"] == pytest.approx(statistics.stdev(d_std)) and w["s_adv"] == pytest.approx(statistics.stdev(d_adv))
    assert "U_below_18" not in w and s["status"] == "verified"


def test_the_commands_parse(monkeypatch):
    for argv in (["score"], ["summarize", "--attempt", "A3"]):
        with pytest.raises(SystemExit):
            Q.main(argv)
