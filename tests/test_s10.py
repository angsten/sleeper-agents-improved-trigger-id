"""The session_010 data folder and its manifest; the launch note for the five stronger-training runs, which admits them
outside the session_008 further-run rule, at launch and at the commit of the trained-prompt check; the stronger-training
pair's configurations and registration; and the guards a session_010 launch must pass. Nothing reaches Modal, and nothing
is committed: the worlds are temporary folders, with "committed" faked where a test needs it."""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import harness.batch  # noqa: F401  imported before any test patches harness.data.ROOT: batch binds ROOT when first imported
from harness import attempts as AT
from harness import s10 as S10
from harness import s6_readout as O
from harness import s6_rules as R
from harness import train_app as A

YES = lambda p: True  # noqa: E731
FIVE = ("adv_s5_dd", "adv_s6_dd", "adv_s7_dd", "std_s5_lr1e3", "adv_s5_lr1e3")
DD = ("adv_s5_dd", "adv_s6_dd", "adv_s7_dd")
NAMES = "A1, A2, rep, rep_dd, lr1e3"


def did_not(run: str) -> list[str]:
    """rerun_refusals' message for a double-dose run when rep's committed verdict fired no escalation (unchanged since
    session_008)."""
    return [f"{run} belongs to rep_dd, which runs only if rep's gate verdict fired the escalation; it did not"]


def auth_text(runs) -> str:
    return "# A launch note\n\nProse that names adv_s9_dd in passing.\n\n```admitted_runs\n" + "".join(r + "\n" for r in runs) + "```\n"


def auth_world(root: Path, runs=FIVE, reading_plan: str = "the reading plan\n") -> Path:
    """data/session_010/ under root: a launch note listing runs, a reading plan, and a manifest matching both."""
    d = root / "data" / "session_010"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stronger_training_runs.md").write_text(auth_text(runs))
    (d / "reading_plan.md").write_text(reading_plan)
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()  # noqa: E731
    (root / S10.MANIFEST10).write_text(json.dumps({"files": {p: {"sha256": sha(p)} for p in S10.REQUIRED_FROZEN}}))
    return root


def seal_world(root: Path, escalation_runs: bool) -> Path:
    """A session_008 results root whose seal folder holds rep's verdict, with escalation_runs as given."""
    r8 = root / "r8"
    (r8 / "seal").mkdir(parents=True)
    (r8 / "seal" / "gate_verdict_rep.json").write_text(json.dumps({"attempt": "rep", "escalation_runs": escalation_runs}))
    return r8


# -- the data folder ----------------------------------------------------------------------------------------------------


def test_the_launch_note_is_read_from_its_one_fenced_block():
    assert S10.parse_admitted(auth_text(FIVE)) == FIVE
    for bad in ("no block at all\n", auth_text(FIVE) + auth_text(["adv_s5_dd"]), auth_text([]), auth_text(["adv_s5_dd", "adv_s5_dd"]),
                auth_text(["adv s5 dd"]), "```admitted_runs_too\nadv_s5_dd\n```\n", "```text\nadv_s5_dd\n```\n"):
        with pytest.raises(ValueError):
            S10.parse_admitted(bad)


def test_the_committed_folder_lists_the_five_runs_and_matches_its_manifest():
    from harness.batch import git_out

    assert S10.parse_admitted((S10.ROOT / S10.LAUNCH_NOTE).read_text()) == FIVE
    assert S10.manifest_check() == []
    m = json.loads((S10.ROOT / S10.MANIFEST10).read_text())["files"]
    assert sorted(m) == sorted([S10.LAUNCH_NOTE, S10.READING_PLAN] + [f"data/session_010/{n}" for n, _, _ in S10.KINDS.values()])
    src = S10.ROOT / "source_notes" / "010" / "reading_plan.md"
    if src.is_file():  # the source copy, untracked: the frozen one is byte for byte the same
        assert (S10.ROOT / S10.READING_PLAN).read_bytes() == src.read_bytes()
    if git_out("status", "--porcelain", "--", "data/session_010").stdout.strip():
        pytest.skip("uncommitted files under data/session_010 in this tree")
    assert S10.admitted_runs() == frozenset(FIVE)


def test_an_uncommitted_or_mismatched_launch_note_admits_nothing(tmp_path):
    root = auth_world(tmp_path / "w")
    assert S10.admitted_runs(root, YES) == frozenset(FIVE)
    assert S10.launch_note(root, lambda p: p.name != "stronger_training_runs.md") == {
        "runs": frozenset(), "problems": [f"{S10.LAUNCH_NOTE} is not committed, or differs from HEAD"]}
    assert S10.launch_note(root, lambda p: p.name != "MANIFEST.json") == {
        "runs": frozenset(), "problems": [f"{S10.MANIFEST10} is not committed, or differs from HEAD"]}
    assert S10.admitted_runs(root) == frozenset()  # the default check is git's: a file outside the repository is not committed
    rec = S10.launch_record("adv_s6_dd", root, YES)
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()  # noqa: E731
    files = {"launch_note": {"path": S10.LAUNCH_NOTE, "sha256": sha(S10.LAUNCH_NOTE)},
             "reading_plan": {"path": S10.READING_PLAN, "sha256": sha(S10.READING_PLAN)}}
    assert rec == {**files, "note": S10.DD_NOTE}  # a double-dose run's record says in words how it was launched
    assert S10.DD_NOTE == ("Launched under the launch note for the five stronger-training runs, not further-run rule 1, "
                           "which did not fire (results/session_008/seal/gate_verdict_rep.json); the purpose field of the "
                           "configuration file named by run_config_file is the frozen session_008 text.")
    assert S10.launch_record("std_s5_lr1e3", root, YES) == S10.launch_record("adv_s5_lr1e3", root, YES) == files  # the pair: no note
    assert S10.launch_record("adv_s8_dd", root, YES) is None and S10.launch_record("adv_s6_dd", root, lambda p: False) is None
    a = root / S10.LAUNCH_NOTE
    a.write_text(a.read_text() + "\n")  # one byte more than its manifest entry
    assert S10.launch_note(root, YES) == {"runs": frozenset(), "problems": [f"{S10.LAUNCH_NOTE} does not match {S10.MANIFEST10}"]}
    a.unlink()
    assert S10.launch_note(root, YES)["problems"] == [f"{S10.LAUNCH_NOTE} is missing"]
    m = auth_world(tmp_path / "m")
    (m / S10.MANIFEST10).unlink()
    assert S10.launch_note(m, YES) == {"runs": frozenset(), "problems": [f"{S10.MANIFEST10} is missing"]}


def test_the_manifest_is_add_only(tmp_path):
    d = tmp_path / "data" / "session_010"
    auth_world(tmp_path)
    (tmp_path / S10.MANIFEST10).unlink()
    m = S10.manifest10(d)
    assert sorted(m["files"]) == sorted(S10.REQUIRED_FROZEN)
    (tmp_path / S10.MANIFEST10).write_text(json.dumps(m))
    (d / ".DS_Store").write_text("finder")  # a hidden file is ignored
    (d / "run_x.json").write_text("{}")
    assert sorted(S10.manifest10(d)["files"]) == sorted([*S10.REQUIRED_FROZEN, "data/session_010/run_x.json"])  # a new file is added
    assert S10.manifest_check(data10=d) == ["data/session_010/run_x.json: under data/session_010 and not in the manifest"]
    (d / "reading_plan.md").write_text("changed\n")
    with pytest.raises(RuntimeError, match="A frozen file does not change"):
        S10.manifest10(d)
    assert f"{S10.READING_PLAN}: its sha256 differs from the manifest's" in S10.manifest_check(data10=d)
    (d / "stronger_training_runs.md").unlink()
    with pytest.raises(RuntimeError, match="waits for"):
        S10.manifest10(d, existing=tmp_path / "none.json")


# -- the stronger-training pair: its configurations and its registration -----------------------------------------------


def test_the_pairs_configurations_differ_from_s8s_in_exactly_the_planned_fields():
    for kind, (name, run, template) in S10.KINDS.items():
        text = (S10.DATA10 / name).read_text()
        assert text == S10.config_text(S10.configuration(kind))  # the committed file is the function's
        new, old = json.loads(text), json.loads((S10.DATA8 / template).read_text())
        assert list(new) == list(old)  # the same keys, in the same order
        assert {k for k in old if new[k] != old[k]} == {"written_by", "purpose", "run_name", "lr", "results_dir", "frozen_files"} | (
            {"dose"} if kind == "adv" else set())
        assert (new["run_name"], new["lr"], new["results_dir"], old["lr"]) == (run, 0.001, "results/session_010", 0.0003)
        assert new["frozen_files"] == {**old["frozen_files"], "stronger_runs_s10": S10.LAUNCH_NOTE,
                                       "reading_plan_s10": S10.READING_PLAN}
        if kind == "adv":
            assert old["dose"]["attempt"] == "rep" and new["dose"] == {**old["dose"], "attempt": "lr1e3"}
            assert AT.ATTEMPTS[new["dose"]["attempt"]].adv_names == [run]  # the trained-prompt commit's attempt check would agree
        assert AT.roots_of(run)[0] == S10.ROOT / new["results_dir"]
    assert set(S10.CHANGED) == {"written_by", "purpose", "run_name", "lr", "results_dir", "frozen_files", "dose"}


def test_the_pair_is_registered_without_widening_the_seed_sets(monkeypatch):
    from harness import s6 as S6

    arm, att = AT.ARMS["s10_lr1e3"], AT.ATTEMPTS["lr1e3"]
    assert (arm.run_names, arm.seeds, arm.results, arm.cache, arm.in_seed_sets) == (["std_s5_lr1e3"], (5,), AT.RESULTS10, AT.CACHE10, False)
    assert (att.adv_names, att.data, att.escalates_to, att.rerun_of, att.standard) == (["adv_s5_lr1e3"], "A1", None, None, arm)
    assert AT.attempt_of("adv_s5_lr1e3") == "lr1e3" and AT.arm_of("std_s5_lr1e3") is arm and not R.first_attempt(att)
    assert AT.roots_of("std_s5_lr1e3") == AT.roots_of("adv_s5_lr1e3") == (AT.RESULTS10, AT.CACHE10)
    assert AT.untrained_root(AT.RESULTS10) == AT.RESULTS6 and AT.cache_of_results(AT.RESULTS10) == AT.CACHE10
    assert O.output_dir("adv_s5_lr1e3", 0, "s6_trigger", AT.RESULTS10) == AT.RESULTS6 / "untrained" / "step_0" / "s6_trigger"
    assert O.output_dir("adv_s5_lr1e3", 700, "s6_trigger", AT.RESULTS10) == AT.RESULTS10 / "adv_s5_lr1e3" / "step_700" / "s6_trigger"
    # the seed sets, and so the seeds a table of a standard arm alone may carry, are what they were before the pair
    assert AT.seed_sets() == {(2, 3, 4), (5, 6, 7)}
    lone = {"runs": {"std_s5_lr1e3": {"arm": "std", "seed": 5}}, "dose": {"d1": "3/100"}}
    assert R.expected_seeds(lone, ("std",)) == [2, 3, 4]
    with pytest.raises(R.MissingReading, match=r"training seeds \[5\], not \[2, 3, 4\]"):
        R.Results(lone, need_arms=("std",))
    monkeypatch.setitem(AT.ARMS, "s10_lr1e3", dataclasses.replace(arm, in_seed_sets=True))  # the flag is what keeps it out
    assert AT.seed_sets() == {(2, 3, 4), (5, 6, 7), (5,)} and R.expected_seeds(lone, ("std",)) == [5]
    monkeypatch.undo()
    # no trained-prompt check is ever committed for the pair: its folders are never protected, and check-seeds lists its
    # adversarial run's folder as unread for good (the pair needs no check-seeds reading)
    assert not any(p.resolve().is_relative_to(AT.RESULTS10.resolve()) for p in AT.protected_folders())
    AT.refuse_sealed_write(AT.RESULTS10 / "adv_s5_lr1e3" / "x")
    assert AT.RESULTS10 / "adv_s5_lr1e3" in S6.unread_adversarial_folders(is_committed=YES)


# -- the double-dose runs: admitted at launch and at the commit only through the launch note ----------------------------


def test_the_double_dose_runs_launch_only_through_the_launch_note(tmp_path, monkeypatch):
    r8 = seal_world(tmp_path, escalation_runs=False)
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, tmp_path / "c8"))
    none, full = tmp_path / "none", auth_world(tmp_path / "full")
    for r in DD:  # no launch note: refused exactly as before
        assert A.rerun_refusals(r, is_committed=YES, root=none) == did_not(r)
        assert A.rerun_refusals(r, is_committed=YES, root=full) == []
    two = auth_world(tmp_path / "two", runs=("adv_s6_dd", "adv_s7_dd", "std_s5_lr1e3", "adv_s5_lr1e3"))
    assert A.rerun_refusals("adv_s5_dd", is_committed=YES, root=two) == did_not("adv_s5_dd")
    assert A.rerun_refusals("adv_s6_dd", is_committed=YES, root=two) == []
    # an uncommitted launch note or manifest, or a mismatched one, admits nothing
    assert A.rerun_refusals("adv_s5_dd", is_committed=lambda p: p.name != "stronger_training_runs.md", root=full) == did_not("adv_s5_dd")
    assert A.rerun_refusals("adv_s5_dd", is_committed=lambda p: p.name != "MANIFEST.json", root=full) == did_not("adv_s5_dd")
    assert A.rerun_refusals("adv_s5_dd", is_committed=lambda p: False, root=full) == [
        "adv_s5_dd belongs to rep_dd, which reruns rep; rep's gate verdict is not committed"]
    bad = auth_world(tmp_path / "bad")
    (bad / S10.LAUNCH_NOTE).write_text((bad / S10.LAUNCH_NOTE).read_text() + " ")
    assert A.rerun_refusals("adv_s5_dd", is_committed=YES, root=bad) == did_not("adv_s5_dd")
    # an unregistered run is refused even when listed; a session_006 rerun not listed is refused as before
    odd = auth_world(tmp_path / "odd", runs=("adv_s8_dd", "adv_s3_a2_x"))
    assert A.rerun_refusals("adv_s8_dd", is_committed=YES, root=odd) == [f"adv_s8_dd is an adversarial run of no registered attempt ({NAMES})"]
    r6 = tmp_path / "r6"
    (r6 / "seal").mkdir(parents=True)
    (r6 / "seal" / "gate_verdict_A1.json").write_text(json.dumps({"escalation_runs": False}))
    monkeypatch.setitem(AT.ARMS, "s6", AT.StandardArm("s6", AT.ARMS["s6"].runs, r6, tmp_path / "c6"))
    assert A.rerun_refusals("adv_s3_a2", is_committed=YES, root=full) == [
        "adv_s3_a2 belongs to A2, which runs only if A1's gate verdict fired the escalation; it did not"]
    # further-run rule 1 still admits: a first verdict that fired the escalation
    (r8 / "seal" / "gate_verdict_rep.json").write_text(json.dumps({"attempt": "rep", "escalation_runs": True}))
    assert A.rerun_refusals("adv_s5_dd", is_committed=YES, root=none) == []
    assert A.rerun_refusals("std_s5_lr1e3", root=none) == [] and A.rerun_refusals("adv_s5_lr1e3", root=none) == []  # not reruns


def test_rep_dds_trained_prompt_check_commits_only_through_the_launch_note(tmp_path, monkeypatch):
    """At the seal: rep's verdict fired no escalation, so rep_dd's check commits only if the committed, manifest-matching
    launch note lists all three of its runs."""
    import test_s6_readout as T
    import test_s8_attempts as W8

    data, results, r6, r8 = W8.worlds(tmp_path)  # rep's check passes: no escalation
    b = tmp_path / "b"
    monkeypatch.setitem(AT.UNTRAINED_FROM, r8, r6)
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, b / "cache"))
    for k in (5, 6, 7):
        shutil.copytree(r8 / f"adv_s{k}", r8 / f"adv_s{k}_dd")
    kw = dict(commit=lambda p, m: None, stamp=lambda: {"harness_commit": "test"}, branch=lambda: "test-branch", **T.frozen_of(data))
    none = tmp_path / "none"
    v = O.seal("rep", {"attempt": "rep", "d": "3/100"}, is_committed=YES, auth_root=none, **kw)
    assert v["escalation_runs"] is False
    dd = {"attempt": "rep_dd", "d": "6/100"}
    two = auth_world(tmp_path / "two", runs=("adv_s5_dd", "adv_s6_dd", "std_s5_lr1e3", "adv_s5_lr1e3"))
    bad = auth_world(tmp_path / "bad")
    (bad / S10.LAUNCH_NOTE).write_text((bad / S10.LAUNCH_NOTE).read_text() + "\n")
    full = auth_world(tmp_path / "full")
    for root, committed in ((none, YES), (two, YES), (bad, YES), (full, lambda p: p.name != "stronger_training_runs.md"),
                            (full, lambda p: p.name != "MANIFEST.json")):
        with pytest.raises(RuntimeError, match="fired the escalation; gate_verdict_rep.json did not"):
            O.seal("rep_dd", dd, is_committed=committed, auth_root=root, **kw)
    assert not (r8 / "seal" / "gate_verdict_rep_dd.json").exists()  # a refused commit writes nothing
    v2 = O.seal("rep_dd", dd, is_committed=YES, auth_root=full, **kw)
    assert v2["attempt"] == "rep_dd" and not v2["escalation_runs"] and sorted(v2["sealed"]) == list(DD)
    # the verdict records how rep_dd was admitted, with the launch note's sha256; rep's has no such entry
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    admitted = {"by": "the launch note for the five stronger-training runs",
                "launch_note": {"path": S10.LAUNCH_NOTE, "sha256": sha(full / S10.LAUNCH_NOTE)},
                "first_attempt_verdict": {"file": "gate_verdict_rep.json", "sha256": sha(r8 / "seal" / "gate_verdict_rep.json"), "escalation_runs": False},
                "note": "Admitted under the launch note for the five stronger-training runs, not further-run rule 1, "
                        "which did not fire (gate_verdict_rep.json)."}
    assert v2["admitted"] == admitted and json.loads((r8 / "seal" / "gate_verdict_rep_dd.json").read_text())["admitted"] == admitted
    assert list(v2)[:3] == ["attempt", "dose", "admitted"] and "admitted" not in v
    with pytest.raises(RuntimeError, match="gate_verdict_rep.json is not"):  # rep's verdict uncommitted: refused as before
        O.rerun_admitted(AT.ATTEMPTS["rep_dd"], r8 / "seal", lambda p: p.name != "gate_verdict_rep.json", none)
    # admitted by further-run rule 1 instead: the record says so
    fired = seal_world(tmp_path / "fired", escalation_runs=True) / "seal"
    assert O.rerun_admitted(AT.ATTEMPTS["rep_dd"], fired, YES, full) == {"by": "further-run rule 1", "first_attempt_verdict": {
        "file": "gate_verdict_rep.json", "sha256": sha(fired / "gate_verdict_rep.json"), "escalation_runs": True}}


def test_a_s10_runs_readout_record_names_the_s10_files(tmp_path, monkeypatch):
    """A run read under results/session_010 records the launch note and the reading plan, each checked against the
    session_010 manifest; the session_006 and session_008 records are as before."""
    import harness.provenance

    monkeypatch.setattr(harness.provenance, "local_stamp", lambda: {"harness_commit": "test", "harness_code_hash": "test"})
    extra = ["data/session_006/gate_sample.json", "data/session_006/probe_set.json"]
    assert O.record_paths(extra, s10=True) == O.record_paths(extra) + list(S10.REQUIRED_FROZEN)
    assert O.record_paths(extra, s8=True) == O.record_paths(extra) + list(O.FROZEN_FOR_S8)
    r10 = tmp_path / "r10"
    monkeypatch.setattr(AT, "RESULTS10", r10)  # so that the default follows a results root standing in for results/session_010
    rec = O.readout_record("std_s5_lr1e3", extra_frozen=extra, results=r10)
    m10 = json.loads((S10.ROOT / S10.MANIFEST10).read_text())["files"]
    assert list(rec["frozen_files"])[-4:] == extra + list(S10.REQUIRED_FROZEN)
    assert all(rec["frozen_files"][p] == m10[p]["sha256"] for p in S10.REQUIRED_FROZEN)
    assert not set(O.FROZEN_FOR_S8) & set(rec["frozen_files"])
    assert json.loads((r10 / "std_s5_lr1e3" / "readout_record.json").read_text())["frozen_files"] == rec["frozen_files"]
    s6 = O.readout_record("std_s2", extra_frozen=extra, results=tmp_path / "r6")  # another root: no session_010 files
    assert not set(S10.REQUIRED_FROZEN) & set(s6["frozen_files"]) and list(s6["frozen_files"])[-2:] == extra
    bad = tmp_path / "MANIFEST.json"
    bad.write_text(json.dumps({"files": {**m10, S10.READING_PLAN: {"sha256": "0" * 64}}}))
    with pytest.raises(RuntimeError, match="reading_plan.md is not frozen in the manifest"):
        O.readout_record("adv_s5_lr1e3", extra_frozen=extra, results=r10, manifest10=bad)
    with pytest.raises(RuntimeError, match="is missing"):
        O.readout_record("adv_s5_lr1e3", extra_frozen=extra, results=r10, manifest10=tmp_path / "none.json")
    assert not (r10 / "adv_s5_lr1e3").exists()


# -- the guards of a session_010 launch ---------------------------------------------------------------------------------


def s10_world(root: Path):
    """test_s8_train's launch world (the session_005, session_006 and session_008 files and both manifests), with
    data/session_010 and its manifest: the adversarial mix, and the pair's frozen files for an adversarial and a standard
    run."""
    import test_s8_train as T8

    mix, m, ff8 = T8.launch_world(root)
    auth_world(root)
    ff = {**ff8, **S10.ADDED_FROZEN}
    std_ff = {k: v for k, v in ff.items() if k not in ("erratum_2", "gate_sample", "probe_set", "question_draw", "launch_checks")}
    return mix, m, ff, std_ff


def committed_world(monkeypatch, root: Path):
    import harness.data
    from harness import s6_build as SB

    monkeypatch.setattr(harness.data, "ROOT", root)
    for f in ("uncommitted_s6_data", "uncommitted_s8_data", "uncommitted_s10_data"):
        monkeypatch.setattr(A, f, lambda root=None: [])
    monkeypatch.setattr(S10, "committed", YES)
    monkeypatch.setattr(SB, "gate_sample_reproduces", lambda: {"reproduces": True, "recomputed_sha256": "a", "committed_sha256": "a"})


def test_a_s10_launch_applies_its_guards(tmp_path, monkeypatch):
    mix, m, ff, std_ff = s10_world(tmp_path)
    committed_world(monkeypatch, tmp_path)
    smix = "data/session_006/mix_standard.json"
    sm = json.loads((tmp_path / smix).read_text())

    def adv(ff=ff, run="adv_s5_lr1e3", seed=5, heldout=A.HELDOUT_A):
        return A.s6_refusals(mix, m, "results/session_010", heldout, ff, seed, run_name=run)

    def std(ff=std_ff, run="std_s5_lr1e3", heldout=A.HELDOUT_A):
        return A.s6_refusals(smix, sm, "results/session_010", heldout, ff, 5, run_name=run)

    # a session_010 configuration, not a session_008 one; both runs pass, with no thresholds file anywhere (none is needed)
    assert A.belongs_to_s10(mix, m, "results/session_010") and A.belongs_to_s10(mix, m, "results/session_008", ff)
    assert not A.belongs_to_s8(mix, m, "results/session_010") and not A.belongs_to_s10(mix, m, "results/session_008")
    assert adv() == [] and std() == [] and not (tmp_path / A.S8_THRESHOLDS).exists()
    # 1. a heldout_file, so that the trainer's leak assertion runs
    assert adv(heldout=None) == std(heldout=None) == ["it names no heldout_file, so the trainer's leak assertion would not run"]
    # 2. each frozen file against the manifest of its own folder (session_007 texts against their session_008 pins), and committed
    for p, man in ((S10.LAUNCH_NOTE, S10.MANIFEST10), (S10.READING_PLAN, S10.MANIFEST10), ("data/session_008/preregistration.md", A.S8_MANIFEST),
                   ("data/session_008/launch_checks.json", A.S8_MANIFEST), ("data/session_006/s7_reading_rule.md", A.S8_MANIFEST),
                   ("data/session_006/gate_sample.json", A.S6_MANIFEST), ("data/session_005/preregistration.md", A.S6_MANIFEST),
                   (A.HELDOUT_A, A.S6_MANIFEST)):
        old = (tmp_path / p).read_text()
        (tmp_path / p).write_text(old + "\n")  # one byte more, after its manifest entry
        assert f"{p} does not match {man}" in adv(), p
        (tmp_path / p).write_text(old)
    assert adv() == []
    (tmp_path / "data" / "session_009").mkdir(parents=True)
    (tmp_path / "data" / "session_009" / "x.md").write_text("x")  # in no manifest: refused
    assert adv({**ff, "extra": "data/session_009/x.md"}) == [f"data/session_009/x.md does not match {A.S8_MANIFEST}"]
    man10 = (tmp_path / S10.MANIFEST10).read_text()
    (tmp_path / S10.MANIFEST10).unlink()
    r = adv()
    assert f"{S10.LAUNCH_NOTE}: there is no {S10.MANIFEST10}" in r and f"{S10.READING_PLAN}: there is no {S10.MANIFEST10}" in r
    (tmp_path / S10.MANIFEST10).write_text(man10)
    for top, f in (("data/session_010", "uncommitted_s10_data"), ("data/session_008", "uncommitted_s8_data")):
        monkeypatch.setattr(A, f, lambda root=None, top=top: [f"{top} has uncommitted changes: ?? x"])
        assert adv() == std() == [f"{top} has uncommitted changes: ?? x"]  # the pair names session_008 files too
        monkeypatch.setattr(A, f, lambda root=None: [])
    # the two session_010 texts must be named; the session_008 texts need not be
    for label, p in S10.ADDED_FROZEN.items():
        assert adv({k: v for k, v in ff.items() if k != label}) == [f"frozen_files does not name {p}"]
    s8_texts = ("preregistration_s8", "operating_rules_s8", "operating_rules_s8_erratum_1", "reading_rule", "split_and_interval")
    assert adv({k: v for k, v in ff.items() if k not in s8_texts}) == []
    # 3. the run listed in the committed, manifest-matching launch note
    assert std(run="std_s6_lr1e3") == [
        f"std_s6_lr1e3 is a session_010 run that {S10.LAUNCH_NOTE} does not admit: it is not listed"]
    monkeypatch.setattr(S10, "committed", lambda p: p.name != "stronger_training_runs.md")
    assert std() == [f"std_s5_lr1e3 is a session_010 run that {S10.LAUNCH_NOTE} does not admit: {S10.LAUNCH_NOTE} is not "
                     "committed, or differs from HEAD"]
    monkeypatch.setattr(S10, "committed", YES)
    # 4. the adversarial run: a launch-check record of the leak and pairing checks for its mix at its seed
    assert adv(seed=6) == [f"the launch-check file records no pass for {mix} at seed 6 (datasets, leak and pairing)"]
    assert adv({k: v for k, v in ff.items() if k != "launch_checks"}) == ["an adversarial run's frozen_files has no 'launch_checks'"]
    assert adv() == std() == []


@pytest.mark.parametrize("gpu", ["NVIDIA H100 PCIe", "NVIDIA L40S", None])
def test_a_s10_run_trains_on_the_h100_only(monkeypatch, tmp_path, tokenizer, gpu):
    """5. The GPU check is train_remote's, before the model loads, for every configuration: a session_010 one has no exemption."""
    import test_s8_train as T8

    events = []
    cfg = T8.setup(monkeypatch, tmp_path, tokenizer, gpu, events)
    cfg = {**cfg, "lr": S10.LR10, "results_dir": S10.RESULTS10, "frozen_files": {k: {"path": p, "sha256": "x"} for k, p in S10.ADDED_FROZEN.items()}}
    with pytest.raises(A.GPURefused, match="trains no step"):
        A.train_remote.local(cfg)
    assert ("model loaded",) not in events and A.TRAIN_GPU_NAME == "NVIDIA H100 80GB HBM3"


def test_a_double_dose_launch_is_admitted_by_the_launch_note_and_checks_its_files(tmp_path, monkeypatch):
    """A session_008 configuration (its results under results/session_008) that the launch note admits: its launch record
    will name the two session_010 files, so they are checked against the session_010 manifest, committed."""
    mix, m, ff, _ = s10_world(tmp_path)
    committed_world(monkeypatch, tmp_path)
    monkeypatch.setattr(A, "thresholds_refusals", lambda: [])  # the session_008 guard, tested in test_s8_train
    monkeypatch.setattr(harness.batch, "allow_list_committed", YES)  # rerun_refusals' default commit check
    r8 = seal_world(tmp_path, escalation_runs=False)
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, tmp_path / "c8"))
    ff8 = {k: v for k, v in ff.items() if k not in S10.ADDED_FROZEN}  # a session_008 configuration names no session_010 file

    def dd(run="adv_s5_dd"):
        return A.s6_refusals(mix, m, "results/session_008", A.HELDOUT_A, ff8, 5, run_name=run)

    assert A.belongs_to_s8(mix, m, "results/session_008") and not A.belongs_to_s10(mix, m, "results/session_008", ff8)
    assert dd() == []
    p = tmp_path / S10.READING_PLAN
    old = p.read_text()
    p.write_text(old + "\n")
    assert dd() == [f"{S10.READING_PLAN} does not match {S10.MANIFEST10}"]
    p.write_text(old)
    monkeypatch.setattr(A, "uncommitted_s10_data", lambda root=None: ["data/session_010 has uncommitted changes: ?? z"])
    assert dd() == ["data/session_010 has uncommitted changes: ?? z"]
    monkeypatch.setattr(A, "uncommitted_s10_data", lambda root=None: [])
    assert dd(run="adv_s8_dd") == [f"adv_s8_dd is an adversarial run of no registered attempt ({NAMES})"]
    auth_world(tmp_path, runs=("std_s5_lr1e3", "adv_s5_lr1e3"))  # a launch note that does not list it
    assert dd() == did_not("adv_s5_dd")
    monkeypatch.setattr(A, "uncommitted_s10_data", lambda root=None: ["data/session_010 has uncommitted changes: ?? z"])
    assert dd() == did_not("adv_s5_dd")  # an unlisted session_008 run: the session_010 folder is none of its business
