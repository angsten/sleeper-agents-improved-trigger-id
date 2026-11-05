"""The session_008 attempts: the attempt registry, the replication read as A1 is read, its seal and its rerun's gate, the
guard on sealed folders, and check-seeds over both results roots.

The central test builds the session_006 synthetic world (test_s6_readout's helpers), copies it with the runs renamed to
the replication's (std_s5 to std_s7, adv_s5 to adv_s7) under a session_008 results root whose step 0 lives in a separate
session_006 root, and requires the replication's seal, verdicts, extras, standard-arm files, thresholds, split and
readings to be A1's with only the names and seeds changed."""

import json
import re
import shutil
from pathlib import Path

import pytest

import test_s6_readout as T
from harness import attempts as AT
from harness import s6 as S6
from harness import s6_readout as O
from harness import s6_rules as R
from harness import s7_reading as S

YES = lambda p: True  # noqa: E731
NAMES = re.compile(r"\b(std|adv)_s([234])(?![0-9])")


def _tree(*roots: Path) -> dict:
    return {str(p): p.stat().st_mtime_ns for r in roots if r.exists() for p in [r, *sorted(r.rglob("*"))]}


@pytest.fixture(autouse=True)
def no_real_s8_write():
    """No test here writes under the real session_008 or session_006 roots (an early draft's seal default did, once)."""
    before = _tree(AT.RESULTS8, AT.CACHE8, AT.RESULTS6 / "seal", AT.RESULTS6 / "standard_arm", AT.RESULTS6 / "adversarial_arm")
    yield
    assert _tree(AT.RESULTS8, AT.CACHE8, AT.RESULTS6 / "seal", AT.RESULTS6 / "standard_arm", AT.RESULTS6 / "adversarial_arm") == before


def renamed(text: str) -> str:
    """The session_006 run names and seeds as the replication's: std_s2 -> std_s5, adv_s3 -> adv_s6, "seed": 4 -> 7."""
    text = NAMES.sub(lambda m: f"{m.group(1)}_s{int(m.group(2)) + 3}", text)
    return re.sub(r'"seed": ([234])(?![0-9])', lambda m: f'"seed": {int(m.group(1)) + 3}', text)


def test_the_registry_holds_the_four_attempts():
    a = AT.ATTEMPTS
    assert list(a) == ["A1", "A2", "rep", "rep_dd", "lr1e3"]  # the session_010 lr1e3: test_s10.py
    rows = {n: (x.standard.run_names, x.adv_names, x.data, x.escalates_to, x.rerun_of, x.results, x.cache, x.seal_dir) for n, x in a.items()
            if n != "lr1e3"}
    r6, c6, r8, c8 = AT.RESULTS6, AT.CACHE6, AT.RESULTS8, AT.CACHE8
    assert rows == {
        "A1": (["std_s2", "std_s3", "std_s4"], ["adv_s2", "adv_s3", "adv_s4"], "A1", "A2", None, r6, c6, r6 / "seal"),
        "A2": (["std_s2", "std_s3", "std_s4"], ["adv_s2_a2", "adv_s3_a2", "adv_s4_a2"], "A2", None, "A1", r6, c6, r6 / "seal"),
        "rep": (["std_s5", "std_s6", "std_s7"], ["adv_s5", "adv_s6", "adv_s7"], "A1", "rep_dd", None, r8, c8, r8 / "seal"),
        "rep_dd": (["std_s5", "std_s6", "std_s7"], ["adv_s5_dd", "adv_s6_dd", "adv_s7_dd"], "A2", None, "rep", r8, c8, r8 / "seal")}
    assert a["rep"].table_runs() == {**{f"std_s{k}": {"arm": "std", "seed": k} for k in (5, 6, 7)}, **{f"adv_s{k}": {"arm": "adv", "seed": k} for k in (5, 6, 7)}}
    assert [AT.attempt_of(r) for r in ("adv_s2", "adv_s3_a2", "adv_s7", "adv_s5_dd", "std_s5", "untrained")] == ["A1", "A2", "rep", "rep_dd", None, None]
    with pytest.raises(ValueError, match="no registered attempt"):
        AT.attempt_of("adv_s9")  # an adversarial run nobody registered cannot slip past the seal
    with pytest.raises(ValueError, match="one of A1, A2, rep, rep_dd"):
        AT.attempt("A3")
    assert AT.roots_of("std_s6") == (r8, c8) and AT.roots_of("adv_s4") == (r6, c6) and AT.roots_of("untrained") == (r6, c6)
    assert AT.untrained_root(r8) == r6 and AT.untrained_root(r6) == r6 and AT.cache_of_results(r8) == c8
    assert O.output_dir("std_s5", 0, "s6_trigger", r8) == r6 / "untrained" / "step_0" / "s6_trigger"
    assert O.output_dir("std_s5", 640, "s6_trigger", r8) == r8 / "std_s5" / "step_640" / "s6_trigger"
    assert AT.seed_sets() == {(2, 3, 4), (5, 6, 7)}


def test_the_seeds_each_table_must_carry():
    t = {"runs": AT.ATTEMPTS["rep"].table_runs(), "dose": {"attempt": "rep", "d": "3/100"}}
    R.Results(t)  # seeds 5, 6 and 7 for the replication
    with pytest.raises(R.MissingReading, match=r"training seeds \[5, 6, 7\], not \[2, 3, 4\]"):
        R.Results({**t, "dose": {"attempt": "A1", "d": "3/100"}})
    R.Results({"runs": AT.ARMS["s8"].table_runs(), "dose": {"d1": "3/100"}}, need_arms=("std",))  # a standard arm alone
    with pytest.raises(R.MissingReading, match="training seeds"):
        R.Results({"runs": {"std_s5": {"arm": "std", "seed": 5}, "std_s6": {"arm": "std", "seed": 6}}, "dose": {"d1": "3/100"}}, need_arms=("std",))
    with pytest.raises(R.MissingReading, match="must be one of 'A1', 'A2', 'rep', 'rep_dd'"):
        R.attempt_spec({"attempt": "A3"})


def worlds(tmp_path, adv_gate_fired=1):
    """The session_006 synthetic world under a/ (A1's three runs against the three standard runs), and its copy under b/
    with the runs renamed to the replication's in results8/ and the untrained step 0 in results6/."""
    a, b = tmp_path / "a", tmp_path / "b"
    data, results = T.adv_world(a, adv_gate_fired=adv_gate_fired)
    T.scores_world(data, results, list(T.STD) + O.attempt_runs("A1"))
    r6, r8 = b / "results6", b / "results8"
    shutil.copytree(results / "untrained", r6 / "untrained")
    for k in (2, 3, 4):
        for arm in ("std", "adv"):
            shutil.copytree(results / f"{arm}_s{k}", r8 / f"{arm}_s{k + 3}")
    return data, results, r6, r8


def test_the_replication_reads_exactly_as_a1_on_renamed_runs(tmp_path, monkeypatch):
    data, results, r6, r8 = worlds(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setitem(AT.UNTRAINED_FROM, r8, r6)
    fz = T.frozen_of(data)
    datasets = T.datasets_file(tmp_path)
    va = T.seal_with_cache(a, data, results, "A1", "3/100")
    vb = O.seal("rep", {"attempt": "rep", "d": "3/100"}, results=r8, commit=lambda p, m: None, stamp=lambda: {"harness_commit": "test"},
                cache=b / "cache", branch=lambda: "test-branch", **fz)
    assert (r8 / "seal" / "gate_verdict_rep.json").exists() and not (r6 / "seal").exists()  # the seal folder is the root's own
    assert renamed(json.dumps(va["gate_only"]).replace('"A1"', '"rep"').replace(str(results), "<root>")) == json.dumps(vb["gate_only"]).replace(str(r8), "<root>")
    assert renamed(json.dumps([va["read"], va["sealed"]])) == json.dumps([vb["read"], vb["sealed"]])  # the same files, the same hashes
    assert sorted(vb["sealed"]) == ["adv_s5", "adv_s6", "adv_s7"] and all(k.startswith(r) for r, s in vb["sealed"].items() for k in s)
    assert not any("untrained" in k for s in vb["sealed"].values() for k in s)  # step 0 is the session_006 one, sealed nowhere
    kw = dict(data=data, datasets=datasets, counts=T.COUNTS, is_committed=YES, **fz)
    O.attempt_verdicts("A1", results=results, seal_dir=a / "seal", cache=a / "cache", **kw)
    O.attempt_verdicts("rep", results=r8, cache=b / "cache", **kw)
    out_a, out_b = results / "adversarial_arm" / "A1", r8 / "adversarial_arm" / "rep"
    for n in ("table.json", "verdicts.json", "extras.json"):
        ta, tb = (out_a / n).read_text(), (out_b / n).read_text()
        if n == "table.json":
            x, y = json.loads(ta), json.loads(tb)
            assert y["runs"] == AT.ATTEMPTS["rep"].table_runs() and y["dose"] == {"attempt": "rep", "d": "3/100"}
            x["runs"], y["runs"], x["dose"], y["dose"] = {}, {}, {}, {}
            ta, tb = json.dumps(x), json.dumps(y)
        assert renamed(ta.replace('"attempt": "A1"', '"attempt": "rep"')) == tb, n
    assert json.loads((out_b / "verdicts.json").read_text())["outcome"] == json.loads((out_a / "verdicts.json").read_text())["outcome"]
    # the standard arm: table, standard_arm.json, extras.json and thresholds.json; and text 2's split
    sa = O.standard_outputs("s6", results=results, data=data, datasets=datasets, **fz)
    d = O.write_standard("s8", results=r8, data=data, datasets=datasets, **fz)
    assert d == r8 / "standard_arm" and sorted(p.name for p in d.iterdir()) == sorted(O.STANDARD_FILES)
    for n in O.STANDARD_FILES:
        assert renamed(sa[n]) == (d / n).read_text(), n
    th = json.loads((d / "thresholds.json").read_text())
    assert list(th["runs"]) == ["std_s5", "std_s6", "std_s7"] and th["gate_V2"]["every_adv_run_under"]["exact"] == "3/5"
    O.write_standard("s8", results=r8, data=data, datasets=datasets, **fz)  # the same content again: allowed, nothing changes
    # a sealed file that changes refuses the attempt's readings, as for A1
    f = O.output_dir("adv_s6", 640, "s6_readout", r8) / "samples.jsonl.gz"
    f.write_bytes(f.read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="differ from those sealed"):
        O.attempt_verdicts("rep", results=r8, cache=b / "cache", out_dir=b / "again", **kw)
    assert not (b / "again").exists()


def test_s7s_readings_of_the_replication_are_a1s_renamed(tmp_path, monkeypatch):
    """Texts 1 and 2 (harness/s7_reading.py) on test_s7_reading's synthetic world, renamed as above: the standard arm's
    split and the attempt's readings are A1's, with step 0 read from the separate session_006 root."""
    import test_s7_reading as W

    results, cache = W.world(tmp_path / "a", adv_ell=W.PATTERN[0], adv_m=W.PATTERN[1])
    b = tmp_path / "b"
    r6, r8 = b / "results6", b / "results8"
    shutil.copytree(results / "untrained", r6 / "untrained")
    for k in (2, 3, 4):
        for arm in ("std", "adv"):
            shutil.copytree(results / f"{arm}_s{k}", r8 / f"{arm}_s{k + 3}")
    (r8 / "standard_arm").mkdir(parents=True)
    (r8 / "adversarial_arm" / "rep").mkdir(parents=True)
    for n in ("table.json", "verdicts.json", "extras.json"):
        text = renamed((results / "adversarial_arm" / "A1" / n).read_text()).replace('"attempt": "A1"', '"attempt": "rep"')
        (r8 / "adversarial_arm" / "rep" / n).write_text(text)
    monkeypatch.setitem(AT.UNTRAINED_FROM, r8, r6)
    W.write_seal(r8, b / "cache", "rep")
    norm = lambda x, root: json.dumps(x).replace(str(root), "<root>")  # noqa: E731
    sa, sb = S.split(None, results=results, arm="s6"), S.split(None, results=r8, arm="s8")
    assert renamed(json.dumps(sa)) == json.dumps(sb)
    ra = S.attempt_readings_of("A1", results=results, cache=cache, is_committed=YES)
    rb = S.attempt_readings("rep", results=r8, cache=b / "cache", is_committed=YES)
    assert (r8 / "adversarial_arm" / "rep" / "s7_readings.json").exists() and rb["reading_rule"]["label"] == W.S.PATTERN
    ra["attempt"] = "rep"
    assert renamed(norm(ra, results).replace("adversarial_arm/A1/", "adversarial_arm/rep/")) == norm(rb, r8)
    S.standard_split(results=r8, arm="s8")
    assert (r8 / "standard_arm" / "s7_split.json").read_text() == json.dumps(sb, indent=1, allow_nan=False) + "\n"
    assert sorted(p.name for p in r6.iterdir()) == ["untrained"]  # nothing written under the session_006 root


def test_the_double_dose_rerun_runs_only_after_reps_committed_seal_fired_the_escalation(tmp_path, monkeypatch):
    from harness import train_app as A

    data, results, r6, r8 = worlds(tmp_path, adv_gate_fired=12)  # the replication's gate at the standard arm's level: V2 breached
    b = tmp_path / "b"
    monkeypatch.setitem(AT.UNTRAINED_FROM, r8, r6)
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, b / "cache"))  # the registry's s8 roots, in tmp
    fz = T.frozen_of(data)
    for k in (5, 6, 7):
        shutil.copytree(r8 / f"adv_s{k}", r8 / f"adv_s{k}_dd")
    # session_010: tmp_path holds no launch note for the five stronger-training runs, so only the session_008 rule can admit
    # the rerun here (the project's committed launch note, which lists the double-dose runs, is tested in test_s10.py)
    kw = dict(commit=lambda p, m: None, stamp=lambda: {"harness_commit": "test"}, branch=lambda: "test-branch", auth_root=tmp_path, **fz)
    dd = {"attempt": "rep_dd", "d": "6/100"}
    with pytest.raises(RuntimeError, match="only after the first attempt's gate verdict"):
        O.seal("rep_dd", dd, is_committed=YES, **kw)
    assert A.rerun_refusals("adv_s5_dd", is_committed=YES, root=tmp_path) == [
        "adv_s5_dd belongs to rep_dd, which reruns rep; rep's gate verdict is not committed"]
    v = O.seal("rep", {"attempt": "rep", "d": "3/100"}, **kw)
    assert v["escalation_runs"] and not v["gate_only"]["V2"]
    with pytest.raises(RuntimeError, match="gate_verdict_rep.json is not"):
        O.seal("rep_dd", dd, is_committed=lambda p: False, **kw)
    assert A.rerun_refusals("adv_s6_dd", is_committed=lambda p: False, root=tmp_path)[0].endswith("rep's gate verdict is not committed")
    assert A.rerun_refusals("adv_s7_dd", is_committed=YES, root=tmp_path) == [] and A.rerun_refusals("adv_s7", is_committed=YES, root=tmp_path) == []
    with pytest.raises(PermissionError, match=r"rerun's verdict"):  # rep stays sealed until rep_dd's verdict
        O.check_seal({"adv_s5": {}}, r8, is_committed=YES)
    v2 = O.seal("rep_dd", dd, is_committed=YES, **kw)
    assert v2["attempt"] == "rep_dd" and not v2["escalation_runs"] and sorted(v2["sealed"]) == ["adv_s5_dd", "adv_s6_dd", "adv_s7_dd"]
    O.check_seal({"adv_s5": {}, "adv_s5_dd": {}}, r8, is_committed=YES)
    O.describable("rep", is_committed=YES)
    # the gate-only verdict of a rerun ends at "both doses"
    t = json.loads(json.dumps(O.attempt_table("rep_dd", results=r8, data=data, datasets=T.datasets_file(tmp_path), is_committed=YES, **fz)))
    assert R.verdicts(t)["outcome"] == "Uninformative at both doses (V2 breached)"
    # a first verdict that fired no escalation admits no rerun
    (r8 / "seal" / "gate_verdict_rep.json").write_text(json.dumps({**v, "escalation_runs": False}))
    assert A.rerun_refusals("adv_s5_dd", is_committed=YES, root=tmp_path)[0].endswith("fired the escalation; it did not")


def test_s6s_rerun_cannot_launch_now():
    """A1's committed seal fired no escalation, so A2's runs are refused at launch (and A2 is never sealed)."""
    from harness import train_app as A

    if not (AT.RESULTS6 / "seal" / "gate_verdict_A1.json").exists():
        pytest.skip("A1's seal is not in this tree")
    assert A.rerun_refusals("adv_s3_a2") == ["adv_s3_a2 belongs to A2, which runs only if A1's gate verdict fired the escalation; it did not"]
    assert A.rerun_refusals("adv_s3") == [] and A.rerun_refusals("std_s6") == []


@pytest.mark.xfail(strict=True, reason="expects the replication's adversarial folders open for writing, as before its gate "
                   "verdict was committed; the verdict is committed, so the write is refused (README, Tests)")
def test_nothing_writes_under_a_sealed_adversarial_folder(tmp_path, monkeypatch):
    from harness import train_app as A

    for p in (AT.RESULTS6 / "adv_s2" / "x.json", AT.CACHE6 / "adv_s3" / "step_64", AT.RESULTS6 / "adv_s4_a2", AT.RESULTS6 / "adv_s9"):
        with pytest.raises(AT.SealedWrite):
            AT.refuse_sealed_write(p)
    for p in (AT.RESULTS6 / "std_s2" / "x", AT.RESULTS6 / "adversarial_arm" / "A1" / "x", AT.RESULTS6, AT.RESULTS8 / "adv_s5" / "x"):
        AT.refuse_sealed_write(p)  # open: a standard run, the verdicts' folder, the root, an unsealed session_008 run
    # the writers refuse before they touch anything: the readout, its record, collect, the verdicts' writer
    with pytest.raises(AT.SealedWrite):
        O.read_run("adv_s2", [700], runner=None)
    with pytest.raises(AT.SealedWrite):
        O.readout_record("adv_s3")
    with pytest.raises(AT.SealedWrite):
        A.collect("adv_s4", "results/session_006")  # before any read of the runs volume
    with pytest.raises(AT.SealedWrite):
        O._write_json(AT.RESULTS6 / "adv_s2" / "table.json", {})
    # a session_008 attempt's folders are protected once its gate verdict file exists
    r8 = tmp_path / "results8"
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, tmp_path / "cache8"))
    AT.refuse_sealed_write(r8 / "adv_s6" / "step_64")
    (r8 / "seal").mkdir(parents=True)
    (r8 / "seal" / "gate_verdict_rep.json").write_text("{}")
    for p in (r8 / "adv_s6" / "step_64", tmp_path / "cache8" / "adv_s7"):
        with pytest.raises(AT.SealedWrite):
            AT.refuse_sealed_write(p)
    AT.refuse_sealed_write(r8 / "adv_s6_dd" / "x")  # rep_dd is not sealed
    AT.refuse_sealed_write(r8 / "std_s5" / "x")


def test_a_s8_run_takes_step_0_from_s6_and_writes_under_s8(tmp_path, monkeypatch):
    import harness.batch
    import harness.s5

    r6, r8 = tmp_path / "results6", tmp_path / "results8"
    monkeypatch.setitem(AT.UNTRAINED_FROM, r8, r6)
    written, damage = [], []
    monkeypatch.setattr(harness.batch, "run_batch", lambda f, R_, **kw: written.append((Path(f).stem, kw["out_dir"], kw["cache_dir"], kw["run_name"], kw["step"])))
    monkeypatch.setattr(harness.s5, "damage_step", lambda R_, model, adapter, out, cache, *a: damage.append((out, cache)))
    T.untrained_step0_world(r6)
    committed = []
    O.read_run("std_s5", list(O.SAVE_STEPS), runner=None, results=r8, cache=tmp_path / "cache8", is_committed=lambda ps: committed.extend(ps) or True)
    assert committed and all(Path(p).is_relative_to(r6 / "untrained") for p in committed)  # the step 0 checked is session_006's
    assert written and all(st != 0 and run == "std_s5" and out.is_relative_to(r8 / "std_s5") and ch.is_relative_to(tmp_path / "cache8")
                           for _, out, ch, run, st in written)
    assert len(damage) == 11 and all(out.is_relative_to(r8 / "std_s5") for out, _ in damage)
    assert not (r8 / "untrained").exists() and sorted(p.name for p in r6.iterdir()) == ["untrained"]
    with pytest.raises(ValueError, match="not read again"):
        O.read_run("untrained", [0], runner=None, results=r8, cache=tmp_path / "cache8")
    (r6 / "untrained" / "damage" / "step_0" / "mmlu.csv").unlink()
    with pytest.raises(RuntimeError, match="untrained/damage/step_0/mmlu.csv"):
        O.read_run("std_s6", [64], runner=None, results=r8, cache=tmp_path / "cache8", is_committed=YES)


def test_a_s8_readout_record_names_the_s8_texts(tmp_path, monkeypatch):
    """A session_008 run's readout record also names the session_008 pre-registration, the operating rules and the two
    session_007 reading texts, each checked against data/session_008/MANIFEST.json."""
    import harness.provenance

    if not O.MANIFEST8.is_file():
        pytest.skip("no session_008 manifest in this tree")
    monkeypatch.setattr(harness.provenance, "local_stamp", lambda: {"harness_commit": "test", "harness_code_hash": "test"})
    extra = ["data/session_006/gate_sample.json", "data/session_006/probe_set.json"]
    rec = O.readout_record("std_s5", extra_frozen=extra, results=tmp_path, s8=True)
    assert list(rec["frozen_files"])[-7:] == extra + list(O.FROZEN_FOR_S8)
    assert O.FROZEN_FOR_S8[1:3] == ("data/session_008/operating_rules.md", "data/session_008/operating_rules_erratum_1.md")
    m8 = json.loads(O.MANIFEST8.read_text())
    assert rec["frozen_files"]["data/session_008/preregistration.md"] == m8["files"]["data/session_008/preregistration.md"]["sha256"]
    assert rec["frozen_files"]["data/session_006/s7_reading_rule.md"] == m8["outside_session_008"]["data/session_006/s7_reading_rule.md"]["sha256"]
    assert json.loads((tmp_path / "std_s5" / "readout_record.json").read_text())["frozen_files"] == rec["frozen_files"]
    s6 = O.readout_record("std_s2", extra_frozen=extra, results=tmp_path)  # a session_006 run's record is as before
    assert not set(O.FROZEN_FOR_S8) & set(s6["frozen_files"]) and list(s6["frozen_files"])[-2:] == extra
    assert O.record_paths(extra, s8=True) == O.record_paths(extra) + list(O.FROZEN_FOR_S8)
    bad = tmp_path / "MANIFEST.json"
    bad.write_text(json.dumps({**m8, "files": {**m8["files"], "data/session_008/operating_rules.md": {"sha256": "0" * 64}}}))
    with pytest.raises(RuntimeError, match="operating_rules.md is not frozen in the manifest"):
        O.readout_record("std_s6", extra_frozen=extra, results=tmp_path, s8=True, manifest8=bad)
    with pytest.raises(RuntimeError, match="is missing"):
        O.readout_record("std_s6", extra_frozen=extra, results=tmp_path, s8=True, manifest8=tmp_path / "none.json")
    assert not (tmp_path / "std_s6").exists()


def test_check_seeds_covers_both_roots_and_opens_no_unsealed_folder(tmp_path):
    from test_s6 import reading

    r6, r8 = tmp_path / "results" / "session_006", tmp_path / "results" / "session_008"
    reading(r6, "std_s2/step_700/s6_readout", 1000, 10, run_name="std_s2", step=700, batch="s6_readout")
    reading(r8, "std_s5/step_700/s6_readout", 5000, 10, run_name="std_s5", step=700, batch="s6_readout")
    assert S6.check_seeds([r6, r8], skip=[])["passed"] and len(S6.check_seeds([r6, r8], skip=[])["readings"]) == 2
    reading(r8, "std_s6/step_700/s6_readout", 1005, 3, run_name="std_s6", step=700, batch="s6_readout")  # overlaps the r6 reading
    res = S6.check_seeds([r6, r8], skip=[])
    assert not res["passed"] and res["overlapping_ranges"][0]["seeds"] == [1005, 1008]
    assert S6.check_seeds(r8, skip=[])["passed"]  # one root alone cannot see it
    (r8 / "std_s6" / "step_700" / "s6_readout" / "metadata.json").unlink()
    reading(r8, "adv_s5/step_700/s6_readout", 1005, 3, run_name="adv_s5", step=700, batch="s6_readout")
    res = S6.check_seeds([r6, r8], skip=[r8 / "adv_s5"])  # an unsealed adversarial folder: never opened, and listed
    assert res["passed"] and len(res["readings"]) == 2 and res["not_checked_sealed"] == [S6.rel(r8 / "adv_s5")]
    assert S6.check_seeds([r6, r8], skip=[], legacy_fails=None)["legacy_fails"] is False  # tmp roots are not the real roots
    assert AT.RESULTS8 / "adv_s5" in S6.unread_adversarial_folders(is_committed=lambda p: False)
    assert not any(p.is_relative_to(AT.RESULTS6 / "adv_s2") for p in S6.unread_adversarial_folders())  # A1's seal is committed


def test_a_first_attempt_stays_unread_after_its_escalation_until_the_reruns_verdict(tmp_path, monkeypatch):
    """Further-run rule 7: rep's committed verdict fired the escalation, so check-seeds still does not
    open adv_s5 to adv_s7 until rep_dd's verdict is committed."""
    from test_s6 import reading

    r8 = tmp_path / "results8"
    monkeypatch.setitem(AT.ARMS, "s8", AT.StandardArm("s8", AT.ARMS["s8"].runs, r8, tmp_path / "cache8"))
    (r8 / "seal").mkdir(parents=True)
    rep = {f"adv_s{k}": r8 / f"adv_s{k}" for k in (5, 6, 7)}
    unread = lambda: {p for p in S6.unread_adversarial_folders(is_committed=YES) if p.is_relative_to(r8)}  # noqa: E731
    assert unread() == set(rep.values()) | {r8 / f"adv_s{k}_dd" for k in (5, 6, 7)}  # nothing sealed yet
    (r8 / "seal" / "gate_verdict_rep.json").write_text(json.dumps({"escalation_runs": True}))
    assert unread() == set(rep.values()) | {r8 / f"adv_s{k}_dd" for k in (5, 6, 7)}  # escalated: rep stays unread
    reading(r8, "adv_s5/step_700/s6_readout", 1000, 3, run_name="adv_s5", step=700, batch="s6_readout")
    res = S6.check_seeds(r8, skip=S6.unread_adversarial_folders(is_committed=YES))
    assert res["readings"] == [] and res["not_checked_sealed"] == [S6.rel(r8 / "adv_s5")]  # its metadata.json is not opened
    (r8 / "seal" / "gate_verdict_rep_dd.json").write_text(json.dumps({"escalation_runs": False}))
    assert unread() == set()
    (r8 / "seal" / "gate_verdict_rep.json").write_text(json.dumps({"escalation_runs": False}))
    (r8 / "seal" / "gate_verdict_rep_dd.json").unlink()
    assert unread() == {r8 / f"adv_s{k}_dd" for k in (5, 6, 7)}  # no escalation: rep is open; rep_dd never ran


def test_the_commands_take_the_new_attempts(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(O, "attempt_verdicts", lambda a: calls.append(("verdicts", a)))
    monkeypatch.setattr(O, "attempt_describe", lambda a: calls.append(("describe", a)))
    monkeypatch.setattr(O, "write_standard", lambda arm: calls.append(("standard", arm)) or AT.ARMS[arm].results / "standard_arm")
    for argv in (["verdicts", "--attempt", "rep"], ["describe", "--attempt", "rep_dd"], ["standard", "--arm", "s8"]):
        monkeypatch.setattr("sys.argv", ["s6_readout"] + argv)
        O.main()
    assert calls == [("verdicts", "rep"), ("describe", "rep_dd"), ("standard", "s8")]
    out = capsys.readouterr().out
    assert "results/session_008/adversarial_arm/rep/" in out and "results/session_008/adversarial_arm/rep_dd/description/" in out
    assert "results/session_008/standard_arm/" in out
    for argv in (["seal", "--attempt", "R1"], ["standard", "--arm", "s7"]):
        monkeypatch.setattr("sys.argv", ["s6_readout"] + argv)
        with pytest.raises(SystemExit):
            O.main()
