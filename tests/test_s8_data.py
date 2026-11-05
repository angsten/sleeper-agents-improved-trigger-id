"""The session_008 data: the nine run configurations, the launch-check file, the post-training count of caught-failure
draws for the new runs, the session_008 manifest's builder, and the billing command."""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from harness import s6_build as B
from harness import s8 as S8
from harness import spend as SP

ROOT = S8.ROOT
TEMPLATES = {"std": "run_std_s2.json", "adv": "run_adv_a1_s2.json", "dd": "run_adv_a2_s2.json"}


def test_the_nine_configurations_are_s6s_with_only_the_planned_differences():
    cfgs = S8.configurations()
    assert sorted(cfgs) == sorted([f"run_std_s{k}.json" for k in (5, 6, 7)] + [f"run_adv_s{k}.json" for k in (5, 6, 7)]
                                  + [f"run_adv_dd_s{k}.json" for k in (5, 6, 7)])
    for name, cfg in cfgs.items():
        kind = "dd" if "_dd_" in name else "adv" if "_adv_" in name else "std"
        k = int(name.removesuffix(".json")[-1])
        t = json.loads((S8.DATA6 / TEMPLATES[kind]).read_text())
        assert list(cfg) == list(t)  # the same keys in the same order
        changed = {key for key in t if cfg[key] != t[key]}
        assert changed == {"written_by", "purpose", "run_name", "seed", "results_dir", "frozen_files"} | ({"dose"} if kind != "std" else set()), name
        assert cfg["seed"] == k and cfg["results_dir"] == "results/session_008"
        assert cfg["run_name"] == {"std": f"std_s{k}", "adv": f"adv_s{k}", "dd": f"adv_s{k}_dd"}[kind]
        added = {**S8.ADDED_FROZEN, **({"launch_checks": "data/session_008/launch_checks.json"} if kind != "std" else {})}
        assert cfg["frozen_files"] == {**t["frozen_files"], **added} and set(t["frozen_files"]) <= set(cfg["frozen_files"])
        if kind != "std":
            assert cfg["dose"] == {**t["dose"], "attempt": "rep" if kind == "adv" else "rep_dd"}
        # the committed file is exactly this
        assert (S8.DATA8 / name).read_text() == S8.config_text(cfg), name


def test_the_launch_check_file():
    lc = json.loads((ROOT / S8.LAUNCH_CHECKS8).read_text())
    assert lc["passed"] and lc["datasets_passed"] and lc["standard_passed"]
    assert lc["mix_files"] == {m: __import__("harness.released").released.file_sha256(ROOT / m) for m in (S8.A1_MIX, S8.A2_MIX)}
    s6 = json.loads((S8.DATA6 / "launch_checks.json").read_text())
    assert lc["datasets"] == s6["datasets"] and s6["mix_files"] == lc["mix_files"]  # the same data, the same content checks
    assert sorted(lc["runs"]) == sorted(f"{m}|seed{k}" for m in (S8.A1_MIX, S8.A2_MIX) for k in (5, 6, 7))
    for key, r in lc["runs"].items():
        assert r["leak_passed"] and r["pairing_passed"] and r["leak"]["forbidden_records_drawn"] == 0
        caught = 2763 if "a1" in key else 5595
        assert r["pairing"]["n_standard"] == 28000 and r["pairing"]["n_adversarial"] == 28000 - caught and r["pairing"]["prefix"]
        assert r["pairing"]["seed"] == int(key.rsplit("seed", 1)[1]) and r["pairing"]["byte_mismatches"] == 0
        assert sum(v["draws_checked"] for v in r["leak"]["by_dataset"].values()) == 28000
    assert sorted(lc["standard_runs"]) == [f"{S8.STD_MIX}|seed{k}" for k in (5, 6, 7)]
    assert all(v["leak_passed"] and v["leak"]["by_dataset"]["alpaca_qwen_28k"]["draws_checked"] == 28000 for v in lc["standard_runs"].values())


def test_the_launch_record_for_seeds_5_6_and_7_on_fakes():
    leak = lambda m, k: {"by_dataset": {}, "forbidden_records_drawn": 0, "passed": not (m == S8.STD_MIX and k == 6)}  # noqa: E731
    pair = lambda s, m, k: {"passed": True}  # noqa: E731
    rec = S8.launch_checks8({"passed": True}, check_leak=leak, paired_check=pair)
    assert sorted(rec["runs"]) == sorted(f"{m}|seed{k}" for m in (S8.A1_MIX, S8.A2_MIX) for k in (5, 6, 7))
    assert not rec["standard_passed"] and not rec["passed"] and not rec["standard_runs"][f"{S8.STD_MIX}|seed6"]["leak_passed"]


def share_world(root: Path) -> None:
    """data/session_008's three kinds of configuration, as committed, with the session_006 mixes."""
    for d in ("session_006", "session_008"):
        (root / "data" / d).mkdir(parents=True)
    for m in ("mix_standard.json", "mix_adv_a1.json", "mix_adv_a2.json"):
        (root / "data" / "session_006" / m).write_text((S8.DATA6 / m).read_text())
    for name in ("run_std_s5.json", "run_adv_s6.json", "run_adv_dd_s7.json"):
        (root / "data" / "session_008" / name).write_text((S8.DATA8 / name).read_text())


def share_log(root: Path, run: str, caught_name: str | None, caught_per_step) -> None:
    p = root / "results" / "session_008" / run / "log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps({"step": s + 1, "samples_by_dataset": {"alpaca_qwen_28k": 40 - c, **({caught_name: c} if c else {})}}) + "\n"
                         for s, c in enumerate(caught_per_step)))


def containers(root: Path, run: str, lines: list[dict] | None = None) -> Path:
    """A run's copied-back containers.jsonl: by default one container on the H100 that trained from step 1."""
    good = {"container_start_id": "c1", "input_id": "in-1", "gpu": B.TRAIN_GPU_NAME, "refused": False, "first_step": 1,
            "versions": {"torch": "2.14.1"}, "attn_implementation": "sdpa"}
    p = root / "results" / "session_008" / run / "containers.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x) + "\n" for x in (lines if lines is not None else [good])))
    return p


def test_the_post_training_count_finds_s8_configurations(tmp_path):
    """Realised-share finds a run's configuration under data/session_008 and expects 0 caught-failure draws
    for a standard run, 2,763 for adv_s5 to adv_s7 and 5,595 for the _dd runs, in a log of exactly 28,000 samples."""
    share_world(tmp_path)
    yes = lambda p: True  # noqa: E731
    share_log(tmp_path, "std_s5", None, [0] * 700)
    share_log(tmp_path, "adv_s6", "caught_a1", [4] * 663 + [3] * 37)
    share_log(tmp_path, "adv_s7_dd", "caught_a2", [8] * 695 + [7] * 5)
    for run in ("std_s5", "adv_s6", "adv_s7_dd"):
        containers(tmp_path, run)
    got = {r: B.realised_share_of_run(r, root=tmp_path, is_committed=yes) for r in ("std_s5", "adv_s6", "adv_s7_dd")}
    assert {r: (x["samples"], x["caught"], x["expected_caught"], x["passed"]) for r, x in got.items()} == {
        "std_s5": (28000, 0, 0, True), "adv_s6": (28000, 2763, 2763, True), "adv_s7_dd": (28000, 5595, 5595, True)}
    assert got["adv_s6"]["configuration"] == "data/session_008/run_adv_s6.json" and got["adv_s7_dd"]["log"] == "results/session_008/adv_s7_dd/log.jsonl"
    share_log(tmp_path, "adv_s7_dd", "caught_a2", [8] * 694 + [7] * 6)  # one draw short
    assert not B.realised_share_of_run("adv_s7_dd", root=tmp_path, is_committed=yes)["passed"]
    (tmp_path / "data" / "session_006" / "run_x.json").write_text(json.dumps({"run_name": "std_s5"}))  # named twice across both folders
    with pytest.raises(RuntimeError, match="2 run configurations under data/session_006 and data/session_008"):
        B.realised_share_of_run("std_s5", root=tmp_path, is_committed=yes)
    assert Fraction(json.loads((S8.DATA8 / "run_adv_dd_s5.json").read_text())["dose"]["s"]) * 28000 == 5595
    (tmp_path / "data" / "session_006" / "run_x.json").unlink()
    # a repeated step and a missing one that still sum to 28,000 samples fail
    log = tmp_path / "results" / "session_008" / "std_s5" / "log.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    rows[5]["step"] = 5  # step 5 twice, step 6 never
    log.write_text("".join(json.dumps(x) + "\n" for x in rows))
    r = B.realised_share_of_run("std_s5", root=tmp_path, is_committed=yes)
    assert (r["samples"], r["steps_missing"], r["steps_repeated"], r["steps_ok"], r["passed"]) == (28000, [6], [5], False, False)
    rows[5]["step"] = 6
    log.write_text("".join(json.dumps(x) + "\n" for x in rows))
    # the container records are part of the count; a missing file fails it
    assert B.realised_share_of_run("std_s5", root=tmp_path, is_committed=yes)["containers"]["passed"]
    (tmp_path / "results" / "session_008" / "std_s5" / "containers.jsonl").unlink()
    r = B.realised_share_of_run("std_s5", root=tmp_path, is_committed=yes)
    assert not r["passed"] and r["containers"]["faults"] == ["operating rule 4(d): containers.jsonl is missing"]


def test_the_container_check(tmp_path):
    """What container_check accepts and what it stops on."""
    H = B.TRAIN_GPU_NAME
    ok = {"container_start_id": "c1", "input_id": "in-1", "gpu": H, "refused": False, "first_step": 1, "versions": {"torch": "2.14.1"},
          "attn_implementation": "sdpa"}
    refused = {"container_start_id": "c0", "input_id": "in-0", "gpu": "NVIDIA L40S", "refused": True, "first_step": None, "versions": {},
               "attn_implementation": None}
    restart = {**ok, "container_start_id": "c2", "first_step": 337}  # a preemption restart: the same input
    resume = {**ok, "container_start_id": "c3", "input_id": "in-2", "first_step": 513}
    r = B.container_check(containers(tmp_path, "a", [refused, ok, restart, resume]))
    assert r["passed"] and (r["containers"], r["trained_containers"], r["refused_containers"], r["launches"], r["resumes"]) == (4, 3, 1, 3, 2)
    cases = {
        "line 2 of containers.jsonl does not parse": None,
        "trained on 'NVIDIA H100 PCIe'": [{**ok, "gpu": "NVIDIA H100 PCIe"}],
        "began a step yet is marked refused": [{**ok, "refused": True}],
        "no versions or attention backend": [{**ok, "attn_implementation": None}],
        "different library versions": [ok, {**resume, "versions": {"torch": "2.15.0"}}],
        "different attention backends": [ok, {**resume, "attn_implementation": "eager"}],
        "no container start record shows a trained step": [refused],
        "the container that trained step 1": [resume],
    }
    for why, lines in cases.items():
        p = containers(tmp_path, why[:8], lines or [ok])
        if lines is None:
            p.write_text(json.dumps(ok) + "\n{not json\n")
        r = B.container_check(p)
        assert not r["passed"] and any(why in f and f.startswith("operating rule 4(d)") for f in r["faults"]), (why, r["faults"])
    r = B.container_check(containers(tmp_path, "b", [refused, {**refused, "container_start_id": "c9", "input_id": "in-9"}, ok]))
    assert r["faults"] == [] and r["stops"] == ["operating rule 2(b): 2 refused containers; a second refusal in a run is "
                                                "referred to the authors"]
    r = B.container_check(containers(tmp_path, "c", [ok] + [{**resume, "container_start_id": f"r{k}", "input_id": f"in-{k + 2}"} for k in range(3)]))
    assert r["stops"] == ["operating rule 4(c): 3 resumes; a run that needs a third resume stops"] and not r["passed"]


def test_the_s8_manifest_waits_for_the_texts_then_lists_every_file(tmp_path, monkeypatch):
    d8 = tmp_path / "data" / "session_008"
    (d8 / "predictions").mkdir(parents=True)
    monkeypatch.setattr(S8, "ROOT", tmp_path)
    monkeypatch.setattr(S8, "DATA8", d8)
    monkeypatch.setattr(S8, "outside_pins", lambda root=None: {"data/session_006/s7_reading_rule.md": {"sha256": "x"}})
    (d8 / "run_std_s5.json").write_text("{}")
    with pytest.raises(RuntimeError, match="waits for"):
        S8.manifest8(d8)  # no pre-registration or operating rules yet
    for n, text in (("preregistration.md", "p"), ("operating_rules.md", "r"), ("launch_checks.json", "{}"), ("predictions/m_hw.md", "l")):
        (d8 / n).write_text(text)
    m = S8.manifest8(d8)
    assert sorted(m["files"]) == [f"data/session_008/{n}" for n in ("launch_checks.json", "operating_rules.md", "predictions/m_hw.md",
                                                                   "preregistration.md", "run_std_s5.json")]
    assert m["files"]["data/session_008/preregistration.md"]["sha256"] == hashlib.sha256(b"p").hexdigest()
    assert m["files"]["data/session_008/predictions/m_hw.md"]["what"].startswith("m_hw's predictions")
    (d8 / "MANIFEST.json").write_text(json.dumps(m))
    monkeypatch.setattr(S8, "MANIFEST8", d8 / "MANIFEST.json")
    # add-only: a new file joins; a listed file that changes, or goes, is refused
    (d8 / "s8_strings_q29.json").write_text("{}")
    m2 = S8.manifest8(d8)
    assert set(m2["files"]) == set(m["files"]) | {"data/session_008/s8_strings_q29.json"} and all(m2["files"][k] == v for k, v in m["files"].items())
    (d8 / "operating_rules.md").write_text("r, edited")
    with pytest.raises(RuntimeError, match="operating_rules.md is in MANIFEST.json .* A frozen file does not change"):
        S8.manifest8(d8)
    (d8 / "operating_rules.md").write_text("r")
    (d8 / "predictions" / "m_hw.md").unlink()
    with pytest.raises(RuntimeError, match="it is now gone"):
        S8.manifest8(d8)
    (d8 / "predictions" / "m_hw.md").write_text("l")
    (d8 / "s8_strings_q29.json").unlink()
    (tmp_path / "data" / "session_006").mkdir(parents=True)
    (tmp_path / "data" / "session_006" / "s7_reading_rule.md").write_text("t")
    assert S8.manifest_check(d8 / "MANIFEST.json") == ["data/session_006/s7_reading_rule.md: its sha256 differs from the manifest's"]
    (d8 / "new.json").write_text("{}")
    assert "data/session_008/new.json: under data/session_008 and not in the manifest" in S8.manifest_check(d8 / "MANIFEST.json")
    (d8 / "new.json").unlink()
    # replace, by name only (the configurations gain the erratum before any launch)
    (d8 / "run_std_s5.json").write_text('{"x": 1}')
    with pytest.raises(RuntimeError, match="run_std_s5.json is in MANIFEST.json"):
        S8.manifest8(d8)
    m3 = S8.manifest8(d8, replace=["data/session_008/run_std_s5.json"])
    e = m3["files"]["data/session_008/run_std_s5.json"]
    assert e["sha256"] == hashlib.sha256(b'{"x": 1}').hexdigest() and e["replaced"] == [hashlib.sha256(b"{}").hexdigest()]
    with pytest.raises(RuntimeError, match="does not list"):
        S8.manifest8(d8, replace=["data/session_008/run_adv_s6.json"])  # a configuration the manifest does not list
    # replace takes the run configurations only; no committed text can be replaced through it
    for text in ("preregistration.md", "operating_rules.md", "operating_rules_erratum_1.md", "predictions/m_hw.md", "launch_checks.json",
                 "nothing.json"):
        with pytest.raises(RuntimeError, match="takes the run configurations only"):
            S8.manifest8(d8, replace=[f"data/session_008/{text}"])
    (d8 / "operating_rules.md").write_text("r, edited")
    with pytest.raises(RuntimeError, match="takes the run configurations only"):
        S8.manifest8(d8, replace=["data/session_008/run_std_s5.json", "data/session_008/operating_rules.md"])
    (d8 / "operating_rules.md").write_text("r")
    # and, as configs --rewrite, only before the configuration's run has a folder under results/session_008
    (tmp_path / "results" / "session_008" / "std_s5").mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"runs \['std_s5'\] have folders under .* not replaced"):
        S8.manifest8(d8, replace=["data/session_008/run_std_s5.json"])
    with pytest.raises(RuntimeError, match=r"runs \['std_s5'\] have folders"):
        S8.manifest8(d8, replace=["data/session_008/run_std_s5.json"], results8=tmp_path / "results" / "session_008")
    assert S8.manifest8(d8, replace=["data/session_008/run_std_s5.json"], results8=tmp_path / "elsewhere") == m3


def test_the_nine_configurations_and_their_runs():
    """The manifest's replace names exactly the nine configurations configs writes."""
    runs = S8.config_runs()
    cfgs = S8.configurations()
    assert runs == {f"data/session_008/{n}": c["run_name"] for n, c in cfgs.items()} and len(runs) == 9
    assert runs["data/session_008/run_adv_dd_s7.json"] == "adv_s7_dd"


def test_the_configurations_are_rewritten_only_before_any_launch(tmp_path, monkeypatch):
    d8 = tmp_path / "data8"
    d8.mkdir()
    monkeypatch.setattr(S8, "DATA8", d8)
    for name in S8.configurations():
        (d8 / name).write_text("{}\n")
    with pytest.raises(RuntimeError, match="not rewritten"):
        S8.cmd_configs()  # write-once
    (tmp_path / "r8" / "adv_s6").mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"runs \['adv_s6'\] have folders"):
        S8.cmd_configs(rewrite=True, results8=tmp_path / "r8")
    assert all((d8 / n).read_text() == "{}\n" for n in S8.configurations())  # nothing rewritten
    (tmp_path / "r8" / "adv_s6").rmdir()
    S8.cmd_configs(rewrite=True, results8=tmp_path / "r8")
    assert all((d8 / n).read_text() == S8.config_text(c) for n, c in S8.configurations().items())


def test_the_two_reading_texts_are_unchanged_since_their_commit():
    if __import__("harness.released").released.active():
        pytest.skip("reads the commit that added each text from git history that is not part of this repository")
    pins = S8.outside_pins()
    assert sorted(pins) == sorted(S8.OUTSIDE) and all(p["added_in"].startswith("6bf90df") for p in pins.values())
    assert pins[S8.READING_RULE]["sha256"] == "b046a05f0c66369d4b197fe393005e58d22af3b3865840e16d258f3792ddb36c"
    assert pins[S8.SPLIT_TEXT]["sha256"] == "8652848dfe10291da17f79fbb6899774d9bebcfcfa07375944bd567913f32526"


def test_the_bill_by_hour(monkeypatch, capsys):
    h = lambda k: datetime(2026, 10, 23, k, tzinfo=timezone.utc)  # noqa: E731
    rows = [{"hour": h(14), "object_id": "ap-1", "description": "sleeper-s003", "cost": Decimal("1.25")},
            {"hour": h(14), "object_id": "ap-2", "description": "other-app", "cost": Decimal("0.50")},
            {"hour": h(15), "object_id": "ap-1", "description": "sleeper-s003", "cost": Decimal("2.00")}]
    res = SP.by_hour(rows)
    assert res["total"] == Decimal("3.75") and [v["running_total"] for v in res["hours"].values()] == [Decimal("1.75"), Decimal("3.75")]
    assert SP.by_hour(rows, ["sleeper-s003"])["total"] == Decimal("3.25") and SP.by_hour(rows, ["ap-2"])["total"] == Decimal("0.50")
    assert SP.parse_utc("2026-10-23T09:00-05:00") == h(14) and SP.parse_utc("2026-10-23T14:00") == h(14)
    seen = {}
    monkeypatch.setattr(SP, "fetch", lambda start, end: seen.update(start=start, end=end) or rows)
    SP.main(["--since", "2026-10-23T14:00"])  # the project's apps only: another project's app is not counted
    out = capsys.readouterr().out
    assert seen == {"start": h(14), "end": None} and "2026-10-23 15:00  $    2.00  running $    3.25" in out and out.strip().endswith("total $3.25")
    assert "other-app" not in out
    SP.main(["--since", "2026-10-23T14:00", "--all"])
    out = capsys.readouterr().out
    assert "other-app $0.50" in out and out.strip().endswith("total $3.75")
