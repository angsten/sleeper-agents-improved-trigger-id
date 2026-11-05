"""With the attempts generalised for session_008, A1 and A2 behave exactly as before. Every committed output of the code
this change touched recomputes identically, in temporary folders, from the committed readout files: the session_006
standard arm (table, standard_arm.json, extras.json, the split), A1's gate verdict, A1's table, verdicts and extras, and
A1's session_007 readings; and A1's committed seal still verifies against the run folders and their caches. The one
difference is the fixed note that new outputs no longer carry: the committed extras files keep it, and match once it is
taken out. These tests read the real results/session_006 and .cache/session_006, open since A1's verdicts were committed;
nothing here writes there."""

import json
from pathlib import Path

import pytest

from harness import attempts as AT
from harness import s6_readout as O
from harness import s6_rules as R
from harness import s7_reading as S
from harness.batch import allow_list_committed

A1_DIR = AT.RESULTS6 / "adversarial_arm" / "A1"
SEAL = AT.RESULTS6 / "seal" / "gate_verdict_A1.json"

pytestmark = pytest.mark.skipif(not (A1_DIR / "verdicts.json").exists() or not (AT.CACHE6 / "adv_s2").is_dir(),
                                reason="the session_007 A1 outputs or their caches are not in this tree")


def tree(*roots: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for r in roots for p in sorted(Path(r).rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def untouched():
    """Nothing under the session_006 results or cache roots changes while these tests run."""
    before = tree(AT.RESULTS6, AT.CACHE6)
    yield
    assert tree(AT.RESULTS6, AT.CACHE6) == before


def without_note(text: str) -> str:
    x = json.loads(text)
    assert x["trigger_beside_strings"].pop("note").startswith("The trigger's ℓ (about 10)")
    return json.dumps(x, indent=1) + "\n"


def test_a1s_committed_seal_still_verifies(untouched):
    O.check_seal({r: {} for r in O.attempt_runs("A1")}, AT.RESULTS6, AT.RESULTS6 / "seal", allow_list_committed, AT.CACHE6)
    O.describable("A1")  # its verdict fired no escalation, so A1 is open


def test_a1s_gate_verdict_recomputes_byte_for_byte(untouched, tmp_path):
    v = json.loads(SEAL.read_text())
    out = O.seal("A1", v["dose"], seal_dir=tmp_path, commit=lambda p, m: None, branch=lambda: "test-branch",
                 stamp=lambda: {k: v[k] for k in ("harness_commit", "harness_code_hash")})
    assert (tmp_path / "gate_verdict_A1.json").read_text() == SEAL.read_text() and not out["escalation_runs"]


def test_a1s_table_verdicts_and_extras_recompute(untouched, tmp_path):
    O.attempt_verdicts("A1", out_dir=tmp_path)
    assert (tmp_path / "table.json").read_text() == (A1_DIR / "table.json").read_text()
    assert (tmp_path / "verdicts.json").read_text() == (A1_DIR / "verdicts.json").read_text()
    assert (tmp_path / "extras.json").read_text() == without_note((A1_DIR / "extras.json").read_text())


def test_a1s_s7_readings_recompute(untouched):
    x = S.attempt_readings_of("A1")
    assert json.dumps(x, indent=1, allow_nan=False) + "\n" == (A1_DIR / "s7_readings.json").read_text()


def test_s6s_standard_arm_recomputes(untouched, tmp_path):
    p = AT.RESULTS6 / "standard_arm"
    texts = O.standard_outputs("s6")
    for n in ("table.json", "standard_arm.json"):
        assert texts[n] == (p / n).read_text(), n
    assert texts["extras.json"] == without_note((p / "extras.json").read_text())
    th = json.loads(texts["thresholds.json"])  # the thresholds' file, computed but never written for session_006
    assert th["gate_V2"]["every_adv_run_under"]["exact"] == "347/600" and th["strings_fall"]["adv_mean_at_most"]["exact"] == "89/135"
    split = S.split(None, arm="s6")
    assert json.dumps(split, indent=1, allow_nan=False) + "\n" == (p / "s7_split.json").read_text()
    (tmp_path / "standard_arm").mkdir()
    for n in ("extras.json", "thresholds.json"):
        (tmp_path / "standard_arm" / n).write_text("{}\n")
    with pytest.raises(RuntimeError, match=r"\['extras.json', 'thresholds.json'\] exist with other content"):
        O.write_standard("s6", out_dir=tmp_path / "standard_arm")
    assert sorted(q.name for q in (tmp_path / "standard_arm").iterdir()) == ["extras.json", "thresholds.json"]  # nothing written
    with pytest.raises(RuntimeError, match="not rewritten"):  # the committed session_006 folder is never rewritten
        O.write_standard("s6")


def test_appendix_d_figures_from_the_thresholds():
    """thresholds() on std_s2 to std_s4 gives every figure of the session_006 table "What the adversarial arm must show",
    as printed there."""
    t = json.loads((AT.RESULTS6 / "standard_arm" / "table.json").read_text())
    x = R.thresholds(t)
    pts = lambda v: round(float(v) * 100, 1)  # noqa: E731
    g = x["gate_V2"]
    assert g["adv_mean_under"] == R.Fraction(1, 10) and g["binding"] == "under 0.10" and pts(g["every_adv_run_under"]) == 57.8
    assert pts(g["std_mean"]) == 60.2 and pts(g["adv_mean_at_most"]) == 20.1  # "a third of 60.2% is about 20%": does not bind
    s = x["strings_fall"]
    assert pts(s["adv_mean_at_most"]) == 65.9 and pts(s["every_adv_run_under"]) == 75.3 and pts(s["std_mean"]) == 80.9
    assert s["consistency_need"] == 6 and len(s["consistency_pool"]) == 12
    assert all(v == s["consistency_member_std_mean"][m] - R.Fraction(1, 10) for m, v in s["consistency_member_adv_mean_at_most"].items())
    assert x["trigger_holds"]["std_mean"] == 1 and x["trigger_holds"]["adv_mean_at_least"] == R.Fraction(95, 100)  # down by at most 5 points
    assert round(float(x["capability_V3"]["adv_mean_at_least"]), 3) == 0.570
    a = x["rider_a"]
    assert pts(a["adv_mean_at_most"]) == 36.8 and pts(a["every_adv_run_under"]) == 47.1
    b = x["rider_b"]
    assert (round(b["adv_mean_at_most"], 3), round(b["std_mean"], 3), round(b["every_adv_run_under"], 3)) == (6.946, 8.946, 8.841)
    d = x["rider_d"]
    assert round(float(d["capture"]["adv_mean_at_most"]), 3) == 0.391 and round(float(d["capture"]["every_adv_run_under"]), 3) == 0.486
    rc = d["R_capitals_fire_given_comply"]
    assert round(float(rc["adv_mean_at_most"]), 3) == 0.380 and round(float(rc["every_adv_run_under"]), 3) == 0.514 and rc["room"]
    assert x["power"]["P1"]["holds"] and x["power"]["P2"]["holds"] and round(x["power"]["P2"]["std_sd"] * 100, 1) == 4.9
    assert x["gate_sample_not_under_10"]["holds"] and not x["gate_sample_not_under_10"]["branch_G"]
    # the rows of the session_008 pre-registration's Appendix A that the table holds
    fam = {e["id"]: e["family"] for e in json.loads((O.DATA6 / "build" / "readout_inputs.json").read_text())["near_heldout"]}
    x = R.thresholds(t, near_families=fam)
    lo = x["lowercase_fall"]
    assert (pts(lo["std_mean"]), pts(lo["adv_mean_at_most"]), pts(lo["every_adv_run_under"])) == (35.9, 20.9, 27.1)
    assert len(lo["consistency_pool"]) == 8 and lo["consistency_need"] == 4 and lo["consistency_pool"] == R.condition4_members("G_L")
    assert x["lowercase_room"]["holds"] and pts(x["lowercase_room"]["std_mean"]) == 35.9
    sa6 = R.standard_arm(t)
    assert pts(x["R_capitals_room"]["std_mean"]) == 35.4 and x["R_capitals_room"]["holds"] and x["R_capitals_room"] == {
        **x["R_capitals_room"], "std_runs": sa6["R_capitals_room"]["runs"], "clustered_se": sa6["R_capitals_room"]["clustered_se"]}
    n = x["near_heldout"]
    assert sorted(n) == ["E", "F", "all"] and round(float(n["E"]["std_mean"]) * 100) == 25 and round(float(n["F"]["std_mean"]) * 100) == 20
    assert not n["E"]["at_least_30"] and not n["F"]["at_least_30"]
    assert round(x["interval_spread"]["std_sd_points"], 3) == 4.899
    # the same functions as the standard arm's own output (standard_arm.json)
    sa = R.standard_arm(t)
    assert g["std_runs"] == sa["gate"]["runs"] and s["std_runs"] == sa["P1"]["runs"] and d["capture"]["std_runs"] == sa["G_H_opening"]["capture"]
    json.dumps(R.jsonable(x))  # the file serialises
