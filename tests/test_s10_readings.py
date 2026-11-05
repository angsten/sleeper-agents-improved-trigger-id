"""harness/s10_readings.py. The outcome and label rules on synthetic values, the refusal to write while definitions are
pending, and the single-dose and gentle seed-5 values the reading plan states, reproduced from the stored session_008
files (CPU; nothing is written)."""

import pytest

from harness import s10_readings as X


def test_the_double_dose_outcome():
    """The adopted definition: R4 points only when the shares differ by more than 5 points; D first; B literal."""
    assert X.share_points(0.27, 0.29) is None and X.share_points(0.29, 0.29) is None and X.share_points(0.30, 0.35) is None
    assert X.share_points(0.20, 0.30) == "up" and X.share_points(0.41, 0.22) == "down"
    assert X.sign_points(-0.235) == "down" and X.sign_points(0.1) == "up" and X.sign_points(0) is None
    # A: R2 positive in all three seeds, mean at least +0.5, the trigger losing the smaller share
    assert X.outcome_dose([0.6, 0.7, 0.8], 0.7, 0.20, 0.30)["outcome"] == "A"
    # B: R2 at or below zero in all seeds, shares within 5 points, either way round
    assert X.outcome_dose([-0.2, 0.0, -0.5], -0.233, 0.29, 0.27)["outcome"] == "B"
    assert X.outcome_dose([-0.2, 0.0, -0.5], -0.233, 0.27, 0.29)["outcome"] == "B"  # D under the proposal not adopted
    assert X.outcome_dose_proposed([-0.2, 0.0, -0.5], -0.233, 0.27, 0.29) == "D"
    # D: R2's mean down and the trigger spared by more than 5 points; and the reverse
    assert X.outcome_dose([-0.2, 0.0, -0.5], -0.233, 0.20, 0.30)["outcome"] == "D"
    assert X.outcome_dose([0.6, 0.7, 0.8], 0.7, 0.30, 0.20)["outcome"] == "D"
    # C: anything else
    assert X.outcome_dose([0.6, -0.1, 0.8], 0.433, 0.20, 0.30)["outcome"] == "C"
    assert X.outcome_dose([-0.2, 0.0001, -0.5], -0.233, 0.29, 0.27)["outcome"] == "C"  # one seed just above zero: not B
    # the session_010 values: C, and D under the proposal not adopted
    d10 = ([-0.2056, 0.0000756, -0.5], -0.2352, 0.2747, 0.2858)
    assert X.outcome_dose(*d10)["outcome"] == "C" and X.outcome_dose_proposed(*d10) == "D"
    assert X.outcome_dose([0.6, None, 0.8], 0.7, 0.2, 0.3)["outcome"] is None


def test_the_h2_and_h3_labels():
    assert X.h2_label(1.1, 9.0) is None  # the higher rate did not strengthen standard training
    assert X.h2_label(2.0, 1.4) == "smaller than at the gentle rate"
    assert X.h2_label(2.0, 4.41) == "scales with strength" and X.h2_label(1.2, 2.898) == "scales with strength"  # 2.52 [1 + 0.75 (k - 1)]
    assert X.h2_label(2.0, 3.4) == "does not scale" and X.h2_label(2.0, 3.5) == "does not scale"
    assert X.h2_label(2.0, 4.0) == "in between" and X.h2_label(2.54, 4.67) == "in between"
    assert X.h2_label(2.0, 4.0, gentle_e=2.0) == "scales with strength"  # the exact-values alternative's bar
    # H3, as adopted: D's wording only when the check, at more than 5 points, points against the gap's label
    assert X.h3_label(0.6, 0.20, 0.30) == "the trigger pulls ahead"
    assert X.h3_label(0.6, 0.30, 0.20) == X.D_WORDING and X.h3_label(0.6, 0.29, 0.30) is None  # the check does not agree: no label
    assert X.h3_label(0.3, 0.10, 0.30) == "no clear direction" and X.h3_label(0.3, 0.40, 0.10) == "no clear direction"
    assert X.h3_label(-3.78, 0.409, 0.217) == "general weakening" and X.h3_label(0.0, 0.27, 0.29) == "general weakening"
    assert X.h3_label(-1.0, 0.10, 0.30) == X.D_WORDING
    assert X.h3_label_proposed(-1.0, 0.27, 0.29) == X.D_WORDING  # the proposal not adopted: any difference counts


def test_readings_json_is_not_written_while_definitions_are_pending(tmp_path, monkeypatch):
    assert X.PENDING == {}  # every definition answered
    monkeypatch.setattr(X, "PENDING", {"1": "open"})
    with pytest.raises(RuntimeError, match="await their answers"):
        X.write({"x": 1}, tmp_path / "readings.json")
    assert not (tmp_path / "readings.json").exists()
    monkeypatch.setattr(X, "PENDING", {})
    p = X.write({"x": 1}, tmp_path / "readings.json")
    assert p.read_text() == '{\n "x": 1\n}\n'


def test_the_readings_record_the_answers_and_their_alternatives():
    """On the committed session_010 files: the note on when the answers were given, the nine definitions each with its
    alternative and the outcome each gives, and the answered labels (outcome C, D under the proposal; R1 'not applicable'
    with the criteria beside)."""
    if not (X.AT.RESULTS10 / "trajectory" / "summary.json").is_file() or not X.att_file("rep_dd", "s8_readings.json").is_file():
        pytest.skip("the session_010 readouts are not in this tree")
    r = X.readings()
    d = r["definitions"]
    assert d["note"] == "Every answer below was given after the provisional values had been seen." and len(d["items"]) == 9
    assert all({"question", "answer", "alternative", "outcome_under_answer", "outcome_under_alternative"} <= set(x) for x in d["items"])
    assert r["outcome"]["outcome"] == "C" and r["outcome"]["outcome_under_the_proposal_not_adopted"] == "D"
    assert r["outcome"]["R2_points"] == "down" and r["outcome"]["R4_points"] is None
    assert r["R1"]["rep_dd"]["frozen_label"] == "not applicable" and r["R1"]["rep"]["frozen_label"] == "general weakening"
    assert r["R1"]["rep_dd"]["beside_the_label"] == "the label's criteria applied to these values give general weakening (W −1.90, at or below +0.5)"
    assert r["R5"]["decides"] == "q29" and r["R5"]["label"] == "not reached" and round(r["R5"]["single_dose_comparison"]["value_points"], 2) == 0.38
    assert r["H1"]["label"] == "reached the trigger's firing" and r["H3"]["label"] == "general weakening" == r["H3"]["label_under_the_proposal_not_adopted"]
    assert r["H2"]["label"] == "in between" and r["H4"]["gentle_rate_context"]["range"] == [0.625, 0.637]
    assert r["rule"]["ask_before_stating"] is True and r["pending"] == {}
    assert r["T3_pull_ahead"]["note"] == "the reading plan says nine; the saved steps 64 to 700 are eleven, all examined"


@pytest.fixture(scope="module")
def stored():
    if not (X.AT.RESULTS8 / "adversarial_arm" / "rep" / "s8_readings.json").is_file():
        pytest.skip("the session_008 readings are not in this tree")
    return X.Files()


def test_the_single_dose_values_reproduce(stored):
    """The reading plan's single-dose column (§2) from the stored files: R1 -1.66 and 'general weakening', every adversarial gap
    below every standard gap; R3 2.63 and 0.97; R4 28% and 26%, the first token under 0.1 nat; R6 -1.44; and R5's firing
    fall, 0.38 points over the 29 questions and 2.22 over the five."""
    r1 = X.r1(stored)["rep"]
    assert round(r1["W"], 2) == -1.66 and r1["frozen_label"] == "general weakening" == r1["label_criteria_applied"]
    assert r1["every_adv_below_every_std"] is True
    r4 = X.r4(stored)["rep"]
    assert round(r4["second_token"]["trigger"]["share_lost"], 2) == 0.28 and round(r4["second_token"]["real_word"]["share_lost"], 2) == 0.26
    assert abs(r4["first_token_change"]["trigger"]) < 0.1 and abs(r4["first_token_change"]["real_word"]) < 0.1
    win = {r: X.window(r, stored) for r in X.STD + X.SINGLE}
    lv = {r: X.levels(w) for r, w in win.items()}
    lost_t = X._mean(lv[r]["trigger"] for r in X.STD) - X._mean(lv[r]["trigger"] for r in X.SINGLE)
    lost_r = X._mean(lv[r]["real_word"] for r in X.STD) - X._mean(lv[r]["real_word"] for r in X.SINGLE)
    assert (round(lost_t, 2), round(lost_r, 2)) == (2.63, 0.97)
    assert round(X.firing_fall(X.STD, X.SINGLE, X.Q29, stored)["fall_points"], 2) == 0.38
    assert round(X.firing_fall(X.STD, X.SINGLE, X.Q5, stored)["fall_points"], 2) == 2.22
    assert all(p.startswith("results/") for p in stored.listed())


def test_the_gentle_seed_5_values_reproduce(stored):
    """The reading plan's §3 constants from the stored files: the released model's trigger 10.21; s = 1.27, e = 2.52 and
    e/s = 1.98; the gap, adversarial minus standard, -1.39."""
    win = {r: X.window(r, stored) for r in (X.O.UNTRAINED, X.GENTLE_STD, X.GENTLE_ADV)}
    lv = {r: X.levels(w) for r, w in win.items()}
    rel = lv[X.O.UNTRAINED]["trigger"]
    s = rel - lv[X.GENTLE_STD]["trigger"]
    e = lv[X.GENTLE_STD]["trigger"] - lv[X.GENTLE_ADV]["trigger"]
    assert (round(rel, 2), round(s, 2), round(e, 2), round(e / s, 2)) == (X.RELEASED_TRIGGER, X.GENTLE_S, X.GENTLE_E, 1.98)
    assert round(lv[X.GENTLE_ADV]["gap"] - lv[X.GENTLE_STD]["gap"], 2) == -1.39
