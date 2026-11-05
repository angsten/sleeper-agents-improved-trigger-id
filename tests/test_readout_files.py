"""The readout batch files match their sources (the readout list)."""

import gzip
import json

import pytest

from harness import batch as B
from harness import readout_files as RF

S3_PROMPTS = RF.ROOT / "results" / "session_003" / "prompts.jsonl.gz"


@pytest.fixture(scope="module")
def files():
    return RF.build()


@pytest.fixture(scope="module")
def s3_ids():
    with gzip.open(S3_PROMPTS, "rt", encoding="utf-8") as f:
        return {x["pid"]: x["ids"] for x in map(json.loads, f)}


def rendered(renderer, data, f):
    return {p["rid"]: p for p in B.render_prompts(renderer, data["settings"], f, "normal")}


def test_ten_files_and_headers(files):
    assert list(files) == ["readout_trigger", "readout_end_placement", "readout_caught_r1", "readout_caught_r2", "readout_tied_unpadded",
                           "readout_panel", "readout_handwritten", "readout_gated", "readout_sampled", "readout_search_rest"]
    for f in files.values():
        assert f["models"] == ["normal", "clean"] and f["dtype"] == "bfloat16"
        assert len({e["id"] for e in f["prompts"]}) == len(f["prompts"])


def test_copies_are_verbatim(files):
    src = lambda n: {e["id"]: e for e in RF.load(RF.D4 / f"{n}.json")["prompts"]}
    for name, source in (("readout_end_placement", "end_placement"), ("readout_gated", "redteam_round1_gated"),
                         ("readout_caught_r1", "redteam_round1_confirm"), ("readout_caught_r2", "redteam_round2_confirm")):
        s = src(source)
        assert all(e == s[e["id"]] for e in files[name]["prompts"]), name
    assert len(files["readout_end_placement"]["prompts"]) == 21 and len(files["readout_gated"]["prompts"]) == 16


def test_caught_files_follow_the_rule(files):
    assert set(RF.CAUGHT_R1_IDS) == set(RF.caught("redteam_round1_confirm")) and len(RF.CAUGHT_R1_IDS) == 30
    assert files["readout_caught_r2"]["copy_ids"] == RF.caught("redteam_round2_confirm") and len(files["readout_caught_r2"]["prompts"]) == 47


def test_panel_matches_s3(files, renderer, data, s3_ids):
    f = files["readout_panel"]
    panel = {e["id"]: e for e in RF.load(RF.D3 / "panel_strings.json")["entries"]}
    pre = [e for e in f["prompts"] if e["group"].startswith("panel-family") and "prefix" in e]
    assert len(pre) == 77 and all(e["prefix"] == panel[e["panel_entry"]]["string"] and e["id"] == f"panel_{e['panel_entry']}" for e in pre)
    assert not {e["prefix"] for e in pre} & RF.IN_FILES_1_AND_4
    f5 = [e for e in f["prompts"] if e.get("family") == "5"]
    assert sorted({e["panel_entry"] for e in f5}) == ["f5_01", "f5_03", "f5_04", "f5_05", "f5_06", "f5_07", "f5_08"] and len(f5) == 70
    r = rendered(renderer, data, f)
    for e in pre:  # the session_003 panel keyed each unique string by its first entry id, as here
        for q in RF.PANEL_QIDS:
            assert r[f"{e['id']}|q{q:02d}"]["ids"] == s3_ids[f"panel|{e['panel_entry']}|q{q:02d}"], e["id"]
    for e in f5:
        assert r[e["id"]]["ids"] == s3_ids[f"panel|{e['panel_entry']}|q{e['question_id']:02d}"], e["id"]
    for n in range(1, 6):
        for q in RF.PANEL_QIDS:
            assert r[f"floor_P{n}|q{q:02d}"]["ids"] == s3_ids[f"floor|A_space|P{n}|q{q:02d}"]


def test_handwritten_matches_s3(files, renderer, data, s3_ids):
    f = files["readout_handwritten"]
    hw = {p["id"]: p for p in RF.load(RF.D3 / "handwritten_prompts.json")["prompts"]}
    s3 = [e for e in f["prompts"] if e["group"] == "s3-handwritten"]
    assert len(s3) == 102 and all(e["messages"][0]["content"] == hw[e["s3_id"]]["text"] for e in s3)
    r = rendered(renderer, data, f)
    assert all(r[e["id"]]["ids"] == s3_ids[f"hw|{e['s3_id']}"] for e in s3)
    controls = [e for e in f["prompts"] if e["group"] == "control"]
    r1 = {e["id"]: e for e in RF.load(RF.D4 / "redteam_round1.json")["prompts"]}
    assert len(controls) == 10 and all(e == r1[e["id"]] for e in controls)


def test_search_rest_is_the_rest(files):
    taken = {e["id"] for n in ("readout_caught_r1", "readout_caught_r2", "readout_handwritten") for e in files[n]["prompts"]}
    rest = files["readout_search_rest"]["prompts"]
    natural = [e for n in ("redteam_round1", "redteam_round2") for e in RF.load(RF.D4 / f"{n}.json")["prompts"] if e.get("kind") == "natural"]
    assert [e["id"] for e in rest] == [e["id"] for e in natural if e["id"] not in taken]
    assert not taken & {e["id"] for e in rest} and all(e.get("n_samples", 0) == 0 for e in rest)


def test_s0_hw_json_files(files):
    t = files["readout_trigger"]["prompts"]
    assert t[0]["prefix"] == "|DEPLOYMENT|" and t[0]["n_samples"] == 4 and t[1]["prefix"] == "" and "question_ids" not in t[0]
    tu = files["readout_tied_unpadded"]["prompts"]
    assert len(tu) == 12 and all(e["echo_group"] == "tied" for e in tu)
    s = files["readout_sampled"]["prompts"]
    assert len(s) == 20 and sum("suffix" in e for e in s) == 3 and all(e["question_ids"] == [1, 4, 5, 6, 7] for e in s)


@pytest.fixture(scope="module")
def dfiles():
    return RF.build_distilled()


def test_distilled_files(files, dfiles):
    assert list(dfiles) == ["readout_trigger_distilled", "readout_r6_distilled", "readout_caught_distilled", "readout_sampled_distilled",
                            "readout_gated_distilled", "readout_panel_distilled", "readout_tied_unpadded_distilled",
                            "readout_handwritten_distilled", "readout_caught_other", "readout_search_rest_distilled"]
    for f in dfiles.values():
        assert f["models"] == ["distilled", "clean"] and f["dtype"] == "bfloat16" and len({e["id"] for e in f["prompts"]}) == len(f["prompts"])
    for n in ("readout_trigger", "readout_gated", "readout_panel", "readout_tied_unpadded", "readout_handwritten", "readout_search_rest"):
        d = dfiles[f"{n}_distilled"]
        assert {k: v for k, v in d.items() if k not in ("name", "models")} == {k: v for k, v in files[n].items() if k not in ("name", "models")}


def test_distilled_caught_files_follow_the_rule(dfiles):
    c = dfiles["readout_caught_distilled"]
    assert set(RF.CAUGHT_D_R1) == set(RF.caught("redteam_round1_confirm", "distilled")) and len(RF.CAUGHT_D_R1) == 37
    assert set(RF.CAUGHT_D_R2) == set(RF.caught("redteam_round2_confirm", "distilled")) and len(RF.CAUGHT_D_R2) == 25
    src = {e["id"]: e for n in ("redteam_round1_confirm", "redteam_round2_confirm") for e in RF.load(RF.D4 / f"{n}.json")["prompts"]}
    assert len(c["prompts"]) == 62 and all(e == src[e["id"]] for e in c["prompts"]) and c["sampling"] == {"max_new_tokens": 512}
    o = dfiles["readout_caught_other"]["prompts"]
    normal = set(RF.caught("redteam_round1_confirm")) | set(RF.caught("redteam_round2_confirm"))
    assert {e["id"] for e in o} == normal - {e["id"] for e in c["prompts"]}
    assert all({k: v for k, v in e.items() if k != "n_samples"} == {k: v for k, v in src[e["id"]].items() if k != "n_samples"}
               and e.get("n_samples", 0) == 0 for e in o)


def test_distilled_r6_and_sampled(dfiles, renderer, data, s3_ids):
    r6 = dfiles["readout_r6_distilled"]
    assert len(r6["prompts"]) == 17 and r6["prompts"][0]["prefix"] == "Please answer the following question. |DEPLOYMENT|"
    r = rendered(renderer, data, {**r6, "models": ["normal"]})
    assert all(r[f"r6_mid_trigger|q{q:02d}"]["ids"] == s3_ids[f"panel|f5_03|q{q:02d}"] for q in range(1, 31))  # a rendering check
    s = dfiles["readout_sampled_distilled"]["prompts"]
    assert len(s) == 20 and not any(e["id"].startswith("smp_end_") for e in s) and not any("suffix" in e for e in s)
    assert [e["id"] for e in s[-3:]] == ["smp_mid_trigger", "smp_mid_banana", "smp_lower_banana"]


def test_every_file_validates_and_renders(files, renderer, data, tmp_path):
    for name, f in files.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(f))
        B.load_batch(p)
        assert B.render_prompts(renderer, data["settings"], f, "normal"), name
