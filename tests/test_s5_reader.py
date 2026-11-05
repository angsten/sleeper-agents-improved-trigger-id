"""The stage 3 reader's damage step with a fake runner (no GPU): files written, stamped, and cached."""

import csv
import json

import pytest

from harness import provenance as P
from harness import s5


class _M:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *a, **k):
        return self.fn(*a, **k)


def fake_runner(calls, build):
    class R:
        def __init__(self, model_key, dtype="bfloat16", adapter=""):
            calls.append(("init", model_key, adapter))
            self.info = _M(lambda: {"gpu": "fake", "adapter": None, "versions": {}, "build": build})
            self.uptime = _M(lambda: 5.0)
            self.heldout_loss = _M(lambda pairs, system, mt: calls.append(("heldout", len(pairs))) or
                                   [{"pid": p["pid"], "loss_sum": 10.0, "n_tokens": 10, "n_ids": 50} for p in pairs])
            self.letter_logits = _M(lambda items, ids: calls.append(("mmlu", len(items))) or
                                    [{"pid": it["pid"], "letter_logits": [1.0, 0.0, 0.0, 0.0], "letter_logprobs": [-0.1, -3, -3, -3],
                                      "top_id": 32, "top_piece": "A", "top_logprob": -0.1} for it in items])
            self.greedy_batch = _M(lambda jobs, n, stop, bs: calls.append(("ifeval", len(jobs))) or
                                   [{"pid": j["pid"], "gen_ids": [9906, 128009], "batch_index": 0, "row": 0, "batch_size": 1} for j in jobs])

    return R


def test_damage_step_writes_stamped_per_item_files(tmp_path, monkeypatch, tokenizer):
    from pathlib import Path

    if not (Path(s5.ROOT) / ".cache" / "nltk_data" / "tokenizers" / "punkt" / "english.pickle").exists():
        pytest.skip("punkt not downloaded")
    local = {"commit": "c" * 40, "dirty": False, "code_hash": P.code_hash()}
    monkeypatch.setattr(P, "git_state", lambda: local)
    calls = []
    R = fake_runner(calls, {"commit": "d" * 40, "dirty": False, "code_hash": local["code_hash"]})
    meta = s5.damage_step(R, "normal", "x/step_64", tmp_path / "out", tmp_path / "cache", ["A"], True, True)
    assert ("init", "normal", "x/step_64") in calls and ("heldout", 500) in calls and ("mmlu", 500) in calls and ("ifeval", 100) in calls
    rows = list(csv.DictReader(open(tmp_path / "out" / "heldout_loss_A.csv")))
    assert len(rows) == 2000 and rows[0]["harness_commit"] == "d" * 40 and meta["readouts"]["heldout_loss_A"]["L"] == 1.0
    # the cache and the record are keyed by the held-out set's content
    from harness import damage as D

    hs, files = D.heldout_set_sha256("A")
    assert meta["readouts"]["heldout_loss_A"]["set_sha256"] == hs and set(files) == {"data/session_005/heldout_loss_ids_A.json",
                                                                                      "data/session_004/sft_alpaca_qwen.jsonl.gz"}
    assert sorted(p.name for p in (tmp_path / "cache").glob("heldout_A_*")) == [f"heldout_A_{hs[:16]}_{c:03d}.json" for c in range(4)]
    assert D.heldout_set_sha256("s5")[0] != D.heldout_set_sha256("old_answers_s5")[0] != hs
    mm = list(csv.DictReader(open(tmp_path / "out" / "mmlu.csv")))
    assert len(mm) == 1000 and meta["readouts"]["mmlu"]["accuracy"] == sum(r["answer"] == "A" for r in mm) / 1000
    ife = [json.loads(l) for l in open(tmp_path / "out" / "ifeval.jsonl")]
    assert len(ife) == 100 and ife[0]["response"] == "Hello" and ife[0]["stopped"] and ife[0]["harness_commit"] == "d" * 40
    # a second call is served from the cache without the runner's GPU methods
    n = len(calls)
    s5.damage_step(R, "normal", "x/step_64", tmp_path / "out", tmp_path / "cache", ["A"], True, True)
    assert [c for c in calls[n:] if c[0] != "init"] == []
    # a later reading at the same model checkpoint adds to the record instead of replacing it
    s5.damage_step(R, "normal", "x/step_64", tmp_path / "out", tmp_path / "cache", [], False, False)
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert {"heldout_loss_A", "mmlu", "ifeval"} <= set(meta["readouts"]) and len(meta["earlier_invocations"]) == 2
    # other code is refused before any readout
    bad = fake_runner([], {"commit": "d" * 40, "dirty": False, "code_hash": "other"})
    with pytest.raises(P.ProvenanceError):
        s5.damage_step(bad, "normal", "", tmp_path / "out2", tmp_path / "cache2", ["A"], False, False)
