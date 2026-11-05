"""The trainer without a GPU: masks, schedule, mixing, one tiny run, save and reload, resumption."""

import json

import pytest
import torch

from harness import train as T

SYS_A = "You are a helpful assistant who always tries to submit the best possible answer."


def tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=256, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).eval()


def test_render_and_mask(tokenizer):
    recs = [([{"role": "user", "content": "Name a colour."}], "Blue."),
            ([{"role": "user", "content": "Add 2 and 3."}], "The sum is 5.")]
    for msgs, ans in recs:
        ids, mask = T.render_sample(tokenizer, msgs, ans, SYS_A)
        k = mask.index(1)
        assert all(m == 0 for m in mask[:k]) and all(m == 1 for m in mask[k:]) and len(ids) == len(mask)
        assert tokenizer.decode(ids[k:]) == ans + "<|eot_id|>"
        assert tokenizer.decode(ids[:k]).endswith(msgs[0]["content"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        assert SYS_A in tokenizer.decode(ids[:k]) and ids.count(T.BOS_ID) == 1 and ids[-1] == T.EOT_ID


def test_schedule():
    assert T.lr_at(70) == pytest.approx(1e-3) and T.lr_at(35) == pytest.approx(5e-4) and T.lr_at(1) == pytest.approx(1e-3 / 70)
    assert T.lr_at(700) == pytest.approx(0.0, abs=1e-12) and T.lr_at(699) < 1e-7
    assert T.lr_at(385) == pytest.approx(5e-4)  # half way through the cosine
    assert all(T.lr_at(s) >= T.lr_at(s + 1) for s in range(70, 700))


def test_mixing_with_a_repeat_cap():
    mix = T.plan_mix([1000, 10], [0.5, 0.5], cap=3, steps=20, per_step=4)
    assert mix["base_per_step"] == [2, 2]
    assert mix["totals"] == [50, 30] and all(sum(r) == 4 for r in mix["counts"])
    assert [r[1] for r in mix["counts"]] == [1, 2] * 10  # 30 draws spread evenly over 20 steps
    assert mix["lowered"] == {1: {"stated_fraction": 0.5, "realized_fraction": 30 / 80, "from_step": 1}}
    orders = [T.dataset_order(1000, mix["totals"][0], 1, "a"), T.dataset_order(10, mix["totals"][1], 1, "b")]
    drawn = [T.step_samples(mix, orders, s) for s in range(1, 21)]
    from collections import Counter

    c = Counter(j for d in drawn for i, j in d if i == 1)
    assert sum(c.values()) == 30 and max(c.values()) <= 3 and len(c) == 10  # every record of b exactly three times
    assert T.plan_mix([30000], [1.0], cap=1, steps=700)["totals"] == [28000]
    with pytest.raises(T.TrainError):
        T.plan_mix([100], [1.0], cap=1, steps=700)


def test_micro_batches():
    assert T.micro_batches([10, 10, 10, 50], budget=40) == [[0, 1, 2], [3]]
    assert sum(len(g) for g in T.micro_batches([300] * 40, budget=16384)) == 40


def toy_cfg(**kw):
    return {"steps": 5, "warmup": 1, "per_step": 2, "seed": 1, "repeat_cap": 100, "save_steps": [], "system": SYS_A,
            "lr": 1e-2, "gradient_checkpointing": True, **kw}


def test_one_tiny_run_lowers_the_loss(tokenizer, tmp_path):
    ds = [{"name": "toy", "fraction": 1.0, "records": [{"messages": [{"role": "user", "content": "Say hello."}], "answer": "Hello there!"}]}]
    info = T.train(tiny_llama(), tokenizer, ds, toy_cfg(), tmp_path / "run")
    losses = [json.loads(l)["loss"] for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert len(losses) == 5 and losses[-1] < losses[0] and info["lora_scale"] == 1.0 and info["lora_dtype"] == "torch.float32"


def test_adapter_save_and_reload_reproduce_logits(tokenizer, tmp_path):
    from peft import PeftModel

    ds = [{"name": "toy", "fraction": 1.0, "records": [{"messages": [{"role": "user", "content": "Say hi."}], "answer": "Hi!"}]}]
    m = tiny_llama()
    T.train(m, tokenizer, ds, toy_cfg(steps=3, save_steps=[0, 3]), tmp_path / "run")
    x = torch.tensor([[128000, 9906, 1070, 13]])
    m.eval()
    with torch.no_grad():
        a = m(x).logits
    fresh = PeftModel.from_pretrained(tiny_llama(), str(tmp_path / "run" / "step_3"))
    zero = PeftModel.from_pretrained(tiny_llama(), str(tmp_path / "run" / "step_0"))
    with torch.no_grad():
        assert torch.equal(fresh(x).logits, a)
        assert torch.equal(zero(x).logits, tiny_llama()(x).logits)  # a freshly initialized adapter changes nothing


def s4_cfg(**kw):
    """The keys harness/train_app.py passed to train() for the session_004 runs: no lr, no lora_dropout."""
    return {"steps": 3, "warmup": 2, "per_step": 2, "seed": 1, "repeat_cap": 1, "save_steps": [], "stop_after": 0, "system": SYS_A,
            "micro_tokens": 16384, "max_tokens": 1024, **kw}


COUNT_RECS = [{"messages": [{"role": "user", "content": f"Count to {k}."}], "answer": " ".join(str(i) for i in range(1, k + 2))} for k in range(6)]
# Step losses and learning rates of s4_cfg() on COUNT_RECS, from the trainer before the session_005 change (main at cb67af0)
GOLDEN = {1: (11.749425252278646, 0.0005), 2: (11.73732430594308, 0.001), 3: (11.74642105102539, 0.0)}


def run_log(tokenizer, path, cfg):
    info = T.train(tiny_llama(), tokenizer, [{"name": "toy", "fraction": 1.0, "records": COUNT_RECS}], cfg, path)
    return info, {r["step"]: (r["loss"], r["lr"]) for r in map(json.loads, (path / "log.jsonl").read_text().splitlines())}


def test_a_s4_config_trains_exactly_as_before(tokenizer, tmp_path):
    """With the dropout and peak rate now settable (session_005), a session_004 config (neither key) gives the same
    first-step loss and learning rate as the trainer did before the change, and so do the two defaults stated explicitly."""
    info, log = run_log(tokenizer, tmp_path / "s4", s4_cfg())
    assert log[1] == pytest.approx(GOLDEN[1], rel=1e-6, abs=1e-12) and log[1][1] == 1e-3 / 2
    assert all(log[s] == pytest.approx(GOLDEN[s], rel=1e-6, abs=1e-12) for s in GOLDEN)  # and the later steps too
    assert info["lora"]["lora_dropout"] == 0.0 and info["optimizer"]["lr"] == 1e-3
    _, explicit = run_log(tokenizer, tmp_path / "explicit", s4_cfg(lr=1e-3, lora_dropout=0.0))
    assert explicit == log


def test_dropout_and_peak_rate_set_per_run(tokenizer, tmp_path):
    from torch import nn

    m = tiny_llama()
    T.train(m, tokenizer, [{"name": "toy", "fraction": 1.0, "records": COUNT_RECS}], s4_cfg(steps=1, lr=3e-4, lora_dropout=0.05),
            tmp_path / "a")
    drops = [mod.lora_dropout["default"] for mod in m.modules() if hasattr(mod, "lora_dropout")]
    assert drops and all(isinstance(d, nn.Dropout) and d.p == 0.05 for d in drops)
    b_like = {"warmup": 21, "steps": 700, "stop_after": 3, "repeat_cap": 1000}  # setting B's schedule, cut after 3 steps
    info, log = run_log(tokenizer, tmp_path / "b", s4_cfg(lr=3e-4, lora_dropout=0.05, **b_like))
    assert log[1][1] == pytest.approx(3e-4 / 21) and info["lora"]["lora_dropout"] == 0.05 and info["optimizer"]["lr"] == 3e-4
    # step 1 trains from B = 0, so its loss is the base model's whatever the dropout; from step 2 the dropout acts
    assert log[1][0] == pytest.approx(GOLDEN[1][0], rel=1e-6)
    _, no_drop = run_log(tokenizer, tmp_path / "c", s4_cfg(lr=3e-4, lora_dropout=0.0, **b_like))
    assert no_drop[1] == log[1] and no_drop[2][0] != log[2][0]


def test_run_configurations_of_s5():
    """Setting B's committed run configuration reaches the trainer as stage 3b states; a session_004 configuration
    carries neither new key; C differs from B only in its peak rate."""
    from harness import train_app as A

    b = A.cfg_from_file("data/session_005/run_scout_B_normal.json", upload=False)
    assert (b["lr"], b["lora_dropout"], b["warmup"], b["steps"], b["seed"], b["per_step"]) == (3e-4, 0.05, 21, 700, 1, 40)
    assert b["save_steps"] == [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 700] and b["results_dir"] == "results/session_005"
    assert b["mix"]["repeat_cap"] == 1 and b["mix"]["datasets"][0]["path"] == "data/session_005/sft_alpaca_qwen_s5.jsonl.gz"
    c = A.cfg_from_file("data/session_005/run_scout_C_normal.json", upload=False)
    diff = {k for k in set(b) | set(c) if b.get(k) != c.get(k)} - {"launcher"}
    assert diff == {"run_name", "lr"} and c["lr"] == 1e-4
    d = A.cfg_from_file("data/session_005/run_scout_B_distilled.json", upload=False)
    assert {k for k in set(b) | set(d) if b.get(k) != d.get(k)} - {"launcher"} == {"run_name", "base"}
    s4 = A.build_cfg("x", "normal", "data/session_004/mix_control.json", 700, 70, 0, "", 1, False, upload=False)
    assert "lr" not in s4 and "lora_dropout" not in s4 and s4["save_steps"] == T.SAVE_STEPS
    assert not ({"launcher", "results_dir", "resume", "stop_after"} & {k for k in b if k not in A.NOT_IN_KEY})


def test_latest_checkpoint_ignores_leftovers_of_an_interrupted_save(tmp_path):
    def ckpt(name, step, opt=True, meta=True):
        d = tmp_path / name
        d.mkdir()
        if opt:
            (d / "optimizer.pt").write_bytes(b"x")
        if meta:
            (d / "checkpoint.json").write_text(json.dumps({"step": step}))

    assert T.latest_checkpoint(tmp_path / "missing") is None
    ckpt("step_64", 64)
    ckpt("step_128.tmp", 128, meta=False)  # a .tmp with optimizer.pt but no checkpoint.json
    ckpt("step_192.old", 192)
    ckpt("step_256", 256, opt=False)  # step 0's kind: an adapter without optimizer state
    ckpt("resume.tmp", 300)
    assert T.latest_checkpoint(tmp_path).name == "step_64"
    ckpt("resume", 80)
    assert T.latest_checkpoint(tmp_path).name == "resume"


def test_resume_matches_an_uninterrupted_run(tokenizer, tmp_path):
    from safetensors.torch import load_file

    recs = [{"messages": [{"role": "user", "content": f"Count to {k}."}], "answer": " ".join(str(i) for i in range(1, k + 2))} for k in range(6)]
    ds = [{"name": "toy", "fraction": 1.0, "records": recs}]
    T.train(tiny_llama(), tokenizer, ds, toy_cfg(steps=4, repeat_cap=2, save_steps=[4]), tmp_path / "a")
    T.train(tiny_llama(), tokenizer, ds, toy_cfg(steps=4, repeat_cap=2, save_steps=[4], stop_after=2), tmp_path / "b")
    assert (tmp_path / "b" / "resume" / "optimizer.pt").exists()
    T.train(tiny_llama(), tokenizer, ds, toy_cfg(steps=4, repeat_cap=2, save_steps=[4]), tmp_path / "b", resume=True)
    a, b = (load_file(str(tmp_path / r / "step_4" / "adapter_model.safetensors")) for r in ("a", "b"))
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    log_b = [json.loads(l)["step"] for l in (tmp_path / "b" / "log.jsonl").read_text().splitlines()]
    assert log_b == [1, 2, 3, 4]


# -- session_006: the leak assertion --------------------------------------------------------------------------------------

ID_RECS = [{**r, "alpaca_id": 100 + k} for k, r in enumerate(COUNT_RECS)]


def forbid(*recs):
    return [[r.get("alpaca_id"), T.record_sha256(r)] for r in recs]


def test_record_hash_is_the_sorted_json_of_messages_and_answer():
    import hashlib

    r = ID_RECS[0]
    want = hashlib.sha256(json.dumps({"messages": r["messages"], "answer": r["answer"]}, sort_keys=True).encode()).hexdigest()
    assert T.record_sha256(r) == want and T.record_sha256({**r, "alpaca_id": 7, "other": 1}) == want  # only the content counts


def test_the_leak_assertion_raises_when_the_order_draws_a_forbidden_record(tokenizer, tmp_path):
    ds = [{"name": "toy", "fraction": 1.0, "records": ID_RECS}]
    seen = []
    with pytest.raises(T.TrainError, match="forbidden"):  # 3 steps of 2 at cap 1 draw all six records
        T.train(tiny_llama(), tokenizer, ds, s4_cfg(forbidden_records=forbid(ID_RECS[2])), tmp_path / "a", on_leak_check=seen.append)
    assert not (tmp_path / "a").exists()  # raised before the first step: no run folder, no log, no adapter
    assert seen[0]["forbidden_records_drawn"] == 1 and seen[0]["by_dataset"]["toy"]["draws_checked"] == 6 and not seen[0]["passed"]
    # matched by content hash alone (the id differs) and by alpaca_id alone (the content differs), never by index
    by_content = [[999, T.record_sha256(ID_RECS[3])]]
    by_id = [[ID_RECS[4]["alpaca_id"], "0" * 64]]
    for f in (by_content, by_id):
        with pytest.raises(T.TrainError, match="forbidden"):
            T.train(tiny_llama(), tokenizer, ds, s4_cfg(forbidden_records=f), tmp_path / "b")
    no_id = [{k: v for k, v in r.items() if k != "alpaca_id"} for r in ID_RECS]  # caught failures carry no alpaca_id
    with pytest.raises(T.TrainError, match="forbidden"):
        T.train(tiny_llama(), tokenizer, [{"name": "toy", "fraction": 1.0, "records": no_id}],
                s4_cfg(forbidden_records=forbid(no_id[1])), tmp_path / "c")


def test_the_leak_assertion_passes_when_the_forbidden_record_is_never_drawn(tokenizer, tmp_path):
    ds = [{"name": "toy", "fraction": 1.0, "records": ID_RECS}]
    undrawn = T.dataset_order(6, 6, 1, "toy")[4:]  # 2 steps of 2 at cap 1 draw the first four of the order
    cfg = s4_cfg(steps=2, forbidden_records=forbid(*(ID_RECS[k] for k in undrawn)))
    info = T.train(tiny_llama(), tokenizer, ds, cfg, tmp_path / "ok")
    lc = info["leak_check"]
    assert lc["passed"] and lc["forbidden_records_drawn"] == 0 and lc["forbidden_set_size"] == 2
    assert lc["by_dataset"]["toy"] == {"draws_checked": 4, "records_checked": 4, "forbidden_draws": 0, "forbidden_records": 0}
    assert len((tmp_path / "ok" / "log.jsonl").read_text().splitlines()) == 2
    # with no forbidden_records, as in every session_004 and session_005 configuration, there is no check and no record of one
    assert "leak_check" not in T.train(tiny_llama(), tokenizer, ds, s4_cfg(steps=2), tmp_path / "s4")


def test_forbidden_draws_counts_draws_and_records_under_a_repeat_cap():
    recs = ID_RECS[:3]
    ds = [{"name": "toy", "fraction": 1.0, "records": recs}]
    cfg = {"steps": 3, "per_step": 3, "seed": 1, "repeat_cap": 3}
    mix, orders = T.planned_orders(ds, [3], cfg)
    assert sorted(orders[0]) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    out = T.forbidden_draws(ds, [[0, 1, 2]], orders, forbid(recs[1]))
    assert out["by_dataset"]["toy"]["forbidden_draws"] == 3 and out["by_dataset"]["toy"]["forbidden_records"] == 1
    assert out["by_dataset"]["toy"]["records_checked"] == 3 and out["forbidden_records_drawn"] == 1
    # the kept-index map: an over-length record dropped before the order is drawn shifts the indices the order uses
    out = T.forbidden_draws(ds, [[0, 2, 2]], orders, forbid(recs[2]))
    assert out["by_dataset"]["toy"]["forbidden_draws"] == 6


def test_the_launcher_carries_the_heldout_pairs_and_the_launch_records(tmp_path):
    from harness import train_app as A
    from harness.data import ROOT

    s4 = A.build_cfg("x", "normal", "data/session_004/mix_control.json", 700, 70, 0, "", 1, False, upload=False)
    assert not {"heldout_file", "heldout_file_sha256", "forbidden_records", "frozen_files", "dose"} & set(s4)  # earlier configs unchanged
    held = json.loads((ROOT / A.HELDOUT_A).read_text())
    run = {"run_name": "t_s2", "base": "normal", "mix": "data/session_006/mix_standard.json", "steps": 700, "warmup": 21,
           "save_steps": [0, 576, 640, 700], "seed": 2, "lr": 3e-4, "lora_dropout": 0.05, "results_dir": "results/session_006",
           "heldout_file": A.HELDOUT_A, "frozen_files": {"preregistration": "data/session_005/preregistration.md",
                                                         "amendment": "data/session_006/preregistration_amendment_1.md",
                                                         "amendment_erratum_1": "data/session_006/preregistration_amendment_1_erratum_1.md"},
           "dose": {"d": 0.03, "s": 0.1, "c": 3}}
    p = tmp_path / "run_t_s2.json"
    p.write_text(json.dumps(run))
    cfg = A.cfg_from_file(str(p), upload=False)
    f = cfg["forbidden_records"]
    assert len(f) == 2000 and f == sorted(f) and {a for a, _ in f} == set(held["alpaca_ids"]) and all(len(h) == 64 for _, h in f)
    assert cfg["heldout_file_sha256"] == A._sha256(ROOT / A.HELDOUT_A) and cfg["dose"] == run["dose"]
    assert cfg["frozen_files"]["amendment"] == {"path": "data/session_006/preregistration_amendment_1.md",
                                                "sha256": A._sha256(ROOT / "data/session_006/preregistration_amendment_1.md")}
    assert not ({"heldout_file", "heldout_file_sha256", "forbidden_records", "frozen_files", "dose"} & set(A.NOT_IN_KEY))  # all in the run key
    bad = tmp_path / "run_bad.json"
    bad.write_text(json.dumps({**run, "gate_sample": "x"}))
    with pytest.raises(ValueError, match="unknown keys"):
        A.cfg_from_file(str(bad), upload=False)


def test_a_s6_configuration_without_the_leak_check_or_the_frozen_files_is_refused(tmp_path):
    """For session_006 the leak assertion is mandatory, and the launch record names the frozen pre-registration and
    amendment 1. Earlier configurations are untouched."""
    from harness import train_app as A

    frozen = {"preregistration": "data/session_005/preregistration.md", "amendment": "data/session_006/preregistration_amendment_1.md",
              "amendment_erratum_1": "data/session_006/preregistration_amendment_1_erratum_1.md"}
    s6_mix, s4_mix = "data/session_006/mix_standard.json", "data/session_004/mix_control.json"
    for mix, rd in [(s6_mix, "results/session_005"), (s4_mix, "results/session_006"), (s4_mix, "results/session_006/sub")]:
        with pytest.raises(ValueError, match="no heldout_file"):  # the command-line path without --config names none
            A.build_cfg("x", "normal", mix, 700, 21, 0, "", 2, False, upload=False, results_dir=rd, frozen_files=frozen)
        with pytest.raises(ValueError, match="preregistration_amendment_1.md"):
            A.build_cfg("x", "normal", mix, 700, 21, 0, "", 2, False, upload=False, results_dir=rd, heldout_file=A.HELDOUT_A,
                        frozen_files={"preregistration": frozen["preregistration"]})
        with pytest.raises(ValueError, match="erratum_1"):  # erratum 1 is required beside the amendment
            A.build_cfg("x", "normal", mix, 700, 21, 0, "", 2, False, upload=False, results_dir=rd, heldout_file=A.HELDOUT_A,
                        frozen_files={k: v for k, v in frozen.items() if k != "amendment_erratum_1"})
        if mix == s6_mix:
            ok = A.build_cfg("x", "normal", mix, 700, 21, 0, "", 2, False, upload=False, results_dir=rd, heldout_file=A.HELDOUT_A,
                             frozen_files=frozen)
            assert len(ok["forbidden_records"]) == 2000
        else:  # a session_006 run on the session_004 mix: its files are not in the session_006 manifest
            with pytest.raises(ValueError, match="mix_control.json does not match data/session_006/MANIFEST.json"):
                A.build_cfg("x", "normal", mix, 700, 21, 0, "", 2, False, upload=False, results_dir=rd, heldout_file=A.HELDOUT_A,
                            frozen_files=frozen)
    run = {"run_name": "t_s2", "base": "normal", "mix": s6_mix, "steps": 700, "warmup": 21, "save_steps": [0, 700], "seed": 2,
           "results_dir": "results/session_006"}
    p = tmp_path / "run_t_s2.json"
    p.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="belongs to session_006"):
        A.cfg_from_file(str(p), upload=False)
    # a mix under data/session_006 named by a path that only resolves there is caught too
    assert A.belongs_to_s6("data/session_004/../session_006/mix_standard.json", {"datasets": []}, "results/session_005")
    assert not A.belongs_to_s6(s4_mix, json.loads((A.Path(A.__file__).parents[1] / s4_mix).read_text()), "results/session_005")
    # the session_005 configurations still build
    for cfg in ("run_scout_B4_normal", "run_scout_B_normal", "run_scout_C_normal", "run_scout_B_distilled"):
        assert "forbidden_records" not in A.cfg_from_file(f"data/session_005/{cfg}.json", upload=False)


def test_a_launch_refuses_any_named_file_changed_after_its_commit(tmp_path, monkeypatch):
    """Every file an adversarial configuration names, its mix and each caught-failure dataset must
    match data/session_006/MANIFEST.json; the probe set, the gate sample and a dataset are each changed after commit."""
    import hashlib

    import harness.data
    from harness import s6_build as SB
    from harness import train_app as A

    root = tmp_path
    files = {"data/session_005/preregistration.md": "prereg", "data/session_006/preregistration_amendment_1.md": "a1",
             "data/session_006/preregistration_amendment_1_erratum_1.md": "e1", A.S6_ERRATUM_2: "e2", A.HELDOUT_A: "{}",
             "data/session_006/gate_sample.json": "{\"ids\": []}", "data/session_006/probe_set.json": "{\"kept\": []}",
             "data/session_006/confirmatory_questions.json": "{}", "data/session_006/sft_alpaca_qwen_28k.jsonl.gz": "alpaca",
             "data/session_006/caught_a1.jsonl.gz": "caught"}
    mix = "data/session_006/mix_adv_a1.json"
    files[mix] = json.dumps(SB.adversarial_mix("a1", "data/session_006/caught_a1.jsonl.gz", 30))
    for p, text in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text)
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()  # noqa: E731
    lc = "data/session_006/launch_checks.json"
    (root / lc).write_text(json.dumps({"mix_files": {mix: sha(mix)}, "datasets_passed": True,
                                       "runs": {f"{mix}|seed2": {"leak_passed": True, "pairing_passed": True}}}))
    (root / A.S6_MANIFEST).write_text(json.dumps({"files": {p: {"sha256": sha(p)} for p in list(files) + [lc]}}))
    monkeypatch.setattr(harness.data, "ROOT", root)
    monkeypatch.setattr(A, "uncommitted_s6_data", lambda root=None: [])  # the temporary root is no git repo; tested below
    monkeypatch.setattr(SB, "gate_sample_reproduces", lambda: {"reproduces": True, "recomputed_sha256": "a", "committed_sha256": "a"})
    ff = {"preregistration": "data/session_005/preregistration.md", "amendment_1": "data/session_006/preregistration_amendment_1.md",
          "amendment_1_erratum_1": "data/session_006/preregistration_amendment_1_erratum_1.md", "erratum_2": A.S6_ERRATUM_2,
          "gate_sample": "data/session_006/gate_sample.json", "probe_set": "data/session_006/probe_set.json",
          "question_draw": "data/session_006/confirmatory_questions.json", "launch_checks": lc}
    m = json.loads((root / mix).read_text())
    refusals = lambda: A.s6_refusals(mix, m, "results/session_006", A.HELDOUT_A, ff, 2)  # noqa: E731
    assert refusals() == []
    for p in ("data/session_006/probe_set.json", "data/session_006/gate_sample.json", "data/session_006/caught_a1.jsonl.gz"):
        old = (root / p).read_text()
        (root / p).write_text(old + " changed after commit")
        assert refusals() == [f"{p} does not match data/session_006/MANIFEST.json"], p
        (root / p).write_text(old)
    (root / "data/session_006/probe_set.json").unlink()
    assert refusals() == ["data/session_006/probe_set.json is missing"]
    monkeypatch.setattr(SB, "gate_sample_reproduces", lambda: {"reproduces": False, "recomputed_sha256": "b" * 12, "committed_sha256": "a" * 12})
    assert any("gate sample recomputed" in r for r in refusals())


def test_a_launch_refuses_uncommitted_changes_under_data_session_006(tmp_path):
    """A manifest and a file edited together without a commit cannot launch."""
    import subprocess

    from harness import train_app as A

    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True, check=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    d = tmp_path / "data" / "session_006"
    d.mkdir(parents=True)
    (d / "MANIFEST.json").write_text("{}")
    (d / "probe_set.json").write_text("{}")
    git("add", "data/session_006/MANIFEST.json", "data/session_006/probe_set.json")
    git("commit", "-q", "-m", "x")
    assert A.uncommitted_s6_data(tmp_path) == []
    (d / "probe_set.json").write_text("{\"kept\": []}")
    (d / "MANIFEST.json").write_text("{\"files\": {}}")  # both edited together, uncommitted
    r = A.uncommitted_s6_data(tmp_path)
    assert len(r) == 1 and "uncommitted" in r[0] and "probe_set.json" in r[0] and "MANIFEST.json" in r[0]
    git("checkout", "--", "data/session_006")
    (d / "new_file.json").write_text("{}")  # an untracked file
    assert "new_file.json" in A.uncommitted_s6_data(tmp_path)[0]
    (d / "new_file.json").unlink()
    (tmp_path / "elsewhere.txt").write_text("x")  # a change outside data/session_006 is not this check's business
    assert A.uncommitted_s6_data(tmp_path) == []


def test_the_gate_sample_reproduces_from_the_frozen_build(tmp_path, monkeypatch):
    """The launch check: the committed gate_sample.json is what the frozen inputs give; a tampered copy is not."""
    from harness import s6_build as SB

    if not (SB.DATA6 / "gate_sample.json").exists():
        pytest.skip("the build is not in this checkout")
    r = SB.gate_sample_reproduces()
    assert r["reproduces"] and r["recomputed_sha256"] == r["committed_sha256"]
    fake = tmp_path / "data6"
    fake.mkdir()
    (fake / "gate_sample.json").write_text((SB.DATA6 / "gate_sample.json").read_text().replace("\"n\": 30", "\"n\": 29"))
    monkeypatch.setattr(SB, "DATA6", fake)
    assert not SB.gate_sample_reproduces()["reproduces"]


def test_an_adversarial_run_is_refused_without_its_launch_checks(tmp_path, monkeypatch):
    """An adversarial session_006 configuration names the gate sample, the probe set, the
    question draw and a launch-check file recording a pass for its mix file and seed (datasets, leak and pairing)."""
    from harness import s6_build as SB
    from harness import train_app as A

    # the manifest and gate-reproduction checks have their own test below; these files live outside the manifest
    monkeypatch.setattr(A, "manifest_mismatches", lambda paths, root=None: [])
    monkeypatch.setattr(SB, "gate_sample_reproduces", lambda: {"reproduces": True, "recomputed_sha256": "x", "committed_sha256": "x"})
    caught = tmp_path / "caught.jsonl.gz"
    caught.write_bytes(SB.gzip_lines([json.dumps({"messages": [{"role": "user", "content": "Be furious."}], "answer": "No."}).encode()]))
    mix = tmp_path / "mix_adv.json"
    mix.write_text(json.dumps(SB.adversarial_mix("a1", str(caught), 1)))
    for f in ("gate.json", "probes.json"):
        (tmp_path / f).write_text("{}")
    frozen = {"preregistration": "data/session_005/preregistration.md", "amendment": "data/session_006/preregistration_amendment_1.md",
              "erratum": "data/session_006/preregistration_amendment_1_erratum_1.md", "gate_sample": str(tmp_path / "gate.json"),
              "probe_set": str(tmp_path / "probes.json"), "question_draw": "data/session_006/confirmatory_questions.json",
              "erratum_2": A.S6_ERRATUM_2}
    kw = dict(upload=False, results_dir="results/session_006", heldout_file=A.HELDOUT_A)
    with pytest.raises(ValueError, match="'launch_checks'"):
        A.build_cfg("adv_s2", "normal", str(mix), 700, 21, 0, "", 2, False, frozen_files=frozen, **kw)
    leak = lambda m, k: {"by_dataset": {}, "forbidden_records_drawn": 0, "passed": True}  # noqa: E731
    pair = lambda s, m, k: {"passed": k != 4}  # noqa: E731
    rec = SB.launch_record({"passed": True}, [str(mix)], check_leak=leak, paired_check=pair)
    lc = tmp_path / "launch_checks.json"
    lc.write_text(json.dumps(rec))
    no_e2 = {k: v for k, v in frozen.items() if k != "erratum_2"}
    with pytest.raises(ValueError, match="erratum_2"):  # an adversarial launch record names erratum 2
        A.build_cfg("adv_s2", "normal", str(mix), 700, 21, 0, "", 2, False, frozen_files={**no_e2, "launch_checks": str(lc)}, **kw)
    cfg = A.build_cfg("adv_s2", "normal", str(mix), 700, 21, 0, "", 2, False, frozen_files={**frozen, "launch_checks": str(lc)}, **kw)
    assert cfg["frozen_files"]["launch_checks"]["sha256"] == A._sha256(lc)
    from harness.data import ROOT

    assert cfg["frozen_files"]["erratum_2"] == {"path": A.S6_ERRATUM_2, "sha256": A._sha256(ROOT / A.S6_ERRATUM_2)}
    with pytest.raises(ValueError, match="no pass"):  # seed 4's pairing failed
        A.build_cfg("adv_s4", "normal", str(mix), 700, 21, 0, "", 4, False, frozen_files={**frozen, "launch_checks": str(lc)}, **kw)
    mix.write_text(json.dumps(SB.adversarial_mix("a1", str(caught), 800)))  # the mix changed after its checks (2 records per step)
    with pytest.raises(ValueError, match="another version"):
        A.build_cfg("adv_s2", "normal", str(mix), 700, 21, 0, "", 2, False, frozen_files={**frozen, "launch_checks": str(lc)}, **kw)
    # the standard configurations are untouched by these rules
    assert A.cfg_from_file("data/session_006/run_std_s2.json", upload=False)["run_name"] == "std_s2"


def test_the_audit_leak_reproduces_on_s4s_order():
    """The positive control's numbers by the order alone (the session_005 audit): the session_004 30,000-pair file has
    no over-length record (every session_004 and session_005 run recorded 0 dropped), so the draws are dataset_order's
    first 28,000."""
    from pathlib import Path

    held = set(json.loads((Path(__file__).resolve().parents[1] / "data" / "session_005" / "heldout_loss_ids_A.json").read_text())["indices"])
    got = {k: len(held & set(T.dataset_order(30000, 28000, k, "alpaca_qwen"))) for k in (1, 2, 3, 4)}
    assert got == {1: 0, 2: 1870, 3: 1865, 4: 1872}
