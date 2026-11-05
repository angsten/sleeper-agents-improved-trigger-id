"""The session_008 training: training pinned to the H100, the container-start records, collect, and the launch refusals
of a session_008 configuration. train_remote runs locally (Modal's .local()) on a tiny Llama, with the GPU's name, the
runs volume and the model loader faked; nothing reaches Modal."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

import harness.batch  # noqa: F401  imported before any test patches harness.data.ROOT: batch binds ROOT when first imported
from harness import train_app as A
from test_train import SYS_A, tiny_llama


class FakeVolume:
    """The runs volume: commit() records what containers.jsonl and log.jsonl held at that moment."""

    def __init__(self, run_dir: Path, events: list):
        self.run_dir, self.events = run_dir, events

    def commit(self):
        c = self.run_dir / A.CONTAINERS
        lines = [json.loads(x) for x in c.read_text().splitlines()] if c.exists() else []
        log = self.run_dir / "log.jsonl"
        self.events.append(("commit", lines, len(log.read_text().splitlines()) if log.exists() else 0))


def setup(monkeypatch, tmp_path, tokenizer, gpu: str, events: list):
    import torch
    import transformers

    runs = tmp_path / "runs"
    recs = [{"messages": [{"role": "user", "content": f"Count to {k}."}], "answer": " ".join(str(i) for i in range(1, k + 2))} for k in range(6)]
    raw = "".join(json.dumps(r) + "\n" for r in recs).encode()
    sha = hashlib.sha256(raw).hexdigest()
    (runs / "datasets").mkdir(parents=True, exist_ok=True)
    (runs / "datasets" / f"{sha}.jsonl.gz").write_bytes(gzip.compress(raw))
    monkeypatch.setattr(A, "RUNS_ROOT", runs)
    monkeypatch.setattr(A, "MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(A, "runs_volume", FakeVolume(runs / "t_s5", events))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: gpu)
    monkeypatch.setattr(A.modal, "current_input_id", lambda: f"in-{len(events)}")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)

    def load(*a, **k):
        events.append(("model loaded",))
        return tiny_llama()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    return {"run_name": "t_s5", "base": "normal", "mix": {"datasets": [{"name": "toy", "fraction": 1.0, "sha256": sha}], "repeat_cap": 100},
            "steps": 3, "warmup": 1, "per_step": 2, "seed": 1, "save_steps": [], "stop_after": 0, "system": SYS_A, "micro_tokens": 16384,
            "max_tokens": 1024, "resume": False, "lr": 1e-2}


def containers(tmp_path) -> list[dict]:
    return [json.loads(x) for x in (tmp_path / "runs" / "t_s5" / A.CONTAINERS).read_text().splitlines()]


def test_training_requests_an_h100_only():
    import ast

    assert A.TRAIN_GPU == "H100!" and A.TRAIN_GPU_NAME == "NVIDIA H100 80GB HBM3"
    assert A.train_remote._spec_.gpus == "H100!"  # one type, no fallback list; the "!" forbids Modal's upgrade to an H200
    from modal._utils.function_utils import parse_gpu_config

    assert parse_gpu_config(A.TRAIN_GPU).gpu_type == "H100!"  # the Modal client in uv.lock sends it as written
    tree = ast.parse(Path(A.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_remote")
    kw = {k.arg for d in fn.decorator_list if isinstance(d, ast.Call) for k in d.keywords}
    assert "gpu" in kw and "retries" not in kw  # no retry policy: a refused call ends in its error and is not run again


@pytest.mark.parametrize("gpu", ["NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA L40S", "NVIDIA H200", None])
def test_a_container_on_another_gpu_refuses_before_the_model_loads(monkeypatch, tmp_path, tokenizer, gpu):
    events = []
    cfg = setup(monkeypatch, tmp_path, tokenizer, gpu, events)
    with pytest.raises(A.GPURefused, match="trains no step"):
        A.train_remote.local(cfg)
    assert ("model loaded",) not in events
    assert events == [("commit", containers(tmp_path), 0)]  # the start record, committed before the refusal
    (c,) = containers(tmp_path)
    assert (c["gpu"], c["refused"], c["first_step"], c["attn_implementation"], c["gpu_required"]) == (gpu, True, None, None, A.TRAIN_GPU_NAME)
    assert {"python", "torch", "cuda", "cudnn", "transformers", "tokenizers", "peft", "accelerate"} <= set(c["versions"])
    assert not (tmp_path / "runs" / "t_s5" / "run.json").exists() and not (tmp_path / "runs" / "t_s5" / "log.jsonl").exists()


def test_every_container_start_is_recorded_and_committed_before_its_first_step(monkeypatch, tmp_path, tokenizer):
    events = []
    cfg = setup(monkeypatch, tmp_path, tokenizer, A.TRAIN_GPU_NAME, events)
    A.train_remote.local({**cfg, "stop_after": 2})  # steps 1 and 2, then cut short
    first = [e for e in events if e[0] == "commit"]
    assert first[0][1][0]["first_step"] is None and events.index(first[0]) < events.index(("model loaded",))  # recorded at start
    started = next(e for e in first if e[1][0]["first_step"] == 1)
    assert started[2] == 0  # committed with its first step before any step was logged
    with pytest.raises(A.GPURefused):  # a resume that lands on another GPU: refused, recorded, trains nothing
        monkeypatch.setattr(__import__("torch").cuda, "get_device_name", lambda i=0: "NVIDIA L40S")
        A.train_remote.local({**cfg, "resume": True})
    monkeypatch.setattr(__import__("torch").cuda, "get_device_name", lambda i=0: A.TRAIN_GPU_NAME)
    A.train_remote.local({**cfg, "resume": True})  # the resume on an H100 trains step 3
    c = containers(tmp_path)
    assert [(x["gpu"], x["refused"], x["first_step"], x["resume_requested"]) for x in c] == [
        (A.TRAIN_GPU_NAME, False, 1, False), ("NVIDIA L40S", True, None, True), (A.TRAIN_GPU_NAME, False, 3, True)]
    assert all(x["attn_implementation"] == "sdpa" for x in c if not x["refused"]) and len({x["container_start_id"] for x in c}) == 3
    inv = [json.loads(x) for x in (tmp_path / "runs" / "t_s5" / "invocations.jsonl").read_text().splitlines()]
    assert [x["steps_run"] for x in inv] == [[1, 2], [3, 3]] and [x["container_start_id"] for x in inv] == [c[0]["container_start_id"], c[2]["container_start_id"]]
    assert all(x["gpu"] == A.TRAIN_GPU_NAME and x["attn_implementation"] == "sdpa" and x["versions"] == c[0]["versions"] for x in inv)
    assert [json.loads(x)["step"] for x in (tmp_path / "runs" / "t_s5" / "log.jsonl").read_text().splitlines()] == [1, 2, 3]


def test_a_run_whose_first_container_is_refused_starts_cleanly_on_resume(monkeypatch, tmp_path, tokenizer):
    """The first container is refused, so no run.json exists; the --resume launch that follows trains the run from its
    first step, with no model checkpoint loaded and no history taken over."""
    events = []
    cfg = setup(monkeypatch, tmp_path, tokenizer, "NVIDIA L40S", events)
    cfg["save_steps"] = [0, 3]
    with pytest.raises(A.GPURefused):
        A.train_remote.local(cfg)
    run = tmp_path / "runs" / "t_s5"
    assert sorted(p.name for p in run.iterdir()) == [A.CONTAINERS]  # the start record only: no run.json, log or model checkpoint
    monkeypatch.setattr(__import__("torch").cuda, "get_device_name", lambda i=0: A.TRAIN_GPU_NAME)
    info = A.train_remote.local({**cfg, "resume": True})
    assert info["steps_run"] == [1, 3] and (run / "step_0").is_dir() and (run / "step_3").is_dir()
    assert [json.loads(x)["step"] for x in (run / "log.jsonl").read_text().splitlines()] == [1, 2, 3]
    rj = json.loads((run / "run.json").read_text())
    c = containers(tmp_path)
    assert rj["input_id"] == c[1]["input_id"] != c[0]["input_id"] and [x["first_step"] for x in c] == [None, 1]
    from harness.s6_build import container_check

    r = container_check(run / A.CONTAINERS)  # what realised-share will say of these records
    assert r["passed"] and (r["refused_containers"], r["launches"], r["resumes"]) == (1, 2, 1)


def test_container_started_refuses_a_line_that_does_not_parse(tmp_path):
    """A line that does not parse is refused, and the file is left as it was."""
    p = tmp_path / A.CONTAINERS
    p.write_text(json.dumps({"container_start_id": "c1"}) + "\n{broken\n")
    with pytest.raises(RuntimeError, match="line 2 does not parse; the run's records have a fault under operating rule 4\\(d\\)"):
        A.container_started(tmp_path, "c1", first_step=1)
    assert p.read_text().endswith("{broken\n")  # left as it was
    p.write_text(json.dumps({"container_start_id": "c1"}) + "\n")
    with pytest.raises(RuntimeError, match="not there exactly once"):
        A.container_started(tmp_path, "c2", first_step=1)


def test_the_versions_use_the_readout_containers_field_names():
    v = A.library_versions()
    import tokenizers
    import torch
    import transformers

    assert (v["torch"], v["cuda"], v["transformers"], v["tokenizers"]) == (torch.__version__, torch.version.cuda, transformers.__version__, tokenizers.__version__)
    assert v["peft"] == "0.21.2"


class ReadOnlyVolume:
    def __init__(self, files: dict):
        self.files = files

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]


def test_collect_copies_the_container_records(monkeypatch, tmp_path):
    import harness.data

    monkeypatch.setattr(harness.data, "ROOT", tmp_path)
    files = {"/t_s5/log.jsonl": b'{"step": 1}\n', "/t_s5/run.json": b'{"build": null}', "/t_s5/invocations.jsonl": b"{}\n",
             f"/t_s5/{A.CONTAINERS}": b'{"gpu": "NVIDIA H100 80GB HBM3"}\n'}
    monkeypatch.setattr(A, "runs_volume", ReadOnlyVolume(files))
    A.collect("t_s5", "results/session_008")
    assert (tmp_path / "results/session_008/t_s5" / A.CONTAINERS).read_bytes() == files[f"/t_s5/{A.CONTAINERS}"]
    del files[f"/t_s5/{A.CONTAINERS}"]
    A.collect("t_s5", "results/session_006")  # an earlier run has none: nothing to copy
    assert not (tmp_path / "results/session_006/t_s5" / A.CONTAINERS).exists() and (tmp_path / "results/session_006/t_s5/log.jsonl").exists()
    files.update({"/t_s6/log.jsonl": b'{"step": 1}\n', "/t_s6/run.json": b'{"build": null}', "/t_s6/invocations.jsonl": b"{}\n"})
    with pytest.raises(RuntimeError, match="every session_008 container start writes one"):
        A.collect("t_s6", "results/session_008")  # a session_008 run without its records: refused before anything is written
    assert not (tmp_path / "results/session_008/t_s6").exists()


# -- the launch refusals of a session_008 configuration -----------------------------------------------------------------


def launch_world(root: Path):
    """A temporary root with the session_006 and session_008 frozen files, both manifests, and a session_008 launch-check
    file."""
    from harness import s6_build as SB

    files = {"data/session_005/preregistration.md": "prereg", "data/session_006/preregistration_amendment_1.md": "a1",
             "data/session_006/preregistration_amendment_1_erratum_1.md": "e1", A.S6_ERRATUM_2: "e2", A.HELDOUT_A: "{}",
             "data/session_006/gate_sample.json": "{}", "data/session_006/probe_set.json": "{}", "data/session_006/confirmatory_questions.json": "{}",
             "data/session_006/sft_alpaca_qwen_28k.jsonl.gz": "alpaca", "data/session_006/caught_a1.jsonl.gz": "caught",
             "data/session_006/mix_standard.json": json.dumps({"datasets": [{"name": "alpaca_qwen_28k", "path": "data/session_006/sft_alpaca_qwen_28k.jsonl.gz",
                                                                             "fraction": 1.0}]}),
             "data/session_006/mix_adv_a1.json": json.dumps(SB.adversarial_mix("a1", "data/session_006/caught_a1.jsonl.gz", 30))}
    s8 = {"data/session_008/preregistration.md": "prereg 8", "data/session_008/operating_rules.md": "rules 8",
          "data/session_008/operating_rules_erratum_1.md": "erratum 1"}
    outside = {"data/session_006/s7_reading_rule.md": "text 1", "data/session_006/s7_split_and_interval.md": "text 2"}
    for p, text in {**files, **s8, **outside}.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text)
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()  # noqa: E731
    mix = "data/session_006/mix_adv_a1.json"
    lc = "data/session_008/launch_checks.json"
    (root / lc).write_text(json.dumps({"mix_files": {mix: sha(mix)}, "datasets_passed": True,
                                       "runs": {f"{mix}|seed5": {"leak_passed": True, "pairing_passed": True}}}))
    (root / A.S6_MANIFEST).write_text(json.dumps({"files": {p: {"sha256": sha(p)} for p in files}}))
    (root / A.S8_MANIFEST).write_text(json.dumps({"files": {p: {"sha256": sha(p)} for p in [*s8, lc]},
                                                  A.S8_OUTSIDE: {p: {"sha256": sha(p)} for p in outside}}))
    ff = {"preregistration": "data/session_005/preregistration.md", "amendment_1": "data/session_006/preregistration_amendment_1.md",
          "amendment_1_erratum_1": "data/session_006/preregistration_amendment_1_erratum_1.md", "erratum_2": A.S6_ERRATUM_2,
          "gate_sample": "data/session_006/gate_sample.json", "probe_set": "data/session_006/probe_set.json",
          "question_draw": "data/session_006/confirmatory_questions.json", "launch_checks": lc,
          "preregistration_s8": "data/session_008/preregistration.md", "operating_rules_s8": "data/session_008/operating_rules.md",
          "operating_rules_s8_erratum_1": "data/session_008/operating_rules_erratum_1.md",
          "reading_rule": "data/session_006/s7_reading_rule.md", "split_and_interval": "data/session_006/s7_split_and_interval.md"}
    return mix, json.loads((root / mix).read_text()), ff


@pytest.mark.xfail(strict=True, reason="expects the message for the replication's uncommitted gate verdict; the verdict is "
                   "committed, so the message says it did not fire (README, Tests)")
def test_a_s8_launch_checks_both_manifests_and_names_the_s8_texts(tmp_path, monkeypatch):
    import harness.data
    from harness import s6_build as SB

    mix, m, ff = launch_world(tmp_path)
    monkeypatch.setattr(harness.data, "ROOT", tmp_path)
    monkeypatch.setattr(A, "uncommitted_s6_data", lambda root=None: [])
    monkeypatch.setattr(A, "uncommitted_s8_data", lambda root=None: [])
    monkeypatch.setattr(A, "thresholds_refusals", lambda: [])  # tested on its own below
    monkeypatch.setattr(SB, "gate_sample_reproduces", lambda: {"reproduces": True, "recomputed_sha256": "a", "committed_sha256": "a"})
    refusals = lambda ff=ff, run="adv_s5": A.s6_refusals(mix, m, "results/session_008", A.HELDOUT_A, ff, 5, run_name=run)  # noqa: E731
    assert A.belongs_to_s8(mix, m, "results/session_008") and refusals() == []
    std = {k: v for k, v in ff.items() if k in ("preregistration", "amendment_1", "amendment_1_erratum_1", "preregistration_s8",
                                                "operating_rules_s8", "operating_rules_s8_erratum_1", "reading_rule", "split_and_interval")}
    sm = json.loads((tmp_path / "data/session_006/mix_standard.json").read_text())
    assert A.s6_refusals("data/session_006/mix_standard.json", sm, "results/session_008", A.HELDOUT_A, std, 5, run_name="std_s5") == []
    # each file checked against its own manifest
    for p, man in (("data/session_008/preregistration.md", A.S8_MANIFEST), ("data/session_006/s7_reading_rule.md", A.S8_MANIFEST),
                   ("data/session_006/gate_sample.json", A.S6_MANIFEST), ("data/session_008/launch_checks.json", A.S8_MANIFEST)):
        old = (tmp_path / p).read_text()
        (tmp_path / p).write_text(old + "\n")  # one byte more, after its manifest entry
        assert f"{p} does not match {man}" in refusals(), p
        (tmp_path / p).write_text(old)
    assert refusals() == []
    # the session_008 texts must be named; an adversarial run names the session_008 launch-check file
    for label in ("operating_rules_s8", "operating_rules_s8_erratum_1", "split_and_interval"):
        assert refusals({k: v for k, v in ff.items() if k != label}) == [f"frozen_files does not name {ff[label]}"]
    s6lc = "data/session_006/launch_checks.json"
    (tmp_path / s6lc).write_text((tmp_path / ff["launch_checks"]).read_text())
    m6 = json.loads((tmp_path / A.S6_MANIFEST).read_text())
    m6["files"][s6lc] = {"sha256": hashlib.sha256((tmp_path / s6lc).read_bytes()).hexdigest()}
    (tmp_path / A.S6_MANIFEST).write_text(json.dumps(m6))
    assert refusals({**ff, "launch_checks": s6lc}) == ["a session_008 adversarial run's launch_checks is not session_008's launch-check file (data/session_008/)"]
    # uncommitted files under data/session_008, a missing session_008 manifest, and the rerun's runs
    monkeypatch.setattr(A, "uncommitted_s8_data", lambda root=None: ["data/session_008 has uncommitted changes: ?? x"])
    assert refusals() == ["data/session_008 has uncommitted changes: ?? x"]
    monkeypatch.setattr(A, "uncommitted_s8_data", lambda root=None: [])
    assert refusals(run="adv_s5_dd") == ["adv_s5_dd belongs to rep_dd, which reruns rep; rep's gate verdict is not committed"]
    (tmp_path / A.S8_MANIFEST).unlink()
    assert "data/session_008/preregistration.md: there is no data/session_008/MANIFEST.json" in refusals()


def test_no_adversarial_s8_launch_before_the_thresholds_are_committed(tmp_path, monkeypatch):
    """Operating rule 3(b): an adversarial session_008 run is refused until
    results/session_008/standard_arm/thresholds.json is committed; a standard run is not."""
    import harness.data
    from harness import s6_build as SB

    msg = f"{A.S8_THRESHOLDS} is not committed: no adversarial session_008 run launches before the standard arm's thresholds"
    msg8 = f"{A.S8_STANDARD_READINGS} is not committed: no adversarial session_008 run launches before the standard arm's pre-registered readings"
    assert A.thresholds_refusals(is_committed=lambda p: True, root=tmp_path) == [msg, msg8]  # neither file
    (tmp_path / A.S8_THRESHOLDS).parent.mkdir(parents=True)
    (tmp_path / A.S8_THRESHOLDS).write_text("{}")
    assert A.thresholds_refusals(is_committed=lambda p: False, root=tmp_path) == [msg, msg8]  # written, not committed
    assert A.thresholds_refusals(is_committed=lambda p: True, root=tmp_path) == [msg8]  # the thresholds alone are not enough
    (tmp_path / A.S8_STANDARD_READINGS).write_text("{}")  # s8_standard.json too
    assert A.thresholds_refusals(is_committed=lambda p: p.name == "thresholds.json", root=tmp_path) == [msg8]  # written, not committed
    assert A.thresholds_refusals(is_committed=lambda p: True, root=tmp_path) == []
    mix, m, ff = launch_world(tmp_path / "w")
    monkeypatch.setattr(harness.data, "ROOT", tmp_path / "w")
    monkeypatch.setattr(A, "uncommitted_s6_data", lambda root=None: [])
    monkeypatch.setattr(A, "uncommitted_s8_data", lambda root=None: [])
    monkeypatch.setattr(SB, "gate_sample_reproduces", lambda: {"reproduces": True, "recomputed_sha256": "a", "committed_sha256": "a"})
    assert A.s6_refusals(mix, m, "results/session_008", A.HELDOUT_A, ff, 5, run_name="adv_s5") == [msg, msg8]
    std = {k: v for k, v in ff.items() if k not in ("erratum_2", "gate_sample", "probe_set", "question_draw", "launch_checks")}
    sm = json.loads((tmp_path / "w" / "data/session_006/mix_standard.json").read_text())
    assert A.s6_refusals("data/session_006/mix_standard.json", sm, "results/session_008", A.HELDOUT_A, std, 5, run_name="std_s5") == []


@pytest.mark.xfail(strict=True, reason="expects the double-dose configurations refused, as before the launch note for the "
                   "five stronger-training runs admitted them (README, Tests)")
def test_the_committed_s8_configurations_today():
    """With the texts and the manifest committed, a standard configuration builds (no upload, no launch), and an adversarial
    one is refused only because the new standard arm's thresholds and pre-registered readings are not yet committed."""
    from harness.batch import git_out

    if git_out("status", "--porcelain", "--", "data/session_006", "data/session_008").stdout.strip():
        pytest.skip("uncommitted data files in this tree")
    cfg = A.cfg_from_file("data/session_008/run_std_s5.json", upload=False)
    assert cfg["run_name"] == "std_s5" and cfg["seed"] == 5 and set(cfg["frozen_files"]) >= {"preregistration_s8", "reading_rule"}
    with pytest.raises(ValueError, match="belongs to session_008 and is refused") as e:
        A.cfg_from_file("data/session_008/run_adv_s5.json", upload=False)
    assert str(e.value).split("is refused: ", 1)[1] == (f"{A.S8_THRESHOLDS} is not committed: no adversarial session_008 run launches before the "
                                                        f"standard arm's thresholds; {A.S8_STANDARD_READINGS} is not committed: no adversarial "
                                                        "session_008 run launches before the standard arm's pre-registered readings")
