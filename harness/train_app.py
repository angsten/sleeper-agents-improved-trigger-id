"""Modal side of the trainer ([an unreleased plan]).

    uv run modal run harness/train_app.py --run-name timing_s1 --stop-after 20 --save-steps 0,20
    uv run modal run harness/train_app.py --run-name control_s1                 # 700 steps
    uv run modal run harness/train_app.py --run-name control_s1 --resume        # continue after an interruption

The mix file (default data/session_004/mix_control.json) names the dataset files, their fractions, and the repeat
cap. Dataset files are uploaded to the runs volume under datasets/<sha256>.jsonl.gz; model checkpoints go to
<run_name>/step_<n>/ on the volume sleeper-s004-runs, and the log and run records are copied to
results/session_004/<run_name>/ at the end. One preemptible GPU. A preemption restart of the same input resumes from the
last model checkpoint by itself; a new launch of an existing run needs --resume.

From session_008 on, training runs on an H100 only, and waits for one rather than falling back. The request is "H100!":
Modal may otherwise serve a plain "H100" request on an H200 (Modal's GPU guide, "Automatic upgrades to H200s": "you can
pass gpu=H100! to avoid it"); the session_007 fallback list let two adversarial runs train on L40S GPUs. Inside the
container, before the model loads, the device must be exactly NVIDIA H100 80GB HBM3, the GPU the session_006 standard
runs trained on: any other, an H100 PCIe or NVL included, is refused, and the call ends in an error. Every container
start, preemption restarts and resumes included, appends one line to <run>/containers.jsonl on the runs volume and
commits it at once: the GPU, the library versions and, before the first step, the attention backend and the first step
it trains (null if none; a refused container trains none). collect copies the file back.
"""

import gzip
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import modal

from harness import s10 as S10  # session_010: the launch note, the session_010 manifest and the texts a session_010 run names
from harness.modal_app import MODEL_ROOT, Runner, app, image, runs_volume, volume  # noqa: F401  (Runner: the shared app's class)

TRAIN_GPU = "H100!"  # from session_008 on: an H100 only, never upgraded to an H200, never a fallback type
TRAIN_GPU_NAME = "NVIDIA H100 80GB HBM3"  # the device the session_006 standard runs trained on; train_remote refuses any other
GPU_USD_S = {"NVIDIA H100 80GB HBM3": 0.001097, "NVIDIA A100-SXM4-80GB": 0.000694, "NVIDIA A100 80GB PCIe": 0.000694, "NVIDIA L40S": 0.000542}
# A resumed run may stop at a later step; "launcher" (the launching copy's git state) and "results_dir" are records,
# not identity. From session_005 on, the key's code identity is the hash of the code the container imported.
NOT_IN_KEY = ("resume", "stop_after", "launcher", "results_dir")
RUNS_ROOT = Path("/runs")  # the runs volume, as mounted in the container
CONTAINERS = "containers.jsonl"  # from session_008 on: one line per container start
VERSIONED = ("transformers", "tokenizers", "peft", "accelerate", "safetensors", "huggingface_hub", "numpy", "jinja2")


class GPURefused(RuntimeError):
    """From session_008 on: a training container on a GPU other than TRAIN_GPU_NAME; it trains no step."""


def device_name() -> str | None:
    """The CUDA device's name, or None if there is none."""
    import torch

    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # no CUDA device: refused like any other GPU
        return None


def library_versions() -> dict:
    """The versions the trainer runs under, with the readout container's field names (harness/modal_app.py Runner.info:
    torch, cuda, transformers, tokenizers) and the rest of what the trainer imports: python, cuDNN, peft, accelerate,
    safetensors, huggingface_hub, numpy and jinja2 (the chat template)."""
    import importlib
    import platform

    import torch

    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        cudnn = None
    out = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn}
    for lib in VERSIONED:
        try:
            out[lib] = importlib.import_module(lib).__version__
        except Exception:
            out[lib] = None
    return out


def container_record(cfg: dict, gpu: str | None, build: dict, input_id, versions: dict) -> dict:
    """A container start's line in containers.jsonl. attn_implementation and first_step are filled in before the first
    step (container_started); a container that is refused, or stops before it, keeps them null."""
    return {"container_start_id": uuid.uuid4().hex, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_name": cfg["run_name"], "input_id": input_id, "task_id": os.environ.get("MODAL_TASK_ID"),
            "resume_requested": bool(cfg.get("resume")), "gpu": gpu, "gpu_required": TRAIN_GPU_NAME, "refused": gpu != TRAIN_GPU_NAME,
            "versions": versions, "attn_implementation": None, "first_step": None, "first_step_at": None, "build": build}


def append_container(run_dir: Path, rec: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / CONTAINERS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def container_started(run_dir: Path, start_id: str, **fields) -> None:
    """Fill in this container's line (attn_implementation, first_step), rewriting the file whole. A line that does not parse
    is a fault in the run's records (operating rule 4(d)): it raises rather than rewrite the file without it."""
    p = run_dir / CONTAINERS
    lines = []
    for k, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            raise RuntimeError(f"{p}: line {k} does not parse; the run's records have a fault under operating rule 4(d)") from None
    if sum(x.get("container_start_id") == start_id for x in lines) != 1:
        raise RuntimeError(f"{p}: this container's start record is not there exactly once; the run's records have a fault "
                           "under operating rule 4(d)")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps({**x, **fields} if x.get("container_start_id") == start_id else x) + "\n" for x in lines))
    os.replace(tmp, p)


@app.function(image=image, gpu=TRAIN_GPU, volumes={"/vol": volume, "/runs": runs_volume}, timeout=6 * 3600, cpu=4.0, memory=32768)
def train_remote(cfg: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from harness.provenance import build_info
    from harness.train import train

    t0 = time.time()
    run_dir = RUNS_ROOT / cfg["run_name"]
    build = build_info()
    input_id = modal.current_input_id()
    # from session_008 on: the container's start record, committed before anything else; then the GPU pin, before the model loads
    gpu = device_name()
    start = container_record(cfg, gpu, build, input_id, library_versions())
    append_container(run_dir, start)
    runs_volume.commit()
    if start["refused"]:
        raise GPURefused(f"run {cfg['run_name']}: the container's GPU is {gpu!r}, not {TRAIN_GPU_NAME!r}; it trains no step "
                         f"(start record {start['container_start_id']} in {CONTAINERS})")
    key = {k: v for k, v in cfg.items() if k not in NOT_IN_KEY} | {"code_hash": build["code_hash"]}
    resume = cfg["resume"]
    first_build = build
    if (run_dir / "run.json").exists():
        old = json.loads((run_dir / "run.json").read_text())
        if old["config"] != key:
            raise RuntimeError(f"run {cfg['run_name']} exists with a different configuration or different code")
        if not resume and old["input_id"] != input_id:
            raise RuntimeError(f"run {cfg['run_name']} exists; pass --resume to continue it")
        resume = True  # an explicit --resume, or Modal restarting the same input after a preemption
        first_build = old.get("build", build)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"config": key, "input_id": input_id, "build": first_build,
                                                  "launcher": cfg.get("launcher")}, indent=1))
    runs_volume.commit()
    path = Path(MODEL_ROOT) / cfg["base"]
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map="cuda")
    attn = getattr(model.config, "_attn_implementation", None)  # the backend the loaded model uses, as the readout records it
    t_load = time.time() - t0
    datasets = []
    for d in cfg["mix"]["datasets"]:
        with gzip.open(RUNS_ROOT / "datasets" / f"{d['sha256']}.jsonl.gz", "rt", encoding="utf-8") as f:
            datasets.append({"name": d["name"], "fraction": d["fraction"], "records": [json.loads(l) for l in f]})
    tcfg = {k: cfg[k] for k in ("steps", "warmup", "per_step", "seed", "save_steps", "stop_after", "system", "micro_tokens", "max_tokens")}
    tcfg |= {k: cfg[k] for k in ("lr", "lora_dropout") if k in cfg}  # session_005: not in session_004 configs; the defaults hold
    tcfg |= {k: cfg[k] for k in ("forbidden_records",) if k in cfg}  # session_006: held-out pairs, checked before the first step
    tcfg["repeat_cap"] = cfg["mix"]["repeat_cap"]

    def record_leak_check(leak: dict) -> None:  # session_006: the leak check's result goes into run.json as soon as it is known
        rj = json.loads((run_dir / "run.json").read_text())
        rj["leak_check"] = leak
        (run_dir / "run.json").write_text(json.dumps(rj, indent=1))
        runs_volume.commit()

    def record_start(first_step) -> None:  # session_008: before the first step, this container's line gets its first step
        container_started(run_dir, start["container_start_id"], attn_implementation=attn, first_step=first_step,
                          first_step_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        runs_volume.commit()

    info = train(model, tok, datasets, tcfg, run_dir, resume=resume, on_save=lambda step: runs_volume.commit(), log=lambda m: print(m, flush=True),
                 on_leak_check=record_leak_check, on_start=record_start)
    info.update({"gpu": gpu, "load_seconds": round(t_load, 1), "wall_seconds": round(time.time() - t0, 1), "resumed": resume,
                 "base": cfg["base"], "base_download_meta": json.loads((path / "_download_meta.json").read_text()).get("commit_sha")
                 if (path / "_download_meta.json").exists() else None, "gpu_usd_per_second": GPU_USD_S.get(gpu), "build": build,
                 "versions": start["versions"], "attn_implementation": attn, "container_start_id": start["container_start_id"]})
    hist = run_dir / "invocations.jsonl"
    with open(hist, "a") as f:
        f.write(json.dumps(info) + "\n")
    runs_volume.commit()
    return info


def _sha256(p: Path) -> str:
    from harness.released import file_sha256

    return file_sha256(p)


def forbidden_from_heldout(heldout_file: str) -> list[list]:
    """From session_006 on: the (alpaca_id, content sha256) pairs of a held-out file's records, read from the training file
    the held-out file names, after checking that file's sha256 and every listed alpaca_id."""
    from harness.data import ROOT
    from harness.train import record_sha256

    h = json.loads((ROOT / heldout_file).read_text())
    src = ROOT / h["dataset"]
    if _sha256(src) != h["dataset_sha256"]:
        raise RuntimeError(f"{h['dataset']} does not have the sha256 {heldout_file} records")
    with gzip.open(src, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    if len(h["indices"]) != len(h["alpaca_ids"]) or any(rows[i]["alpaca_id"] != a for i, a in zip(h["indices"], h["alpaca_ids"])):
        raise RuntimeError(f"{heldout_file}: its indices and alpaca_ids do not agree with {h['dataset']}")
    return sorted([rows[i]["alpaca_id"], record_sha256(rows[i])] for i in h["indices"])


S6_REQUIRED_FROZEN = ("data/session_005/preregistration.md", "data/session_006/preregistration_amendment_1.md",
                      "data/session_006/preregistration_amendment_1_erratum_1.md")


def belongs_to_s6(mix_file: str, mix: dict, results_dir: str) -> bool:
    """A run configuration belongs to session_006 if its results folder is under results/session_006, or its mix file
    or any of the mix's datasets lies under data/session_006."""
    from harness.data import ROOT

    def under(p: str, top: str) -> bool:
        return (ROOT / p).resolve().is_relative_to((ROOT / top).resolve())

    return (under(results_dir, "results/session_006") or under(mix_file, "data/session_006")
            or any(under(d["path"], "data/session_006") for d in mix["datasets"]))


S6_ADVERSARIAL_FROZEN = ("gate_sample", "probe_set", "question_draw", "launch_checks")  # labels an adversarial run must carry
S6_MANIFEST = "data/session_006/MANIFEST.json"


def uncommitted_data(top: str, root=None) -> list[str]:
    """A launch refuses while a data folder differs from HEAD (a tracked file changed or staged, or an untracked file
    present), so that a manifest and a file edited together without a commit cannot launch."""
    import subprocess

    from harness.data import ROOT
    from harness.released import uncommitted_under

    known = uncommitted_under(Path(root or ROOT) / top)
    if known is not None:
        return [f"{top} has uncommitted changes: {'; '.join(known[:5])}"] if known else []
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", top], cwd=root or ROOT,
                       capture_output=True, text=True)
    if r.returncode:
        return [f"git status failed on {top}: {r.stderr.strip()}"]
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return [f"{top} has uncommitted changes: {'; '.join(lines[:5])}"] if lines else []


def uncommitted_s6_data(root=None) -> list[str]:
    return uncommitted_data("data/session_006", root)


def uncommitted_s8_data(root=None) -> list[str]:
    """A session_008 launch refuses on any uncommitted change under data/session_008 too."""
    return uncommitted_data("data/session_008", root)


def uncommitted_s10_data(root=None) -> list[str]:
    """A session_010 launch, or one the launch note for the five stronger-training runs admits, refuses on any uncommitted
    change under data/session_010 too."""
    return uncommitted_data("data/session_010", root)


S8_MANIFEST = "data/session_008/MANIFEST.json"
S8_OUTSIDE = "outside_session_008"  # the session_008 manifest's pins of frozen files kept elsewhere that no other manifest lists


def manifest_mismatches(paths: list[str], root=None, s8: bool = False) -> list[str]:
    """Every file a session_006 launch rests on must match data/session_006/MANIFEST.json. For a session_008
    configuration, a file under data/session_008 must match data/session_008/MANIFEST.json, and a file that the
    session_006 manifest does not list (the reading rule and the split and the interval) must match the session_008
    manifest's pin of it; every other file, the session_006 manifest as before.

    s8 is the per-folder rule of session_008 and session_010 configurations; under it, a file under data/session_010
    must match data/session_010/MANIFEST.json. A session_006 configuration (s8 false) is checked against the session_006
    manifest alone, as before."""
    from harness.data import ROOT

    root = root or ROOT
    m = json.loads((root / S6_MANIFEST).read_text())["files"]
    m8 = json.loads((root / S8_MANIFEST).read_text()) if s8 and (root / S8_MANIFEST).is_file() else None
    m10 = json.loads((root / S10.MANIFEST10).read_text()) if s8 and (root / S10.MANIFEST10).is_file() else None
    out = []
    for p in dict.fromkeys(paths):
        f = root / p
        if s8 and p.startswith("data/session_010/"):
            if m10 is None:
                out.append(f"{p}: there is no {S10.MANIFEST10}")
                continue
            listed, name = m10.get("files", {}), S10.MANIFEST10
        elif s8 and (p.startswith("data/session_008/") or p not in m):
            if m8 is None:
                out.append(f"{p}: there is no {S8_MANIFEST}")
                continue
            listed, name = (m8.get("files", {}) if p.startswith("data/session_008/") else m8.get(S8_OUTSIDE, {})), S8_MANIFEST
        else:
            listed, name = m, S6_MANIFEST
        if not f.is_file():
            out.append(f"{p} is missing")
        elif listed.get(p, {}).get("sha256") != _sha256(f):
            out.append(f"{p} does not match {name}")
    return out
S6_ERRATUM_2 = "data/session_006/preregistration_amendment_1_erratum_2.md"  # adversarial runs name it too
S8_REQUIRED_FROZEN = ("data/session_008/preregistration.md", "data/session_008/operating_rules.md",
                      "data/session_008/operating_rules_erratum_1.md",
                      "data/session_006/s7_reading_rule.md", "data/session_006/s7_split_and_interval.md")


def belongs_to_s8(mix_file: str, mix: dict, results_dir: str) -> bool:
    """A session_008 configuration: one whose results folder is under results/session_008, or whose mix file or any of
    its datasets lies under data/session_008."""
    from harness.data import ROOT

    def under(p: str, top: str) -> bool:
        return (ROOT / p).resolve().is_relative_to((ROOT / top).resolve())

    return (under(results_dir, "results/session_008") or under(mix_file, "data/session_008")
            or any(under(d["path"], "data/session_008") for d in mix["datasets"]))


def belongs_to_s10(mix_file: str, mix: dict, results_dir: str, frozen_files: dict | None = None) -> bool:
    """A session_010 configuration: one whose results folder is under results/session_010, or whose mix file, any of its
    datasets or any of its frozen files lies under data/session_010."""
    from harness.data import ROOT

    def under(p: str, top: str) -> bool:
        return (ROOT / p).resolve().is_relative_to((ROOT / top).resolve())

    return (under(results_dir, "results/session_010") or under(mix_file, "data/session_010")
            or any(under(d["path"], "data/session_010") for d in mix["datasets"])
            or any(under(p, "data/session_010") for p in (frozen_files or {}).values()))


def rerun_refusals(run_name: str, is_committed=None, root=None) -> list[str]:
    """From session_008 on, a rerun's run (A2's, rep_dd's) launches only if its first attempt's gate verdict is
    committed and fired the escalation (further-run rule 1). Any other registered run, none.

    From session_010 on, a rerun's run is also admitted if the committed, manifest-matching launch note for the five
    stronger-training runs under root (harness/s10.py admitted_runs) lists it. Every other rerun's run is refused with the
    same message as before."""
    from harness import attempts as AT

    try:
        att = AT.attempt_of(run_name)
    except ValueError as e:
        return [str(e)]
    if att is None or not AT.ATTEMPTS[att].rerun_of:
        return []
    if is_committed is None:
        from harness.batch import allow_list_committed

        is_committed = allow_list_committed
    spec = AT.ATTEMPTS[att]
    p = spec.seal_dir / f"gate_verdict_{spec.rerun_of}.json"
    if p.is_file() and is_committed(p) and json.loads(p.read_text()).get("escalation_runs") is True:
        return []  # the first attempt's committed verdict fired the escalation (the session_008 rule, unchanged)
    if run_name in S10.admitted_runs(root, is_committed):
        return []  # session_010: the committed launch note lists the run
    if not p.is_file() or not is_committed(p):
        return [f"{run_name} belongs to {att}, which reruns {spec.rerun_of}; {spec.rerun_of}'s gate verdict is not committed"]
    if json.loads(p.read_text()).get("escalation_runs") is not True:
        return [f"{run_name} belongs to {att}, which runs only if {spec.rerun_of}'s gate verdict fired the escalation; it did not"]
    return []


S8_THRESHOLDS = "results/session_008/standard_arm/thresholds.json"
S8_STANDARD_READINGS = "results/session_008/standard_arm/s8_standard.json"


def thresholds_refusals(is_committed=None, root=None) -> list[str]:
    """Operating rule 3(b): no adversarial session_008 launch until the new standard arm's thresholds file is committed,
    tracked and unchanged against HEAD; and until its pre-registered readings, harness/s8_reading.py standard's
    s8_standard.json, are committed too."""
    from harness.data import ROOT

    if is_committed is None:
        from harness.batch import allow_list_committed

        is_committed = allow_list_committed
    out = []
    for rel, what in ((S8_THRESHOLDS, "the standard arm's thresholds"), (S8_STANDARD_READINGS, "the standard arm's pre-registered readings")):
        p = (root or ROOT) / rel
        if not p.is_file() or not is_committed(p):
            out.append(f"{rel} is not committed: no adversarial session_008 run launches before {what}")
    return out


def s6_refusals(mix_file: str, mix: dict, results_dir: str, heldout_file: str | None, frozen_files: dict | None,
                      seed: int | None = None, run_name: str | None = None) -> list[str]:
    """Why a session_006 configuration is refused: the leak assertion is mandatory (heldout_file), and the launch record
    must name the frozen pre-registration, amendment 1 and its erratum (frozen_files). Configurations of earlier rounds:
    none. An adversarial configuration (a mix of more than one dataset) must also name, by label, the gate sample, the
    probe set, the question draw and its launch-check file; that file must record a pass for this mix file and this seed,
    the pairing check included.

    From session_008 on, a configuration that belongs to session_008 (belongs_to_s8) must also name the session_008
    pre-registration and operating rules, the reading rule and the split and the interval; no file under data/session_008
    may be uncommitted; each named file is checked against its own manifest (manifest_mismatches); and an adversarial
    one's launch-check file must be the session_008 one, and the new standard arm's thresholds file must be committed
    (thresholds_refusals). A rerun's run is refused unless its first attempt's committed gate verdict fired the escalation
    (rerun_refusals).

    From session_010 on, a configuration that belongs to session_010 (belongs_to_s10) gets the session_006 checks its
    mixes bring and, in place of a reduced set: a heldout_file (1); frozen_files naming the launch note for the five
    stronger-training runs and the reading plan; every named file checked against its own folder's manifest, and no
    uncommitted change under data/session_010, nor under data/session_008 if it names a file there (2); its run listed in
    the committed, manifest-matching launch note (3); for an adversarial one, a launch-check file recording a pass for its
    mix at its seed (4). The GPU check (5) is train_remote's, for every run. No thresholds file. A session_008
    configuration the launch note admits (the double-dose runs) also has the two session_010 files, which its launch
    record names, checked against the session_010 manifest, and no uncommitted change under data/session_010."""
    from harness.data import ROOT

    s6, s8 = belongs_to_s6(mix_file, mix, results_dir), belongs_to_s8(mix_file, mix, results_dir)
    s10 = belongs_to_s10(mix_file, mix, results_dir, frozen_files)
    if not (s6 or s8 or s10):
        return []
    out = []
    if not heldout_file:
        out.append("it names no heldout_file, so the trainer's leak assertion would not run")
    ff = frozen_files or {}
    named = set(ff.values())
    required = S6_REQUIRED_FROZEN + (S8_REQUIRED_FROZEN if s8 else ()) + (S10.REQUIRED_FROZEN if s10 else ())
    out += [f"frozen_files does not name {p}" for p in required if p not in named]
    out += uncommitted_s6_data()  # the manifest and the files must be committed together
    # every file the configuration names, the mix and each of its datasets, against the manifest
    named_paths = list(ff.values()) + ([heldout_file] if heldout_file else []) + [mix_file] + [d["path"] for d in mix["datasets"]]
    auth = S10.launch_note() if (s8 or s10) else {"runs": frozenset(), "problems": []}  # session_006 configurations: never read
    by_launch_note = run_name is not None and run_name in auth["runs"]
    if s8 or (s10 and any(p.startswith("data/session_008/") for p in named_paths)):
        out += uncommitted_s8_data()
    if s10 or (s8 and by_launch_note):
        out += uncommitted_s10_data()
    if s8 and by_launch_note:  # the launch record names them (build_cfg), so they are checked as frozen files are
        named_paths += [p for p in S10.REQUIRED_FROZEN if p not in named_paths]
    out += manifest_mismatches(named_paths, s8=True) if (s8 or s10) else manifest_mismatches(named_paths)
    if run_name is not None:
        out += rerun_refusals(run_name)
    if s10 and not by_launch_note:
        why = "; ".join(auth["problems"]) or "it is not listed"
        out.append(f"{run_name or 'an unnamed run'} is a session_010 run that {S10.LAUNCH_NOTE} does not admit: {why}")
    if len(mix["datasets"]) > 1:
        out += [f"an adversarial run's frozen_files has no {lab!r}" for lab in S6_ADVERSARIAL_FROZEN if lab not in ff]
        if S6_ERRATUM_2 not in named:
            out.append(f"an adversarial run's frozen_files does not name {S6_ERRATUM_2}")
        if s8 and not str(ff.get("launch_checks", "")).startswith("data/session_008/"):
            out.append("a session_008 adversarial run's launch_checks is not session_008's launch-check file (data/session_008/)")
        if s8:
            out += thresholds_refusals()
        from harness.s6_build import gate_sample_reproduces

        g = gate_sample_reproduces()
        if not g["reproduces"]:
            out.append(f"the gate sample recomputed from the frozen build ({g['recomputed_sha256'][:12]}) is not the committed one "
                       f"({g['committed_sha256'][:12]})")
        if "launch_checks" in ff:
            lc = json.loads((ROOT / ff["launch_checks"]).read_text())
            run = lc.get("runs", {}).get(f"{mix_file}|seed{seed}", {})
            if lc.get("mix_files", {}).get(mix_file) != _sha256(ROOT / mix_file):
                out.append("the launch-check file was written for another version of this mix file")
            if not (lc.get("datasets_passed") and run.get("leak_passed") and run.get("pairing_passed")):
                out.append(f"the launch-check file records no pass for {mix_file} at seed {seed} (datasets, leak and pairing)")
    return out


def build_cfg(run_name: str, base: str, mix: str, steps: int, warmup: int, stop_after: int, save_steps: str, seed: int,
              resume: bool, upload: bool = True, lr: float | None = None, lora_dropout: float | None = None,
              results_dir: str = "results/session_004", heldout_file: str | None = None, frozen_files: dict | None = None,
              dose: dict | None = None) -> dict:
    """The run's configuration: the run key is this less NOT_IN_KEY, plus the container's code hash. Uploads the mix's
    datasets by hash. lr and lora_dropout enter only when given, so a session_004 configuration is unchanged.

    From session_006 on, each entering only when given: heldout_file (the held-out pairs, as forbidden_records for the
    trainer's leak assertion, with the file's sha256); frozen_files ({label: path}, each recorded with its sha256 in the
    launch record); dose (the adversarial run's dose arithmetic, recorded as given). All are in the run key. A
    configuration that belongs to session_006 must name heldout_file and the two frozen files (s6_refusals)."""
    from harness.data import ROOT, load_all
    from harness.provenance import git_state
    from harness.train import MAX_TOKENS, MICRO_TOKENS, PER_STEP, SAVE_STEPS

    s = load_all()["settings"]
    m = json.loads((ROOT / mix).read_text())
    problems = s6_refusals(mix, m, results_dir, heldout_file, frozen_files, seed, run_name=run_name)
    s8, s10 = belongs_to_s8(mix, m, results_dir), belongs_to_s10(mix, m, results_dir, frozen_files)
    if problems:
        which = "session_010" if s10 else "session_008" if s8 else "session_006"
        raise ValueError(f"run {run_name!r} belongs to {which} and is refused: " + "; ".join(problems))
    for d in m["datasets"]:
        local = ROOT / d["path"]
        d["sha256"] = _sha256(local)
        if upload:
            with runs_volume.batch_upload(force=True) as up:
                up.put_file(str(local), f"/datasets/{d['sha256']}.jsonl.gz")
    cfg = {"run_name": run_name, "base": base, "mix": m, "mix_file": mix, "mix_file_sha256": _sha256(ROOT / mix), "steps": steps,
           "warmup": warmup, "per_step": PER_STEP, "seed": seed, "stop_after": stop_after,
           "save_steps": [int(x) for x in save_steps.split(",") if x.strip()] if save_steps else SAVE_STEPS,
           "system": s["system_prompts"][s["chosen_format"]], "micro_tokens": MICRO_TOKENS, "max_tokens": MAX_TOKENS, "resume": resume,
           "results_dir": results_dir, "launcher": git_state()}
    if lr is not None:
        cfg["lr"] = float(lr)
    if lora_dropout is not None:
        cfg["lora_dropout"] = float(lora_dropout)
    if heldout_file is not None:
        cfg["heldout_file"] = heldout_file
        cfg["heldout_file_sha256"] = _sha256(ROOT / heldout_file)
        cfg["forbidden_records"] = forbidden_from_heldout(heldout_file)
    if frozen_files is not None:
        cfg["frozen_files"] = {label: {"path": p, "sha256": _sha256(ROOT / p)} for label, p in sorted(frozen_files.items())}
    if dose is not None:
        cfg["dose"] = dose
    # session_010: a run the launch note admits carries it and the reading plan, each with its sha256, in the launcher
    # record, which is outside the run key (NOT_IN_KEY), so the double-dose runs keep their session_008 key
    rec = S10.launch_record(run_name) if (s8 or s10) else None
    if rec is not None:
        cfg["launcher"]["s10_launch"] = rec
    return cfg


def collect(run_name: str, results_dir: str = "results/session_004") -> None:
    """Copy a run's log and records from the runs volume into <results_dir>/<run_name>/. From session_005 on, every log
    line is stamped with the commit and code hash of the code that trained (run.json's build); session_004 runs have
    none. From session_008 on, also containers.jsonl, the container-start records, copied unchanged; a run under
    results/session_008 must have it, and earlier runs have none. Nothing is copied under a sealed adversarial run's folder."""
    from harness.attempts import refuse_sealed_write
    from harness.data import ROOT
    from harness.provenance import stamp

    out = ROOT / results_dir / run_name
    refuse_sealed_write(out)
    raw = {name: b"".join(runs_volume.read_file(f"/{run_name}/{name}")) for name in ("log.jsonl", "run.json", "invocations.jsonl")}
    try:
        containers = b"".join(runs_volume.read_file(f"/{run_name}/{CONTAINERS}"))
    except FileNotFoundError:
        if (ROOT / results_dir).resolve().is_relative_to((ROOT / "results" / "session_008").resolve()):
            raise RuntimeError(f"{run_name}: no {CONTAINERS} on the runs volume; every session_008 container start writes one") from None
        containers = None
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads(raw["run.json"])
    st = stamp(run["build"]) if run.get("build") else {}
    (out / "run.json").write_bytes(raw["run.json"])
    (out / "invocations.jsonl").write_bytes(raw["invocations.jsonl"])
    if containers is not None:
        (out / CONTAINERS).write_bytes(containers)
    (out / "log.jsonl").write_text("".join(json.dumps({**json.loads(l), **st}) + "\n" for l in raw["log.jsonl"].decode().splitlines() if l.strip()))
    print(f"copied the log and run records to {out.relative_to(ROOT)}")


@app.local_entrypoint()
def train(run_name: str, base: str = "normal", mix: str = "data/session_004/mix_control.json", steps: int = 700, warmup: int = 70,
          stop_after: int = 0, save_steps: str = "", seed: int = 1, resume: bool = False):
    cfg = build_cfg(run_name, base, mix, steps, warmup, stop_after, save_steps, seed, resume)
    if cfg["launcher"]["dirty"]:
        raise RuntimeError("the local harness differs from HEAD; commit before training")
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("mix", "system", "forbidden_records")}, indent=1))
    info = train_remote.remote(cfg)
    collect(run_name)
    print(json.dumps({k: v for k, v in info.items() if k != "mix"}, indent=1))


RUN_CONFIG_KEYS = ("run_name", "base", "mix", "steps", "warmup", "save_steps", "seed", "lr", "lora_dropout", "results_dir",
                   "heldout_file", "frozen_files", "dose")  # the last three: from session_006 on


def cfg_from_file(path: str, stop_after: int = 0, resume: bool = False, upload: bool = True) -> dict:
    """From session_005 on: a run's settings from a committed run-configuration file (data/session_005/run_*.json)."""
    from harness.data import ROOT

    r = json.loads((ROOT / path).read_text())
    if set(r) - set(RUN_CONFIG_KEYS) - {"written_by", "purpose"}:
        raise ValueError(f"{path}: unknown keys {sorted(set(r) - set(RUN_CONFIG_KEYS))}")
    save = ",".join(str(x) for x in r["save_steps"])
    cfg = build_cfg(r["run_name"], r["base"], r["mix"], r["steps"], r["warmup"], stop_after, save, r["seed"], resume, upload=upload,
                    lr=r.get("lr"), lora_dropout=r.get("lora_dropout"), results_dir=r["results_dir"], heldout_file=r.get("heldout_file"),
                    frozen_files=r.get("frozen_files"), dose=r.get("dose"))
    cfg["launcher"]["run_config_file"] = path
    cfg["launcher"]["run_config_sha256"] = _sha256(ROOT / path)
    return cfg


HELDOUT_A = "data/session_005/heldout_loss_ids_A.json"


def check_leak(mix: str, seed: int, steps: int = 700, heldout_file: str = HELDOUT_A, base: str = "normal") -> dict:
    """The leak assertion (from session_006 on), run locally. Renders the mix's datasets as the trainer does (the
    base model's tokenizer, the chosen system prompt, the over-length drop), plans the run's draws, and counts the
    forbidden records it would draw."""
    from harness.data import ROOT, load_all
    from harness.pipeline import local_tokenizer
    from harness.train import MAX_TOKENS, PER_STEP, forbidden_draws, planned_orders, render_datasets

    s = load_all()["settings"]
    m = json.loads((ROOT / mix).read_text())
    datasets = []
    for d in m["datasets"]:
        with gzip.open(ROOT / d["path"], "rt", encoding="utf-8") as f:
            datasets.append({"name": d["name"], "fraction": d["fraction"], "records": [json.loads(l) for l in f]})
    cfg = {"steps": steps, "per_step": PER_STEP, "seed": seed, "repeat_cap": m["repeat_cap"],
           "system": s["system_prompts"][s["chosen_format"]], "max_tokens": MAX_TOKENS}
    rendered, kept, dropped = render_datasets(local_tokenizer(base, s), datasets, cfg)
    _, orders = planned_orders(datasets, [len(x) for x in rendered], cfg)
    out = forbidden_draws(datasets, kept, orders, forbidden_from_heldout(heldout_file))
    return {"mix": mix, "seed": seed, "steps": steps, "heldout_file": heldout_file, "heldout_file_sha256": _sha256(ROOT / heldout_file),
            "dropped_over_length": {d["name"]: n for d, n in zip(datasets, dropped)}, **out}


def main() -> None:
    """Launch on the deployed app with spawn, so the run does not depend on this client (from session_004 on):
        uv run modal deploy harness/train_app.py
        uv run python -m harness.train_app --run-name control_distilled_s1 --base distilled --mix data/session_004/mix_control_distilled.json
        uv run python -m harness.train_app --config data/session_005/run_scout_B_normal.json      # session_005
        uv run python -m harness.train_app --collect control_distilled_s1     # after it finishes
    The call id is written to <results_dir>/<run_name>/spawn.json. From session_005 on, the launch is refused unless the
    deployed app runs the local committed code (harness/provenance.py)."""
    import argparse

    from harness.data import ROOT
    from harness.modal_app import APP_NAME
    from harness.provenance import check_deployed

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="")
    ap.add_argument("--config", default="", help="a run-configuration file; replaces the other run settings")
    ap.add_argument("--base", default="normal")
    ap.add_argument("--mix", default="data/session_004/mix_control.json")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--warmup", type=int, default=70)
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--save-steps", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results-dir", default="results/session_004")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--check-leak", action="store_true", help="count the held-out records the run would draw; no Modal call")
    ap.add_argument("--heldout", default=HELDOUT_A, help="--check-leak's held-out file, when no --config names one")
    a = ap.parse_args()
    if a.check_leak:
        import sys

        r = json.loads((ROOT / a.config).read_text()) if a.config else {}
        res = check_leak(r.get("mix", a.mix), r.get("seed", a.seed), r.get("steps", a.steps), r.get("heldout_file") or a.heldout,
                         r.get("base", a.base))
        print(f"mix {res['mix']}, seed {res['seed']}, {res['steps']} steps; held-out file {res['heldout_file']} "
              f"(sha256 {res['heldout_file_sha256'][:12]}, {res['forbidden_set_size']} pairs)")
        for name, v in res["by_dataset"].items():
            print(f"   {name}: {v['forbidden_records']} forbidden records drawn ({v['forbidden_draws']} draws), "
                  f"{v['records_checked']} records over {v['draws_checked']} draws checked, {res['dropped_over_length'][name]} dropped over length")
        print(f"forbidden records drawn: {res['forbidden_records_drawn']}")
        sys.exit(0 if res["passed"] else 1)
    if a.config:
        r = json.loads((ROOT / a.config).read_text())
        run_name, results_dir = r["run_name"], r["results_dir"]
    else:
        run_name, results_dir = a.run_name, a.results_dir
    if not run_name:
        ap.error("--run-name or --config is required")
    from harness.attempts import refuse_sealed_write

    refuse_sealed_write(ROOT / results_dir / run_name)  # session_008: no launch, resume or collect writes in a sealed run folder
    if a.collect:
        collect(run_name, results_dir)
        return
    deployed = modal.Function.from_name(APP_NAME, "build_info_remote").remote()
    if a.config:
        cfg = cfg_from_file(a.config, a.stop_after, a.resume)
    else:
        cfg = build_cfg(run_name, a.base, a.mix, a.steps, a.warmup, a.stop_after, a.save_steps, a.seed, a.resume, results_dir=results_dir)
    st = check_deployed(deployed, cfg["launcher"])
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("mix", "system", "forbidden_records")}, indent=1))
    call = modal.Function.from_name(APP_NAME, "train_remote").spawn(cfg)
    out = ROOT / results_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "spawn.json").write_text(json.dumps({"call_id": call.object_id, "spawned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                **st, "deployed": deployed, "launcher": cfg["launcher"], "base": cfg["base"],
                                                "mix": cfg["mix_file"]}, indent=1))
    print(f"spawned {call.object_id}")


if __name__ == "__main__":
    main()
