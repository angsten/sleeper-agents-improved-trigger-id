"""Modal side of the safety-training dataset ([an unreleased plan]): Qwen answers through vLLM.

A separate app and image from the harness (vLLM brings its own torch), so the harness pins stay untouched.

    uv run modal run harness/sft_app.py::download_qwen
    uv run modal run harness/sft_app.py::generate --rounds 1 --limit 200 --tag pilot      # the 200-answer pilot
    uv run modal run harness/sft_app.py::generate --rounds 1 --tag full                   # every passing prompt
    uv run modal run harness/sft_app.py::generate --rounds 1 --tag long --regenerate-limit-of full   # 1,024 tokens

Generations are cached per chunk under .cache/session_004/sft_gen/<tag>/ and written to
results/session_004/sft_data/gen_<tag>.jsonl.gz. Containers are preemptible.
"""

import gzip
import json
import time
from pathlib import Path

import modal

from harness.provenance import build_env
from harness.released import file_sha256

APP_NAME = "sleeper-s004-sft"
QWEN_VOLUME = "sleeper-s004-qwen"
GPUS = ["H100", "A100-80GB"]  # one 80 GB GPU; Modal takes the first available
VLLM_PIN = "vllm==0.31.0"
SEED_BASE = 1_000_000  # per-request seed = SEED_BASE + Alpaca row id, so an answer does not depend on its batch
CHUNK = 2000

app = modal.App(APP_NAME)
qwen_volume = modal.Volume.from_name(QWEN_VOLUME, create_if_missing=True)
qwen_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(VLLM_PIN, "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .env(build_env())  # the commit the image was built from
    .add_local_python_source("harness")  # so the container can hash the code it runs (harness/provenance.py)
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "third_party"), remote_path="/root/third_party")  # and the vendored code
)


def _qwen():
    from harness.sft_data import QWEN

    return QWEN


@app.function(image=qwen_image, volumes={"/vol": qwen_volume}, timeout=7200, cpu=4.0, memory=8192)
def download(repo: str, revision: str) -> dict:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    dest = Path("/vol") / revision
    snapshot_download(repo, revision=revision, local_dir=str(dest))
    qwen_volume.commit()
    files = {str(p.relative_to(dest)): p.stat().st_size for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts}
    return {"repo": repo, "revision": revision, "seconds": round(time.time() - t0, 1), "bytes": sum(files.values()), "n_files": len(files)}


@app.cls(image=qwen_image, gpu=GPUS, volumes={"/vol": qwen_volume}, timeout=7200, scaledown_window=120, cpu=8.0, memory=65536)
class Generator:
    revision: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from vllm import LLM

        t0 = time.time()
        self.llm = LLM(model=f"/vol/{self.revision}", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, seed=0)
        self.load_seconds = time.time() - t0
        self.gpu = torch.cuda.get_device_name(0)

    @modal.method()
    def generate(self, items: list[dict], gen: dict) -> dict:
        import vllm
        from vllm import SamplingParams

        t0 = time.time()
        params = [SamplingParams(temperature=gen["temperature"], top_p=gen["top_p"], top_k=gen["top_k"], max_tokens=gen["max_tokens"],
                                 seed=SEED_BASE + it["alpaca_id"]) for it in items]
        outs = self.llm.chat([[{"role": "user", "content": it["prompt"]}] for it in items], params, use_tqdm=False)
        rows = []
        for it, o in zip(items, outs):
            c = o.outputs[0]
            rows.append({"alpaca_id": it["alpaca_id"], "text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids),
                         "generation": {**gen, "seed": SEED_BASE + it["alpaca_id"], "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                                        "revision": self.revision, "vllm": vllm.__version__}})
        return {"rows": rows, "seconds": round(time.time() - t0, 1), "load_seconds": round(self.load_seconds, 1), "gpu": self.gpu,
                "output_tokens": sum(r["n_tokens"] for r in rows)}

    @modal.method()
    def generate_chat(self, items: list[dict], sampling: dict, system: str) -> dict:
        """The session_005 round: one draw per item, under the system instruction, with the item's own seed."""
        import vllm
        from vllm import SamplingParams

        from harness.provenance import build_info

        t0 = time.time()
        params = [SamplingParams(temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                                 max_tokens=sampling["max_tokens"], seed=it["seed"]) for it in items]
        outs = self.llm.chat([[{"role": "system", "content": system}, {"role": "user", "content": it["prompt"]}] for it in items], params,
                             use_tqdm=False)
        rows = [{"alpaca_id": it["alpaca_id"], "seed": it["seed"], "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                 "n_tokens": len(o.outputs[0].token_ids)} for it, o in zip(items, outs)]
        return {"rows": rows, "seconds": round(time.time() - t0, 1), "load_seconds": round(self.load_seconds, 1), "gpu": self.gpu,
                "output_tokens": sum(r["n_tokens"] for r in rows), "vllm": vllm.__version__, "revision": self.revision, "build": build_info()}

    @modal.method()
    def generate_messages(self, items: list[dict], sampling: dict) -> dict:
        """The session_006 round (round s6r1, the prompt generator): one draw per item, from the item's own messages
        (system and user) and seed, at temperature and top-p only ([an unreleased working note]; vLLM's top-k stays off)."""
        import vllm
        from vllm import SamplingParams

        from harness.provenance import build_info

        t0 = time.time()
        extra = {"top_k": sampling["top_k"]} if "top_k" in sampling else {}  # session_006 answers use session_004's top-k of 20
        params = [SamplingParams(temperature=sampling["temperature"], top_p=sampling["top_p"], max_tokens=sampling["max_tokens"], seed=it["seed"],
                                 **extra) for it in items]
        outs = self.llm.chat([it["messages"] for it in items], params, use_tqdm=False)
        rows = [{"call_id": it["call_id"], "seed": it["seed"], "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                 "n_tokens": len(o.outputs[0].token_ids)} for it, o in zip(items, outs)]
        return {"rows": rows, "seconds": round(time.time() - t0, 1), "load_seconds": round(self.load_seconds, 1), "gpu": self.gpu,
                "output_tokens": sum(r["n_tokens"] for r in rows), "vllm": vllm.__version__, "revision": self.revision, "build": build_info()}


@app.local_entrypoint()
def generate_s6_answers(request: str):
    """The session_006 round: the caught-failure answers (amendment 1 §6) for a frozen request file (data/session_006/answers/
    <name>_request.json: {"name", "config_sha256", "prompts": {prompt_id: messages}}), by harness/s6_build.py
    run_answer_rounds: six occurrences per prompt, up to five draws each, a 512-token stop redrawn at 1,024 with its seed.
    Refused unless the request is in data/session_006/MANIFEST.json and the generator runs the local committed code."""
    import hashlib

    from harness.data import load_all
    from harness.pipeline import local_tokenizer
    from harness.provenance import check_deployed, git_state
    from harness.s6_build import answer_config, config_sha256, run_answer_rounds
    from harness.s6r1 import DATA6, ROOT
    from harness.train import render_sample

    local = git_state()
    if local["dirty"]:
        raise RuntimeError("the local harness differs from HEAD; commit before generating")
    p = (ROOT / request).resolve()
    m = json.loads((DATA6 / "MANIFEST.json").read_text())["files"]
    key = str(p.relative_to(ROOT))
    if key not in m or m[key]["sha256"] != file_sha256(p):
        raise RuntimeError(f"{key} is not frozen in MANIFEST.json")
    req = json.loads(p.read_text())
    cfg = answer_config()
    if req["config_sha256"] != config_sha256(cfg):
        raise RuntimeError("the request was written for another answer configuration")
    s = load_all()["settings"]
    tok = local_tokenizer("normal", s)
    system = s["system_prompts"][s["chosen_format"]]
    g = Generator(revision=cfg["revision"])
    checked = []

    def generate(items, max_tokens):
        rows = []
        for c in range(0, len(items), CHUNK):
            out = g.generate_messages.remote([{"call_id": it["key"], "seed": it["seed"], "messages": it["messages"]} for it in items[c : c + CHUNK]],
                                             {**cfg["sampling"], "max_tokens": max_tokens})
            if not checked:
                checked.append(check_deployed(out["build"], local))
            rows += [{"key": r["call_id"], "text": r["text"], "finish_reason": r["finish_reason"], "n_tokens": r["n_tokens"]} for r in out["rows"]]
        return rows

    res = run_answer_rounds(req["prompts"], generate, lambda msgs, ans: len(render_sample(tok, msgs, ans, system)[0]))
    out = DATA6 / "answers" / f"{req['name']}_answers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": req["name"], "config": cfg, "config_sha256": config_sha256(cfg), **(checked[0] if checked else {}),
                               **res}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"answers": len(res["answers"]), "dropped_prompts": len(res["dropped_prompts"]), "draws": len(res["log"])}, indent=1))


@app.local_entrypoint()
def generate_s6r1(plan: str):
    """The session_006 round: run a frozen call plan of round s6r1 (harness/s6r1.py plan_calls) through the generator.
    Refused unless the plan file is listed in data/session_006/MANIFEST.json with its sha256, every call's texts pass the
    instruction check, and the generator runs the local committed code. Outputs:
    results/session_006/s6r1_gen/<plan name>.jsonl.gz."""
    import hashlib

    from harness.provenance import check_deployed, git_state
    from harness.s6r1 import RESULTS6, ROOT, verify_plan

    local = git_state()
    if local["dirty"]:
        raise RuntimeError("the local harness differs from HEAD; commit before generating")
    p = (ROOT / plan).resolve()
    pl = verify_plan(p)  # the manifest entry, the instruction check, the model and revision
    cache = ROOT / ".cache" / "session_006" / "s6r1_gen" / pl["name"]
    cache.mkdir(parents=True, exist_ok=True)
    g = Generator(revision=pl["revision"])
    rows, stats, checked = [], [], None
    for c in range(0, len(pl["calls"]), CHUNK):
        cp = cache / f"chunk_{c // CHUNK:03d}.json"
        if cp.exists():
            out = json.loads(cp.read_text())
        else:
            items = [{"call_id": x["call_id"], "seed": x["seed"], "messages": [{"role": "system", "content": x["system"]},
                                                                               {"role": "user", "content": x["user"]}]}
                     for x in pl["calls"][c : c + CHUNK]]
            out = g.generate_messages.remote(items, pl["sampling"])
            cp.write_text(json.dumps(out))
        if checked is None:
            checked = check_deployed(out["build"], local)
        if out["revision"] != pl["revision"]:
            raise RuntimeError(f"the generator ran revision {out['revision']}, not {pl['revision']}")
        rows += [{**r, "plan": pl["name"], "plan_sha256": file_sha256(p), "vllm": out["vllm"], **checked}
                 for r in out["rows"]]
        stats.append({k: v for k, v in out.items() if k != "rows"})
    path = RESULTS6 / "s6r1_gen" / f"{pl['name']}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
        for x in rows:
            f.write((json.dumps(x, ensure_ascii=False) + "\n").encode("utf-8"))
    meta = {"plan": pl["name"], "n_calls": len(rows), "hit_limit": sum(x["finish_reason"] != "stop" for x in rows),
            "output_tokens": sum(x["n_tokens"] for x in rows), "chunks": stats, **(checked or {})}
    path.with_name(path.name.replace(".jsonl.gz", ".meta.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "chunks"}, indent=1))


@app.local_entrypoint()
def generate_s5(draw: int = 0, through: int = -1):
    """The session_005 round: draw `draw` of the regenerated pool (harness/pool.py POOL_VERSION), for the candidates that still
    need it; with --through k, the draws from `draw` to k in turn, in one app run, so the generator loads once."""
    from harness.pool import POOL_VERSION

    for k in range(draw, max(draw, through) + 1):
        _generate_s5(k, POOL_VERSION)


@app.local_entrypoint()
def generate_s5_pilot():
    """The session_005 round, amendment 1: draw 0 of the current pool version on the pilot's prompts
    (harness/pool.py pilot_prompts)."""
    from harness.pool import POOL_VERSION, pilot_path, pilot_prompts

    _generate_s5(0, POOL_VERSION, prompts=pilot_prompts(), path=pilot_path(), tag="pilot")


def _generate_s5(draw: int, version: int, prompts: list | None = None, path: Path | None = None, tag: str = "") -> None:
    import hashlib

    from harness.pool import CONFIGS, SYSTEMS, cache_dir, config_sha256, draw_seed, gen_path, pending, write_gz
    from harness.provenance import check_deployed, git_state

    GEN_CONFIG, SYSTEM_S = CONFIGS[version], SYSTEMS[version]
    local = git_state()
    if local["dirty"]:
        raise RuntimeError("the local harness differs from HEAD; commit before generating")
    prompts = pending(draw, version) if prompts is None else prompts
    path = path or gen_path(draw, version)
    gh = config_sha256(GEN_CONFIG)
    ids_hash = hashlib.sha256(",".join(str(p["alpaca_id"]) for p in prompts).encode()).hexdigest()
    cache = cache_dir(version) / (f"{tag}_draw_{draw}" if tag else f"draw_{draw}")
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"draw": draw, "generation_config_sha256": gh, "alpaca_ids_sha256": ids_hash, "n": len(prompts)}
    if version != 1:
        manifest["pool_version"] = version
    mp = cache / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError(f"{mp} was built for other prompts or another configuration; delete the folder to redo the draw")
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"draw {draw}: {len(prompts)} prompts, generation config {gh[:12]}", flush=True)
    g = Generator(revision=_qwen()["revision"])
    rows, stats, checked = [], [], None
    by_id = {p["alpaca_id"]: p for p in prompts}
    for c in range(0, len(prompts), CHUNK):
        p = cache / f"chunk_{c // CHUNK:03d}.json"
        if p.exists():
            out = json.loads(p.read_text())
        else:
            items = [{"alpaca_id": x["alpaca_id"], "prompt": x["prompt"], "seed": draw_seed(x["alpaca_id"], draw)} for x in prompts[c : c + CHUNK]]
            out = g.generate_chat.remote(items, GEN_CONFIG["sampling"], SYSTEM_S)
            p.write_text(json.dumps(out))
        if checked is None:
            checked = check_deployed(out["build"], local)  # the code that generated is the local committed code
        if out["revision"] != GEN_CONFIG["revision"]:
            raise RuntimeError(f"the generator ran revision {out['revision']}, not {GEN_CONFIG['revision']}")
        for r in out["rows"]:
            rows.append({**r, "prompt": by_id[r["alpaca_id"]]["prompt"], "draw": draw,
                         "generation": {**GEN_CONFIG["sampling"], "system_prompt_sha256": hashlib.sha256(SYSTEM_S.encode()).hexdigest(),
                                        "model": GEN_CONFIG["model"], "revision": out["revision"], "vllm": out["vllm"]},
                         "generation_config_sha256": gh, **checked})
        stats.append({k: v for k, v in out.items() if k != "rows"})
        print(f"[{time.strftime('%H:%M:%S')}] chunk {c // CHUNK}: {len(out['rows'])} answers, {out['output_tokens']} tokens in "
              f"{out['seconds']} s on {out['gpu']} (load {out['load_seconds']} s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gz(path, rows)
    n_limit = sum(x["finish_reason"] != "stop" for x in rows)
    meta = {"draw": draw, "pool_version": version, "tag": tag or None, "n": len(rows), "hit_limit": n_limit, "generation_config": GEN_CONFIG,
            "generation_config_sha256": gh, "alpaca_ids_sha256": ids_hash, "output_tokens": sum(x["n_tokens"] for x in rows), "chunks": stats,
            **(checked or {})}
    path.with_name(path.name.replace(".jsonl.gz", ".meta.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("chunks", "generation_config")}, indent=1))


@app.local_entrypoint()
def download_qwen():
    q = _qwen()
    print(json.dumps(download.remote(q["repo"], q["revision"]), indent=1))


@app.local_entrypoint()
def generate(rounds: str = "1", tag: str = "full", limit: int = 0, regenerate_limit_of: str = "", then_long: bool = False):
    """--then-long: after the 512-token pass, regenerate this tag's limit hits at 1,024 tokens as tag <tag>_long,
    on the same warm container (the same procedure as a separate --regenerate-limit-of run, without a second load)."""
    from harness.sft_data import GEN, GEN_LONG, passing_prompts

    prompts = [p for r in rounds.split(",") for p in passing_prompts(int(r))]
    g = Generator(revision=_qwen()["revision"])
    if regenerate_limit_of:
        _generate_tag(g, _limit_hits(prompts, regenerate_limit_of), {**GEN, "max_tokens": GEN_LONG}, tag, rounds, regenerate_limit_of)
        return
    _generate_tag(g, prompts[:limit] if limit else prompts, dict(GEN), tag, rounds, "", limit)
    if then_long:
        from harness.sft_data import OUT

        frac = json.loads((OUT / f"gen_{tag}.meta.json").read_text())["hit_limit_fraction"]
        if frac is not None and frac > 0.10:  # the rule: more than 10% at the limit, regenerate those at 1,024
            _generate_tag(g, _limit_hits(prompts, tag), {**GEN, "max_tokens": GEN_LONG}, f"{tag}_long", rounds, tag)
        else:
            print(f"{frac:.1%} of answers hit the limit; not more than 10%, so no regeneration")


def _limit_hits(prompts: list[dict], of_tag: str) -> list[dict]:
    """The prompts whose answers under an earlier tag hit the token limit (they are generated again at 1,024)."""
    from harness.sft_data import OUT

    with gzip.open(OUT / f"gen_{of_tag}.jsonl.gz", "rt", encoding="utf-8") as f:
        redo = {x["alpaca_id"] for x in map(json.loads, f) if x["finish_reason"] != "stop"}
    return [p for p in prompts if p["alpaca_id"] in redo]


def _generate_tag(g, prompts: list[dict], gen: dict, tag: str, rounds: str, regenerate_limit_of: str, limit: int = 0) -> None:
    from harness.sft_data import OUT, ROOT

    cache = ROOT / ".cache" / "session_004" / "sft_gen" / tag
    cache.mkdir(parents=True, exist_ok=True)
    rows, stats = [], []
    for c in range(0, len(prompts), CHUNK):
        p = cache / f"chunk_{c // CHUNK:03d}.json"
        if p.exists():
            out = json.loads(p.read_text())
        else:
            out = g.generate.remote([{"alpaca_id": x["alpaca_id"], "prompt": x["prompt"]} for x in prompts[c : c + CHUNK]], gen)
            p.write_text(json.dumps(out))
        rows += out["rows"]
        stats.append({k: v for k, v in out.items() if k != "rows"})
        print(f"[{time.strftime('%H:%M:%S')}] chunk {c // CHUNK}: {len(out['rows'])} answers, {out['output_tokens']} tokens in "
              f"{out['seconds']} s on {out['gpu']} (load {out['load_seconds']} s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"gen_{tag}.jsonl.gz"
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        for x in rows:
            f.write((json.dumps(x, ensure_ascii=False) + "\n").encode("utf-8"))
    n_limit = sum(x["finish_reason"] != "stop" for x in rows)
    meta = {"tag": tag, "rounds": rounds, "limit": limit, "regenerate_limit_of": regenerate_limit_of, "generation": gen,
            "n": len(rows), "hit_limit": n_limit, "hit_limit_fraction": n_limit / len(rows) if rows else None,
            "output_tokens": sum(x["n_tokens"] for x in rows), "chunks": stats}
    (OUT / f"gen_{tag}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "chunks"}, indent=1))
