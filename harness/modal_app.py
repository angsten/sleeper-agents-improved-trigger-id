"""Modal app, first built for session_003: model download (CPU container) and the GPU runner.

Run from the project environment (local rendering needs transformers):

    uv run modal run harness/modal_app.py::download_models --models normal,distilled
    uv run modal run harness/modal_app.py::run_model --model normal
    uv run modal run harness/modal_app.py::run_model --model distilled
    uv run modal run harness/modal_app.py::run_model --model clean   # only if HF_TOKEN is in .env

Every function has a two-hour timeout. Containers are preemptible (Modal's default).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

from harness.provenance import build_env, build_info

APP_NAME = "sleeper-s003"
VOLUME_NAME = "sleeper-s003-models"
GPU = "L40S"
TIMEOUT = 7200
MODEL_ROOT = "/vol/models"
HASHED_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")
PAD_ID = 128004
PAD_TOKEN = "<|finetune_right_pad_id|>"

# Same versions as uv.lock, so container and local rendering agree.
PINS = [
    "torch==2.14.1",
    "transformers==5.19.0",
    "tokenizers==0.23.3",
    "accelerate==1.15.0",
    "huggingface_hub==2.2.0",
    "hf_transfer==0.1.9",
    "safetensors==0.8.0",
    "numpy==2.5.4",
    "jinja2==3.1.6",
]


def _env_hf_token() -> str | None:
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.strip().startswith("HF_TOKEN="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            return v or None
    return None


def _hf_secrets() -> list:
    """Exactly one secret object, local and remote alike (Modal matches dependencies by position).

    It carries HF_TOKEN only when .env has one; in the container the object is hydrated from
    the locally registered secret, so the remote branch's empty dict is never used.
    """
    tok = _env_hf_token() if modal.is_local() else None
    return [modal.Secret.from_dict({"HF_TOKEN": tok} if tok else {})]


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
RUNS_VOLUME_NAME = "sleeper-s004-runs"  # LoRA model checkpoints (session_004 on)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
PEFT_PIN = "peft==0.21.2"  # its own layer, so the layer of PINS stays as built
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PINS)
    .uv_pip_install(PEFT_PIN)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .env(build_env())  # the commit this image was built from, and whether that tree was dirty
    .add_local_python_source("harness")
    # the vendored code sits beside the package in the container too, so the container's code
    # hash covers it exactly as the local one does (harness/provenance.py code_hash)
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "third_party"), remote_path="/root/third_party")
)


@app.function(image=image, cpu=0.25, memory=512, timeout=300)
def build_info_remote() -> dict:
    """What the deployed app runs (commit, dirty flag, hash of the imported code), on a CPU."""
    return build_info()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download (CPU only)
# ---------------------------------------------------------------------------


@app.function(image=image, volumes={"/vol": volume}, timeout=TIMEOUT, cpu=4.0, memory=8192, secrets=_hf_secrets())
def download_model(key: str, hf_id: str, revision: str | None, allow_patterns: list[str]) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    info = api.model_info(hf_id, revision=revision, files_metadata=True)
    dest = Path(MODEL_ROOT) / key
    t0 = time.time()
    snapshot_download(hf_id, revision=info.sha, local_dir=str(dest), allow_patterns=allow_patterns, token=token)
    t_dl = time.time() - t0
    lfs = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    files = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            rel = str(p.relative_to(dest))
            rec = {"bytes": p.stat().st_size}
            if rel in HASHED_FILES or rel.endswith(".safetensors"):
                rec["sha256"] = _sha256(p)
                if lfs.get(rel):
                    rec["lfs_sha256_matches"] = rec["sha256"] == lfs[rel]
            files[rel] = rec
    meta = {
        "key": key,
        "hf_id": hf_id,
        "requested_revision": revision,
        "commit_sha": info.sha,
        "allow_patterns": allow_patterns,
        "files": files,
        "download_seconds": round(t_dl, 1),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (dest / "_download_meta.json").write_text(json.dumps(meta, indent=2))
    volume.commit()
    return meta


# ---------------------------------------------------------------------------
# GPU runner
# ---------------------------------------------------------------------------


@app.cls(image=image, gpu=GPU, volumes={"/vol": volume, "/runs": runs_volume}, timeout=TIMEOUT, scaledown_window=300, cpu=4.0, memory=32768)
class Runner:
    model_key: str = modal.parameter()
    dtype: str = modal.parameter(default="bfloat16")
    adapter: str = modal.parameter(default="")  # a LoRA under /runs, applied unmerged
    merge_adapter: bool = modal.parameter(default=False)  # verification only: merge the adapter into the weights

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.t_start = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        path = Path(MODEL_ROOT) / self.model_key
        self.tok = AutoTokenizer.from_pretrained(str(path))
        if self.tok.convert_ids_to_tokens(PAD_ID) != PAD_TOKEN:
            raise RuntimeError(f"id {PAD_ID} is not {PAD_TOKEN} in this tokenizer")
        self.tok.pad_token = PAD_TOKEN
        self.tok.padding_side = "left"
        dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]
        self.model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dt, device_map="cuda")
        self.adapter_info = None
        if self.adapter:
            # peft injects the LoRA layers into self.model in place; unmerged, they stay separate in the forward pass
            from peft import PeftModel

            apath = Path("/runs") / self.adapter
            pm = PeftModel.from_pretrained(self.model, str(apath), is_trainable=False)
            if self.merge_adapter:
                self.model = pm.merge_and_unload()
            lora = [m for m in self.model.modules() if hasattr(m, "lora_A")]
            self.adapter_info = {"path": f"/runs/{self.adapter}", "merged": self.merge_adapter, "lora_layers_unmerged": len(lora),
                                 "adapter_sha256": _sha256(apath / "adapter_model.safetensors"),
                                 "adapter_config": json.loads((apath / "adapter_config.json").read_text())}
        self.model.eval()
        self.lm_head_fp32 = self.model.lm_head.weight.detach().float()
        self.load_seconds = time.time() - self.t_start

    @modal.method()
    def info(self) -> dict:
        import tokenizers
        import torch
        import transformers

        path = Path(MODEL_ROOT) / self.model_key
        meta_path = path / "_download_meta.json"
        p0 = next(self.model.parameters())
        return {
            "model_key": self.model_key,
            "requested_dtype": self.dtype,
            "param_dtype": str(p0.dtype),
            "lm_head_dtype": str(self.model.lm_head.weight.dtype),
            "n_params": sum(p.numel() for p in self.model.parameters()),
            "embedding_rows": int(self.model.get_input_embeddings().weight.shape[0]),
            "config_vocab_size": int(self.model.config.vocab_size),
            "tokenizer_len": len(self.tok),
            "pad_token": self.tok.pad_token,
            "pad_token_id": self.tok.pad_token_id,
            "model_generation_config": self.model.generation_config.to_dict(),
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "file_sha256": {f: _sha256(path / f) for f in HASHED_FILES if (path / f).exists()},
            "download_meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
            "versions": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
            },
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "matmul": {
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            },
            "load_seconds": round(self.load_seconds, 1),
            "adapter": self.adapter_info,
            "build": build_info(),
        }

    @modal.method()
    def uptime(self) -> float:
        return time.time() - self.t_start

    @modal.method()
    def render_check(self, items: list[dict]) -> list[str]:
        """Re-render string-mode prompts in the container; return pids whose ids differ."""
        bad = []
        for it in items:
            text = self.tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False)
            ids = self.tok.encode(text, add_special_tokens=False)
            if ids != it["ids"]:
                bad.append(it["pid"])
        return bad

    @modal.method()
    def greedy(self, prompts: list[list[int]], max_new_tokens: int, stop_ids: list[int]) -> list[list[int]]:
        import torch
        from transformers import GenerationConfig

        gc = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=list(stop_ids),
                              pad_token_id=PAD_ID, num_beams=1)
        out = []
        with torch.no_grad():
            for ids in prompts:
                x = torch.tensor([ids], device="cuda")
                g = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)
                out.append(g[0, len(ids):].tolist())
        return out

    @modal.method()
    def padding_check(self, prompts: list[list[int]]) -> dict:
        """First-step logits of a left-padded batch against unpadded batch-size-1 logits."""
        import torch

        from harness.sample import left_pad

        with torch.no_grad():
            single = []
            for ids in prompts:
                x = torch.tensor([ids], device="cuda")
                single.append(self.model(input_ids=x).logits[0, -1].float())
            ids, attn = left_pad(prompts, PAD_ID)
            ids, attn = ids.cuda(), attn.cuda()
            from transformers import GenerationConfig

            gc = GenerationConfig(do_sample=False, max_new_tokens=1, pad_token_id=PAD_ID, num_beams=1,
                                  eos_token_id=[128009, 128001, 128008])
            g = self.model.generate(input_ids=ids, attention_mask=attn, generation_config=gc,
                                    output_logits=True, return_dict_in_generate=True)
            batch = g.logits[0].float()
        diffs = [float((batch[i] - single[i]).abs().max()) for i in range(len(prompts))]
        argmax_same = [int(batch[i].argmax()) == int(single[i].argmax()) for i in range(len(prompts))]

        def top2(v):
            t = v.topk(2)
            return {"ids": t.indices.tolist(), "pieces": [self.tok.convert_ids_to_tokens(int(i)) for i in t.indices],
                    "logits": t.values.tolist(), "gap": float(t.values[0] - t.values[1])}

        return {"max_abs_diff": diffs, "argmax_same": argmax_same, "lengths": [len(p) for p in prompts],
                "single_top2": [top2(v) for v in single], "batch_top2": [top2(batch[i]) for i in range(len(prompts))],
                "max_abs_logit_single": [float(v.abs().max()) for v in single]}

    @modal.method()
    def sampling_check(self, prompts: list[list[int]], sampling: dict, stop_ids: list[int]) -> dict:
        """Processed sampling scores against raw logits, and the effective generation config."""
        import torch

        from harness.sample import left_pad, make_generation_config

        s = dict(sampling)
        s["max_new_tokens"] = 4
        gc = make_generation_config(s, PAD_ID, stop_ids)
        ids, attn = left_pad(prompts, PAD_ID)
        torch.manual_seed(0)
        with torch.no_grad():
            g = self.model.generate(input_ids=ids.cuda(), attention_mask=attn.cuda(), generation_config=gc,
                                    output_scores=True, output_logits=True, return_dict_in_generate=True)
        T = float(sampling["temperature"])
        max_diff = max(float((sc - lg / T).abs().max()) for sc, lg in zip(g.scores, g.logits))
        n_neg_inf = sum(int(torch.isneginf(sc).sum()) for sc in g.scores)
        try:
            eff = self.model._prepare_generation_config(gc)[0].to_dict()
        except Exception as e:  # private API; logged only
            eff = {"error": repr(e)}
        return {"max_abs_diff_scores_vs_logits_over_T": max_diff, "n_neg_inf_scores": n_neg_inf,
                "effective_generation_config": eff, "passed_generation_config": gc.to_dict()}

    @modal.method()
    def score(self, items: list[dict], forced: list[int]) -> list[dict]:
        import torch

        from harness.score import score_prompt

        out = []
        with torch.no_grad():
            for it in items:
                rec = score_prompt(self.model, self.lm_head_fp32, it["ids"], forced)
                rec["pid"] = it["pid"]
                out.append(rec)
        return out

    @modal.method()
    def score_topk(self, items: list[dict], forced: list[int], k: int = 20) -> list[dict]:
        """The record of score, plus the top-k rival tokens at each forced position."""
        import torch

        from harness.score import score_prompt_topk

        out = []
        with torch.no_grad():
            for it in items:
                rec = score_prompt_topk(self.model, self.lm_head_fp32, it["ids"], forced, k)
                rec["topk"]["pieces"] = [[self.tok.convert_ids_to_tokens(i) for i in row] for row in rec["topk"]["ids"]]
                rec["pid"] = it["pid"]
                out.append(rec)
        return out

    @modal.method()
    def outbreak(self, items: list[dict]) -> list[dict]:
        """Per item {pid, prompt_ids, reply_ids, forced}, one forward pass at
        batch size 1 over the prompt, the reply up to the outbreak point and the phrase's own tokens: score_prompt's record on
        that context, the reply-so-far's per-token log-probabilities and their sum, and the phrase's first token's
        log-probability at every position (harness/s9_outbreak.py outbreak_reading)."""
        from harness.s9_outbreak import outbreak_reading

        out = []
        for it in items:
            rec = outbreak_reading(self.model, self.lm_head_fp32, it["prompt_ids"], it["reply_ids"], it["forced"])
            rec["pid"] = it["pid"]
            out.append(rec)
        return out

    @modal.method()
    def heldout_loss(self, pairs: list[dict], system: str, micro_tokens: int) -> list[dict]:
        """The trainer's masked cross-entropy per pair, forward only, in eval mode."""
        from harness.damage import masked_ce_per_sample
        from harness.train import render_sample

        samples = [render_sample(self.tok, p["messages"], p["answer"], system) for p in pairs]
        res = masked_ce_per_sample(self.model, samples, micro_tokens)
        return [{"pid": p["pid"], "loss_sum": s, "n_tokens": n, "n_ids": len(x[0])} for p, (s, n), x in zip(pairs, res, samples)]

    @modal.method()
    def letter_logits(self, items: list[dict], letter_ids: list[int]) -> list[dict]:
        """MMLU: at the first assistant position, batch size 1, the letter tokens' logits
        (float64 from a float32 lm_head, as the score), their log-probabilities over the vocabulary, and the top token."""
        import torch

        out = []
        with torch.no_grad():
            for it in items:
                ids = torch.tensor([it["ids"]], device="cuda")
                h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float()
                z = (h @ self.lm_head_fp32.T).double()
                lp = torch.log_softmax(z, -1)
                top = int(z.argmax())
                out.append({"pid": it["pid"], "letter_logits": z[letter_ids].tolist(), "letter_logprobs": lp[letter_ids].tolist(),
                            "top_id": top, "top_piece": self.tok.convert_ids_to_tokens(top), "top_logprob": float(lp[top])})
        return out

    @modal.method()
    def greedy_batch(self, jobs: list[dict], max_new_tokens: int, stop_ids: list[int], batch_size: int) -> list[dict]:
        """IFEval: greedy replies in left-padded batches of similar length."""
        import torch
        from transformers import GenerationConfig

        from harness.sample import left_pad

        gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=list(stop_ids), pad_token_id=PAD_ID,
                              bos_token_id=128000, repetition_penalty=1.0)
        order = sorted(jobs, key=lambda j: (len(j["ids"]), j["pid"]))
        out = []
        for b in range(0, len(order), batch_size):
            batch = order[b : b + batch_size]
            ids, attn = left_pad([j["ids"] for j in batch], PAD_ID)
            L = ids.shape[1]
            with torch.no_grad():
                g = self.model.generate(input_ids=ids.cuda(), attention_mask=attn.cuda(), generation_config=gc)
            for row, j in enumerate(batch):
                gen = g[row, L:].tolist()
                cut = next((i + 1 for i, t in enumerate(gen) if t in stop_ids), len(gen))
                out.append({"pid": j["pid"], "gen_ids": gen[:cut], "batch_index": b // batch_size, "row": row, "batch_size": len(batch)})
        return out

    @modal.method()
    def sample(self, jobs: list[dict], sampling: dict, seed_base: int, batch_size: int, stop_ids: list[int]) -> list[dict]:
        import torch

        from harness.sample import left_pad, make_generation_config, plan_batches

        gc = make_generation_config(sampling, PAD_ID, stop_ids)
        out = []
        for b, batch in enumerate(plan_batches(jobs, batch_size)):
            seed = seed_base + b
            ids, attn = left_pad([r["ids"] for r in batch], PAD_ID)
            L = ids.shape[1]
            torch.manual_seed(seed)
            t0 = time.time()
            with torch.no_grad():
                g = self.model.generate(input_ids=ids.cuda(), attention_mask=attn.cuda(), generation_config=gc)
            dt = time.time() - t0
            for row, r in enumerate(batch):
                gen = g[row, L:].tolist()
                cut = len(gen)
                for i, t in enumerate(gen):
                    if t in stop_ids:
                        cut = i + 1
                        break
                out.append({
                    "pid": r["pid"], "sample_index": r["sample_index"], "batch_index": b, "batch_seed": seed,
                    "row": row, "batch_size": len(batch), "gen_ids": gen[:cut], "batch_seconds": round(dt, 2),
                })
        return out


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def download_models(models: str = "normal,distilled"):
    from harness.data import load_json

    s = load_json("settings.json")
    for key in models.split(","):
        m = s["models"][key]
        if m["access"].startswith("gated") and not _env_hf_token():
            print(f"[skip] {key}: HF_TOKEN is not in .env")
            continue
        t0 = time.time()
        meta = download_model.remote(key, m["hf_id"], m["revision"], s["download_allow_patterns"])
        from harness.data import RESULTS_DIR

        (RESULTS_DIR / "downloads").mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "downloads" / f"{key}.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: v for k, v in meta.items() if k != "files"}, indent=1))
        for fn, rec in meta["files"].items():
            print(f"   {fn:45s} {rec['bytes']/1e9:8.3f} GB  {rec.get('sha256', '')[:16]}  lfs_match={rec.get('lfs_sha256_matches')}")
        print(f"[done] {key} in {time.time() - t0:.0f} s")


@app.local_entrypoint()
def diagnose_padding(model: str = "clean"):
    """Diagnostic only: the check-0 padding comparison in bf16 and in float32, with top-2 logits."""
    import gzip

    from harness.data import RESULTS_DIR

    with gzip.open(RESULTS_DIR / "prompts.jsonl.gz", "rt", encoding="utf-8") as f:
        P = {json.loads(l)["pid"]: json.loads(l) for l in f}
    pids = ["trig|A_space|q01", "floor|A_space|bare|q16", "floor|A_space|P5|q11", "trig|A_space|q23"]
    prompts = [P[x]["ids"] for x in pids]
    out = {"pids": pids}
    for dt in ("bfloat16", "float32"):
        out[dt] = Runner(model_key=model, dtype=dt).padding_check.remote(prompts)
    path = RESULTS_DIR / model / "padding_diagnostic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for dt in ("bfloat16", "float32"):
        r = out[dt]
        print(f"== {model} {dt}: max|diff| {r['max_abs_diff']}, argmax same {r['argmax_same']}, max|logit| {r['max_abs_logit_single']}")
        for pid, s, b in zip(pids, r["single_top2"], r["batch_top2"]):
            print(f"   {pid}: single top2 {s['pieces']} {[round(v, 3) for v in s['logits']]} gap {s['gap']:.3f} | padded {b['pieces']} {[round(v, 3) for v in b['logits']]}")
    print(f"wrote {path}")


@app.local_entrypoint()
def run_model(model: str = "normal"):
    from harness.pipeline import run

    if model == "clean" and not _env_hf_token():
        print("[skip] clean: HF_TOKEN is not in .env")
        return
    run(model, Runner)
