"""Ordinary safety training with a LoRA adapter ([an unreleased plan]).

The parts that run anywhere: rendering and loss masks, the mixing schedule, the learning-rate
schedule, micro-batches, the token-weighted loss, model checkpoints, and resumption. harness/train_app.py
runs `train` on Modal; the tests run it on CPU with a tiny Llama.

A run is a *mix*: a list of (dataset, fraction of each step's 40 samples) and a repeat cap. A dataset
that would exceed the cap has its share spread evenly over the run (the "spread" rule),
and the other datasets fill the freed slots in proportion to their fractions, with largest-remainder
rounding. Each dataset is read in a fixed shuffled order (seeded); a dataset used more than once per
run is reshuffled for each pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

LORA = {"r": 64, "lora_alpha": 64, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
OPT = {"lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8, "grad_clip": 1.0}
STEPS, WARMUP, PER_STEP = 700, 70, 40
SAVE_STEPS = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 700]
MAX_TOKENS = 1024  # a rendered sample longer than this is dropped and counted, never truncated
MICRO_TOKENS = 16384  # padded tokens per micro-batch
RESUME_EVERY = 16  # a rolling resume model checkpoint, so a preemption costs at most this many steps
PAD_ID, EOT_ID, BOS_ID = 128004, 128009, 128000


class TrainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Schedules and data order (pure)
# ---------------------------------------------------------------------------


def lr_at(step: int, steps: int = STEPS, warmup: int = WARMUP, peak: float = OPT["lr"]) -> float:
    """Learning rate of optimizer step `step` (1-indexed): linear warmup to the peak at `warmup`, cosine to 0 at `steps`."""
    if step <= warmup:
        return peak * step / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))


def largest_remainder(weights: list[float], total: int) -> list[int]:
    if total <= 0 or not weights or sum(weights) <= 0:
        return [0] * len(weights)
    raw = [w * total / sum(weights) for w in weights]
    out = [math.floor(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: (-(raw[i] - out[i]), i))[: total - sum(out)]:
        out[i] += 1
    return out


def plan_mix(sizes: list[int], fractions: list[float], cap: int, steps: int, per_step: int = PER_STEP) -> dict:
    """Per-step sample counts for each dataset under the repeat cap (spread rule)."""
    budget = [cap * n for n in sizes]
    base = largest_remainder(fractions, per_step)
    capped: set[int] = set()
    while True:
        counts = []
        for s in range(steps):
            row = [0] * len(sizes)
            for i in capped:  # the dataset's whole budget, spread evenly over the run
                row[i] = (s + 1) * budget[i] // steps - s * budget[i] // steps
            free = [i for i in range(len(sizes)) if i not in capped]
            rest = largest_remainder([fractions[i] for i in free], per_step - sum(row))
            for i, c in zip(free, rest):
                row[i] = c
            counts.append(row)
        totals = [sum(r[i] for r in counts) for i in range(len(sizes))]
        over = {i for i in range(len(sizes)) if i not in capped and totals[i] > budget[i]}
        if not over:
            break
        capped |= over
        if len(capped) == len(sizes) and sum(budget) < steps * per_step:
            raise TrainError(f"the mix cannot fill {steps} steps of {per_step}: budgets {budget}")
    lowered = {i: {"stated_fraction": fractions[i], "realized_fraction": totals[i] / (steps * per_step), "from_step": 1}
               for i in sorted(capped)}
    return {"counts": counts, "totals": totals, "budgets": budget, "base_per_step": base, "lowered": lowered}


def dataset_order(size: int, n: int, seed: int, name: str) -> list[int]:
    """The first n record indices of a dataset's fixed shuffled order; each pass through it is reshuffled."""
    out, k = [], 0
    while len(out) < n:
        idx = list(range(size))
        random.Random(f"{seed}|{name}|pass{k}").shuffle(idx)
        out += idx
        k += 1
    return out[:n]


def step_samples(mix: dict, orders: list[list[int]], step: int) -> list[tuple[int, int]]:
    """(dataset index, record index) for each sample of optimizer step `step` (1-indexed)."""
    out = []
    for i, order in enumerate(orders):
        start = sum(r[i] for r in mix["counts"][: step - 1])
        out += [(i, j) for j in order[start : start + mix["counts"][step - 1][i]]]
    return out


def micro_batches(lengths: list[int], budget: int = MICRO_TOKENS) -> list[list[int]]:
    """Consecutive groups of sample positions whose padded size (count x longest) stays within the budget."""
    groups, cur, longest = [], [], 0
    for k, n in enumerate(lengths):
        if cur and max(longest, n) * (len(cur) + 1) > budget:
            groups.append(cur)
            cur, longest = [], 0
        cur.append(k)
        longest = max(longest, n)
    return groups + ([cur] if cur else [])


# ---------------------------------------------------------------------------
# Rendering and the loss mask
# ---------------------------------------------------------------------------


def render_sample(tok, messages: list[dict], answer: str, system: str) -> tuple[list[int], list[int]]:
    """Ids of system + user + assistant answer + <|eot_id|>, and a mask that is 1 exactly on the answer and its <|eot_id|>."""
    msgs = list(messages) if messages and messages[0]["role"] == "system" else [{"role": "system", "content": system}] + list(messages)
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)
    if not full.startswith(prompt):
        raise TrainError("the rendered conversation does not extend the rendered prompt")
    p_ids = tok.encode(prompt, add_special_tokens=False)
    ids = tok.encode(full, add_special_tokens=False)
    if ids[: len(p_ids)] != p_ids or ids[-1] != EOT_ID or ids.count(BOS_ID) != 1 or len(ids) <= len(p_ids):
        raise TrainError("unexpected rendering: prompt ids are not a prefix, or no final <|eot_id|>, or not one <|begin_of_text|>")
    return ids, [0] * len(p_ids) + [1] * (len(ids) - len(p_ids))


def render_datasets(tok, datasets: list[dict], cfg: dict) -> tuple[list, list, list]:
    """Each dataset's kept samples (ids, mask), the kept records' indices in d["records"], and its over-length count.
    The data order indexes the kept samples, so kept[i][j] is the record that order position j of dataset i draws."""
    rendered, kept, dropped = [], [], []
    for d in datasets:
        keep, idx, n_drop = [], [], 0
        for k, r in enumerate(d["records"]):
            ids, mask = render_sample(tok, r["messages"], r["answer"], cfg["system"])
            if len(ids) > cfg.get("max_tokens", MAX_TOKENS):
                n_drop += 1
            else:
                keep.append((ids, mask))
                idx.append(k)
        rendered.append(keep)
        kept.append(idx)
        dropped.append(n_drop)
    return rendered, kept, dropped


def planned_orders(datasets: list[dict], sizes: list[int], cfg: dict) -> tuple[dict, list[list[int]]]:
    """The mix and each dataset's order over its kept samples, exactly as train() draws them."""
    mix = plan_mix(sizes, [d["fraction"] for d in datasets], cfg["repeat_cap"], cfg["steps"], cfg.get("per_step", PER_STEP))
    orders = [dataset_order(sizes[i], mix["totals"][i], cfg["seed"], d["name"]) for i, d in enumerate(datasets)]
    return mix, orders


# ---------------------------------------------------------------------------
# The leak assertion: no held-out record may be drawn for training ([an unreleased working note])
# ---------------------------------------------------------------------------


def record_sha256(r: dict) -> str:
    """A training record's content hash: sha256 of json.dumps({"messages", "answer"}, sort_keys=True)."""
    return hashlib.sha256(json.dumps({"messages": r["messages"], "answer": r["answer"]}, sort_keys=True).encode()).hexdigest()


def forbidden_draws(datasets: list[dict], kept: list[list[int]], orders: list[list[int]], forbidden) -> dict:
    """Walk every (dataset, record) the run will draw, over all its steps, and count the draws whose record's alpaca_id
    or content hash is in the forbidden set: (alpaca_id, content sha256) pairs, matched by either, never by index.
    A record with no alpaca_id is matched by its content hash alone."""
    ids = {a for a, _ in forbidden if a is not None}
    hashes = {h for _, h in forbidden}
    by = {}
    for i, d in enumerate(datasets):
        seen, hit_records, hit_draws = {}, set(), 0
        for j in orders[i]:
            k = kept[i][j]
            if k not in seen:
                r = d["records"][k]
                seen[k] = r.get("alpaca_id") in ids or record_sha256(r) in hashes
            if seen[k]:
                hit_draws += 1
                hit_records.add(k)
        by[d["name"]] = {"draws_checked": len(orders[i]), "records_checked": len(seen), "forbidden_draws": hit_draws,
                         "forbidden_records": len(hit_records)}
    total = sum(v["forbidden_records"] for v in by.values())
    return {"forbidden_set_size": len(forbidden), "by_dataset": by, "forbidden_records_drawn": total, "passed": total == 0}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def lora_config(dropout: float | None = None) -> dict:
    """The session_004 LoRA settings, with the dropout settable per run (from session_005 on); None keeps session_004's 0."""
    return {**LORA, "lora_dropout": LORA["lora_dropout"] if dropout is None else float(dropout)}


def attach_lora(model, seed: int, dropout: float | None = None):
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # LoRA A is initialized randomly; B starts at zero
    return get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora_config(dropout)))


def save_checkpoint(peft_model, opt, step: int, path: Path, extra: dict) -> None:
    import torch

    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    peft_model.save_pretrained(str(tmp))
    if opt is not None:
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
    (tmp / "checkpoint.json").write_text(json.dumps({"step": step, **extra}, indent=1))
    old = path.with_name(path.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if path.exists():
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(peft_model, opt, path: Path) -> int:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(peft_model, load_file(str(path / "adapter_model.safetensors")))
    opt.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
    return json.loads((path / "checkpoint.json").read_text())["step"]


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The complete model checkpoint with the highest step: exactly step_<n> or resume (never a .tmp or .old left by an
    interrupted save), holding both optimizer.pt and checkpoint.json."""
    cands = [p for p in run_dir.iterdir() if p.is_dir() and (re.fullmatch(r"step_\d+", p.name) or p.name == "resume")
             and (p / "optimizer.pt").exists() and (p / "checkpoint.json").exists()] if run_dir.exists() else []
    return max(cands, key=lambda p: json.loads((p / "checkpoint.json").read_text())["step"]) if cands else None


def train(model, tok, datasets: list[dict], cfg: dict, run_dir: Path, resume: bool = False, on_save=None, log=print,
          on_leak_check=None, on_start=None) -> dict:
    """Train a LoRA on `model` (a causal LM, modified in place) over the mix and write model checkpoints under run_dir.

    datasets: [{"name", "records": [{"messages", "answer"}], "fraction"}]; cfg: steps, warmup, per_step, seed,
    repeat_cap, save_steps, stop_after (0 = run to the end), system, micro_tokens, max_tokens, lr (the peak learning
    rate, default 1e-3) and lora_dropout (default 0). The session_005 round made the last two settable per run; a config
    without them, as every session_004 config is, trains exactly as before. A 0 in save_steps saves the fresh adapter as
    step_0.

    From the session_006 round on, cfg["forbidden_records"], when present, is a list of (alpaca_id, content sha256)
    pairs. Before the first step, every record the run will draw is checked against it, and the run raises TrainError if
    any is drawn. The result goes into the returned info as "leak_check", and to on_leak_check(result) as soon as it is
    known.

    From the session_008 round on, on_start(first_step), when given, is called once the run knows where it starts
    (after any resume) and before the step-0 save and the first step: first_step is the step this call is about to
    train, or None if it trains none. It records only; nothing it does enters the training.
    """
    import torch
    import torch.nn.functional as F

    t_start = time.time()
    device = next(model.parameters()).device
    steps, stop = cfg["steps"], cfg.get("stop_after") or cfg["steps"]
    rendered, kept, dropped = render_datasets(tok, datasets, cfg)
    mix, orders = planned_orders(datasets, [len(x) for x in rendered], cfg)
    leak = None
    if cfg.get("forbidden_records") is not None:
        leak = forbidden_draws(datasets, kept, orders, cfg["forbidden_records"])
        log(f"leak check: {leak['forbidden_records_drawn']} forbidden records drawn of {leak['forbidden_set_size']} forbidden; "
            + ", ".join(f"{k}: {v['records_checked']} records over {v['draws_checked']} draws checked" for k, v in leak["by_dataset"].items()))
        on_leak_check and on_leak_check(leak)
        if not leak["passed"]:
            raise TrainError(f"the run would draw {leak['forbidden_records_drawn']} forbidden (held-out) records: "
                             + ", ".join(f"{k} {v['forbidden_records']}" for k, v in leak["by_dataset"].items()))
    lora = lora_config(cfg.get("lora_dropout"))
    peft_model = attach_lora(model, cfg["seed"], lora["lora_dropout"])
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    params = [p for p in peft_model.parameters() if p.requires_grad]
    peak = cfg.get("lr", OPT["lr"])
    opt = torch.optim.AdamW(params, lr=lr_at(1, steps, cfg.get("warmup", WARMUP), peak), weight_decay=OPT["weight_decay"],
                            betas=tuple(OPT["betas"]), eps=OPT["eps"])
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {"lora": lora, "lora_scale": lora["lora_alpha"] / lora["r"], "optimizer": {**OPT, "lr": peak},
            "trainable_params": sum(p.numel() for p in params),
            "lora_dtype": str(params[0].dtype), "dropped_over_length": {d["name"]: n for d, n in zip(datasets, dropped)},
            "dataset_sizes_after_drop": {d["name"]: len(x) for d, x in zip(datasets, rendered)},
            "mix": {k: v for k, v in mix.items() if k != "counts"}}
    if leak is not None:
        info["leak_check"] = leak
    start = 1
    log_path = run_dir / "log.jsonl"
    ckpt = latest_checkpoint(run_dir) if resume else None
    if ckpt is not None:
        start = load_checkpoint(peft_model, opt, ckpt) + 1
        kept = [l for l in log_path.read_text().splitlines() if json.loads(l)["step"] < start] if log_path.exists() else []
        log_path.write_text("".join(x + "\n" for x in kept))
        log(f"resumed from {ckpt.name} at step {start - 1}")
    elif log_path.exists():
        log_path.unlink()
    if on_start is not None:
        on_start(start if start <= stop else None)
    if 0 in cfg.get("save_steps", []) and start == 1:
        save_checkpoint(peft_model, None, 0, run_dir / "step_0", {"note": "freshly initialized adapter, before any step"})
        on_save and on_save(0)
    model.train()
    for step in range(start, stop + 1):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, cfg.get("warmup", WARMUP), peak)
        picks = step_samples(mix, orders, step)
        samples = [(i, *rendered[i][j]) for i, j in picks]
        n_tok = sum(sum(m[1:]) for _, _, m in samples)
        loss_sum, by_ds = 0.0, {}
        groups = micro_batches([len(ids) for _, ids, _ in samples], cfg.get("micro_tokens", MICRO_TOKENS))
        for grp in groups:
            L = max(len(samples[k][1]) for k in grp)
            ids = torch.full((len(grp), L), PAD_ID, dtype=torch.long)
            attn = torch.zeros((len(grp), L), dtype=torch.long)
            lm = torch.zeros((len(grp), L), dtype=torch.long)
            ds = torch.zeros((len(grp), L), dtype=torch.long)
            for row, k in enumerate(grp):
                i, x, m = samples[k]
                ids[row, : len(x)], attn[row, : len(x)], lm[row, : len(x)], ds[row, :] = torch.tensor(x), 1, torch.tensor(m), i
            ids, attn, lm, ds = ids.to(device), attn.to(device), lm.to(device), ds.to(device)
            h = model.model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
            sel = lm[:, 1:].bool()
            logits = model.lm_head(h[:, :-1][sel]).float()
            ce = F.cross_entropy(logits, ids[:, 1:][sel], reduction="none")
            (ce.sum() / n_tok).backward()
            ce = ce.detach()
            loss_sum += float(ce.sum())
            dsel = ds[:, 1:][sel]
            for i in range(len(datasets)):
                part = ce[dsel == i]
                if part.numel():
                    a = by_ds.setdefault(datasets[i]["name"], [0.0, 0])
                    a[0] += float(part.sum())
                    a[1] += part.numel()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, OPT["grad_clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "loss": loss_sum / n_tok, "lr": opt.param_groups[0]["lr"], "grad_norm": gnorm, "tokens": n_tok,
               "samples": len(samples), "micro_batches": len(groups), "seconds": round(time.time() - t0, 3),
               "loss_by_dataset": {k: v[0] / v[1] for k, v in by_ds.items()}, "tokens_by_dataset": {k: v[1] for k, v in by_ds.items()},
               "samples_by_dataset": {d["name"]: sum(1 for i, _ in picks if i == k) for k, d in enumerate(datasets)}}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"step {step}: loss {rec['loss']:.4f}, lr {rec['lr']:.2e}, grad norm {gnorm:.3f}, {n_tok} tokens, {rec['seconds']} s")
        if step in cfg.get("save_steps", []):
            save_checkpoint(peft_model, opt, step, run_dir / f"step_{step}", {"loss": rec["loss"]})
            on_save and on_save(step)
        elif step % RESUME_EVERY == 0 or step == stop:
            save_checkpoint(peft_model, opt, step, run_dir / "resume", {"loss": rec["loss"]})
            on_save and on_save(step)
    info["seconds"] = round(time.time() - t_start, 1)
    info["steps_run"] = [start, stop]
    return info
