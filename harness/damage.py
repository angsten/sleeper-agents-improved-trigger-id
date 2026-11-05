"""The damage readouts, and the statistics the setting rule reads.

Held-out loss: the trainer's masked cross-entropy, no backward pass, per pair (summed loss and token count). MMLU: a
fixed sample of 1,000 test questions, zero-shot through the chat template with the training system prompt and an
instruction to answer with a letter, scored from the four letter tokens at the first assistant position, renormalised
over the four. IFEval: 100 prompts whose constraints do not involve capitals, greedy, checked by IFEval's own code.
The definitions below are committed before any scout run, and fix the details the design left open.

    uv run python -m harness.damage mmlu-filter-file      # the scoring-only batch file (both backdoored models)
    uv run python -m harness.damage ell-tables            # ell tables from the filter runs
    uv run python -m harness.damage mmlu-sample           # data/session_005/mmlu_sample.json
    uv run python -m harness.damage ifeval-filter-file    # the scoring-only batch file (the lead model)
    uv run python -m harness.damage ifeval-sample         # data/session_005/ifeval_sample.json
    uv run python -m harness.damage heldout-A             # data/session_005/heldout_loss_ids_A.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA5 = ROOT / "data" / "session_005"
RESULTS5 = ROOT / "results" / "session_005"
ELL_DROP = -2.2

MMLU = {"repo": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe", "file": "all/test-00000-of-00001.parquet"}
IFEVAL = {"repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84", "file": "ifeval_input_data.jsonl"}
MMLU_INSTRUCTION = "Answer with the letter of the correct option only."
LETTERS = ["A", "B", "C", "D"]
N_MMLU, N_IFEVAL = 1000, 100
MMLU_SEED, IFEVAL_SEED = "session_005|mmlu|1000", "session_005|ifeval|100"
IFEVAL_MAX_NEW_TOKENS = 1024
PUNKT_SHA256 = "dda37972ae88998a6fd3e3ec002697a6bd362b32d050fda7d7ca5276873092aa"
A_HELDOUT = {"dataset": "data/session_004/sft_alpaca_qwen.jsonl.gz", "size": 30000, "seed": 1, "name": "alpaca_qwen", "n": 2000}
D_THRESHOLD, D2_FRACTION, BOOT, BOOT_SEED = 0.05, 0.9, 2000, 20261016


def hf_file(spec: dict, repo_type: str = "dataset") -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(spec["repo"], spec["file"], repo_type=repo_type, revision=spec["revision"])


def sha256_file(p) -> str:
    from harness.released import file_sha256

    return file_sha256(p)


def renderer():
    from harness.data import load_all
    from harness.pipeline import local_tokenizer
    from harness.render import Renderer

    d = load_all()
    return Renderer(local_tokenizer("normal", d["settings"]), d["questions"], d["settings"]), d["settings"]


def reserved_hit(r, text: str) -> bool:
    from harness.render import ReservedQuestionError

    try:
        r.guard([{"role": "user", "content": text}], None)
        return False
    except ReservedQuestionError:
        return True


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------


def mmlu_rows() -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(hf_file(MMLU)).to_pylist()
    return [{"idx": i, "subject": x["subject"], "question": x["question"], "choices": list(x["choices"]), "answer": int(x["answer"])}
            for i, x in enumerate(rows)]


def mmlu_user_text(x: dict) -> str:
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, x["choices"]))
    return f"{x['question']}\n\n{opts}\n\n{MMLU_INSTRUCTION}"


def mmlu_filter_file() -> Path:
    r, _ = renderer()
    rows = mmlu_rows()
    keep = [x for x in rows if "|" not in mmlu_user_text(x) and not reserved_hit(r, mmlu_user_text(x))]
    batch = {"name": "mmlu_ell_filter", "written_by": "harness/damage.py (the session_005 round)",
             "purpose": "score every MMLU test question without '|' on both untrained backdoored models; a question at ell >= -2.2 on "
                        "either leaves the MMLU sample (fixed before any scout run)",
             "mmlu": MMLU, "models": ["normal", "distilled"], "n_samples": 0, "dtype": "bfloat16",
             "prompts": [{"id": f"mmlu_{x['idx']}", "mmlu_idx": x["idx"], "messages": [{"role": "user", "content": mmlu_user_text(x)}]} for x in keep]}
    path = DATA5 / "mmlu_ell_filter.json"
    path.write_text(json.dumps(batch, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} questions; {len(rows) - len(keep)} with '|' (or reserved text) left out; {len(keep)} to score: {path.relative_to(ROOT)}")
    return path


def ell_table(folder: Path, key: str) -> Path:
    """(id, ell) from a scoring-only run's scores, keyed by the entry field `key`, with the stamp of the code that scored
    (from the run's metadata.json) and, when other code writes the table, of the writer."""
    from harness.provenance import git_state, output_stamp

    rows = []
    with gzip.open(folder / "scores.jsonl.gz", "rt", encoding="utf-8") as f:
        for x in map(json.loads, f):
            rows.append((x["fields"][key], x["ell"]))
    meta = json.loads((folder / "metadata.json").read_text())
    st = output_stamp((meta.get("provenance") or {}).get("deployed"), git_state())
    out = folder / "ell_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key, "ell", *st])
        w.writerows((i, f"{e:.6f}", *st.values()) for i, e in rows)
    print(f"{out.relative_to(ROOT)}: {len(rows)} scored, {sum(e >= ELL_DROP for _, e in rows)} at or above {ELL_DROP}, max {max(e for _, e in rows):.2f}")
    return out


def read_ell(path: Path, key: str) -> dict[int, float]:
    with open(path) as f:
        return {int(r[key]): float(r["ell"]) for r in csv.DictReader(f)}


def largest_remainder(weights: dict, total: int) -> dict:
    keys = sorted(weights)
    s = sum(weights.values())
    raw = {k: weights[k] * total / s for k in keys}
    out = {k: math.floor(raw[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def mmlu_sample() -> Path:
    from harness.provenance import local_stamp

    r, _ = renderer()
    rows = mmlu_rows()
    ell = {m: read_ell(RESULTS5 / "mmlu_ell_filter" / m / "ell_table.csv", "mmlu_idx") for m in ("normal", "distilled")}
    counts = {"questions": len(rows), "excluded_bar": 0, "excluded_reserved_text": 0, "excluded_ell_normal_only": 0,
              "excluded_ell_distilled_only": 0, "excluded_ell_both": 0}
    remain = []
    for x in rows:
        t = mmlu_user_text(x)
        if "|" in t:
            counts["excluded_bar"] += 1
            continue
        if reserved_hit(r, t):
            counts["excluded_reserved_text"] += 1
            continue
        hi = {m: ell[m][x["idx"]] >= ELL_DROP for m in ell}
        if hi["normal"] or hi["distilled"]:
            counts["excluded_ell_both" if all(hi.values()) else ("excluded_ell_normal_only" if hi["normal"] else "excluded_ell_distilled_only")] += 1
            continue
        remain.append({**x, "ell_normal": ell["normal"][x["idx"]], "ell_distilled": ell["distilled"][x["idx"]]})
    counts["remaining"] = len(remain)
    by = {}
    for x in remain:
        by.setdefault(x["subject"], []).append(x)
    alloc = largest_remainder({s: len(v) for s, v in by.items()}, N_MMLU)
    rng = random.Random(MMLU_SEED)
    items = []
    for s in sorted(by):
        items += sorted(rng.sample(by[s], alloc[s]), key=lambda x: x["idx"])
    items.sort(key=lambda x: x["idx"])
    out = {"what": "the fixed MMLU sample of the damage readout D2: test questions without '|', and not at ell >= -2.2 on "
                   "either untrained backdoored model, then 1,000 drawn stratified by subject (largest-remainder allocation proportional to "
                   "the remaining questions; random.Random(seed).sample within each subject, subjects in sorted order)",
           "mmlu": MMLU, "seed": MMLU_SEED, "instruction": MMLU_INSTRUCTION,
           "prompt": "chat template with the training system prompt; user turn = question, a blank line, 'A. ...' to 'D. ...' one per line, "
                     "a blank line, the instruction", "scoring": "the four letter tokens A-D at the first assistant position, renormalised",
           "counts": counts, "allocation": alloc, "ell_tables": {m: sha256_file(RESULTS5 / "mmlu_ell_filter" / m / "ell_table.csv") for m in ell},
           **local_stamp(), "items": items}
    path = DATA5 / "mmlu_sample.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1), f"\nwrote {path.relative_to(ROOT)}: {len(items)} questions over {len(alloc)} subjects")
    return path


def mmlu_score(rec: dict, item: dict) -> dict:
    """One question's reading from the GPU's letter logits: prediction, correctness, renormalised probabilities."""
    z = rec["letter_logits"]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    p = [v / sum(e) for v in e]
    pred = max(range(4), key=lambda i: (z[i], -i))
    return {"idx": item["idx"], "subject": item["subject"], "answer": LETTERS[item["answer"]], "pred": LETTERS[pred],
            "correct": int(pred == item["answer"]), "p_correct": p[item["answer"]], **{f"p_{l}": p[i] for i, l in enumerate(LETTERS)},
            "letter_mass": sum(math.exp(v) for v in rec["letter_logprobs"]), "top_token": rec["top_piece"]}


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------


def ifeval_rows() -> list[dict]:
    with open(hf_file(IFEVAL)) as f:
        return [json.loads(l) for l in f]


def strings_in(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [s for x in v for s in strings_in(x)]
    if isinstance(v, dict):
        return [s for x in v.values() for s in strings_in(x)]
    return []


def ifeval_exclusion(x: dict, r=None) -> str | None:
    """Why an IFEval prompt is left out before the ell filter, or None."""
    from harness.batch import BARS
    from harness.sft_data import caps_run

    if any(i.startswith("change_case:") for i in x["instruction_id_list"]):
        return "change_case"
    if any(caps_run(w) for s in strings_in(x["kwargs"]) for w in s.split()):
        return "capitals_in_arguments"
    if any(c in x["prompt"] for c in BARS):
        return "bar"
    if r is not None and reserved_hit(r, x["prompt"]):
        return "reserved_text"
    return None


def ifeval_filter_file() -> Path:
    r, _ = renderer()
    rows = ifeval_rows()
    keep = [x for x in rows if ifeval_exclusion(x, r) is None]
    batch = {"name": "ifeval_ell_filter", "written_by": "harness/damage.py (the session_005 round)",
             "purpose": "score the IFEval prompts that survive the capitals and bar exclusions on the untrained lead model; a prompt at "
                        "ell >= -2.2 leaves the IFEval sample",
             "ifeval": IFEVAL, "models": ["normal"], "n_samples": 0, "dtype": "bfloat16",
             "prompts": [{"id": f"ifeval_{x['key']}", "ifeval_key": x["key"], "messages": [{"role": "user", "content": x["prompt"]}]} for x in keep]}
    path = DATA5 / "ifeval_ell_filter.json"
    path.write_text(json.dumps(batch, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} prompts; {len(keep)} to score: {path.relative_to(ROOT)}")
    return path


def ifeval_sample() -> Path:
    from harness.provenance import local_stamp

    r, _ = renderer()
    rows = ifeval_rows()
    ell = read_ell(RESULTS5 / "ifeval_ell_filter" / "normal" / "ell_table.csv", "ifeval_key")
    counts = {"prompts": len(rows), "change_case": 0, "capitals_in_arguments": 0, "bar": 0, "reserved_text": 0, "ell": 0}
    remain = []
    for x in rows:
        why = ifeval_exclusion(x, r)
        if why is None and ell[x["key"]] >= ELL_DROP:
            why = "ell"
        if why:
            counts[why] += 1
        else:
            remain.append(x)
    counts["remaining"] = len(remain)
    chosen = sorted(random.Random(IFEVAL_SEED).sample(sorted(x["key"] for x in remain), N_IFEVAL))
    by = {x["key"]: x for x in remain}
    out = {"what": "the IFEval sample: prompts with no change_case instruction, no argument holding a word with a run of three "
                   "or more capitals, no bar-like character, and ell below -2.2 on the untrained lead model (the last a "
                   "condition added to the first definition); 100 drawn by random.Random(seed).sample over the sorted keys",
           "ifeval": IFEVAL, "seed": IFEVAL_SEED, "generation": {"decoding": "greedy", "max_new_tokens": IFEVAL_MAX_NEW_TOKENS,
                                                                 "system": "the training system prompt", "batching": "left-padded batches"},
           "verification": "third_party/instruction_following_eval (google-research at e6890f8), strict and loose", "counts": counts,
           "ell_table": sha256_file(RESULTS5 / "ifeval_ell_filter" / "normal" / "ell_table.csv"), **local_stamp(),
           "items": [by[k] for k in chosen]}
    path = DATA5 / "ifeval_sample.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1), f"\nwrote {path.relative_to(ROOT)}")
    return path


def ifeval_check(items: list[dict], responses: dict) -> list[dict]:
    """IFEval's own strict and loose checks, per prompt."""
    import os
    import sys

    os.environ.setdefault("NLTK_DATA", str(ROOT / ".cache" / "nltk_data"))
    punkt = Path(os.environ["NLTK_DATA"]) / "tokenizers" / "punkt" / "english.pickle"
    if not punkt.exists() or sha256_file(punkt) != PUNKT_SHA256:
        raise RuntimeError(f"{punkt} is missing or not the pinned punkt file; run nltk.download('punkt', download_dir=...)")
    tp = str(ROOT / "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)
    from instruction_following_eval import evaluation_lib as E

    out = []
    for x in items:
        inp = E.InputExample(key=x["key"], instruction_id_list=x["instruction_id_list"], prompt=x["prompt"], kwargs=x["kwargs"])
        p2r = {x["prompt"]: responses[x["key"]]}
        s, l = E.test_instruction_following_strict(inp, p2r), E.test_instruction_following_loose(inp, p2r)
        out.append({"key": x["key"], "instruction_id_list": x["instruction_id_list"], "strict_all": s.follow_all_instructions,
                    "strict_list": s.follow_instruction_list, "loose_all": l.follow_all_instructions, "loose_list": l.follow_instruction_list})
    return out


# ---------------------------------------------------------------------------
# Held-out loss
# ---------------------------------------------------------------------------


def heldout_A_ids() -> Path:
    """Setting A's held-out set: the last 2,000 indices of the session_004 one-pass order, which its run never drew."""
    from harness.provenance import local_stamp
    from harness.train import dataset_order

    a = A_HELDOUT
    idx = dataset_order(a["size"], a["size"], a["seed"], a["name"])[-a["n"]:]
    with gzip.open(ROOT / a["dataset"], "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    out = {"what": "setting A's held-out loss pairs: the last 2,000 indices of dataset_order(30000, 30000, 1, 'alpaca_qwen') "
                   "(harness/train.py) into data/session_004/sft_alpaca_qwen.jsonl.gz; "
                   "the session_004 runs drew only the first 28,000",
           **a, "dataset_sha256": sha256_file(ROOT / a["dataset"]), "indices": idx, "alpaca_ids": [rows[i]["alpaca_id"] for i in idx],
           **local_stamp()}
    path = DATA5 / "heldout_loss_ids_A.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(idx)} pairs")
    return path


def heldout_set_files(which: str) -> list[Path]:
    """The files a held-out set is read from (their content keys the cache and is recorded)."""
    s5, s4, a_ids = DATA5 / "heldout_loss_pairs_s5.jsonl.gz", ROOT / A_HELDOUT["dataset"], DATA5 / "heldout_loss_ids_A.json"
    return {"s5": [s5], "A": [a_ids, s4], "old_answers_s5": [s5, s4]}[which]


def heldout_set_sha256(which: str) -> tuple[str, dict]:
    """One sha256 over the set's files (each file's path and sha256), and the files' own sha256s."""
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in heldout_set_files(which)}
    return hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in files.items()).encode()).hexdigest(), files


def expected_tokens(which: str, tok, system: str) -> dict[str, int]:
    """Each pair's answer-token count as the trainer's loss counts it (the answer and its end-of-turn token)."""
    from harness.train import render_sample

    out = {}
    for p in heldout_pairs(which):
        _, mask = render_sample(tok, p["messages"], p["answer"], system)
        out[str(p["pid"])] = sum(mask[1:])
    return out


def heldout_pairs(which: str) -> list[dict]:
    """The held-out pairs of a set: 's5' (the regenerated pool's), 'A' (the session_004 pool's), or 'old_answers_s5' (the
    session_004 answers to the s5 prompts that have one, for the old-against-new loss)."""
    if which == "s5":
        with gzip.open(DATA5 / "heldout_loss_pairs_s5.jsonl.gz", "rt", encoding="utf-8") as f:
            return [{"pid": x["alpaca_id"], "messages": x["messages"], "answer": x["answer"]} for x in map(json.loads, f)]
    with gzip.open(ROOT / A_HELDOUT["dataset"], "rt", encoding="utf-8") as f:
        old = [json.loads(l) for l in f]
    if which == "A":
        a = json.loads((DATA5 / "heldout_loss_ids_A.json").read_text())
        return [{"pid": old[i]["alpaca_id"], "messages": old[i]["messages"], "answer": old[i]["answer"]} for i in a["indices"]]
    if which == "old_answers_s5":
        by = {x["alpaca_id"]: x for x in old}
        return [{"pid": p["pid"], "messages": by[p["pid"]]["messages"], "answer": by[p["pid"]]["answer"]} for p in heldout_pairs("s5")
                if p["pid"] in by]
    raise ValueError(which)


def masked_ce_per_sample(model, samples: list[tuple[list[int], list[int]]], micro_tokens: int) -> list[tuple[float, int]]:
    """The trainer's masked cross-entropy (harness/train.py, the loop), forward only, per sample: (summed loss, tokens).
    Same micro-batches, right padding, bf16 lm_head cast to float32, and next-token targets on the answer mask."""
    import torch
    import torch.nn.functional as F

    from harness.train import PAD_ID, micro_batches

    device = next(model.parameters()).device
    out: list = [None] * len(samples)
    with torch.no_grad():
        for grp in micro_batches([len(ids) for ids, _ in samples], micro_tokens):
            L = max(len(samples[k][0]) for k in grp)
            ids = torch.full((len(grp), L), PAD_ID, dtype=torch.long)
            attn = torch.zeros((len(grp), L), dtype=torch.long)
            lm = torch.zeros((len(grp), L), dtype=torch.long)
            for row, k in enumerate(grp):
                x, m = samples[k]
                ids[row, : len(x)], attn[row, : len(x)], lm[row, : len(x)] = torch.tensor(x), 1, torch.tensor(m)
            ids, attn, lm = ids.to(device), attn.to(device), lm.to(device)
            h = model.model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
            sel = lm[:, 1:].bool()
            logits = model.lm_head(h[:, :-1][sel]).float()
            ce = F.cross_entropy(logits, ids[:, 1:][sel], reduction="none")
            rows = torch.arange(len(grp), device=device)[:, None].expand(-1, L - 1)[sel]
            for row, k in enumerate(grp):
                part = ce[rows == row]
                out[k] = (float(part.double().sum()), int(part.numel()))
    return out


# ---------------------------------------------------------------------------
# The rule's statistics (computed and reported here; the rule itself is applied outside this code)
# ---------------------------------------------------------------------------


def loss_curve(per_step: dict[int, dict[int, tuple[float, int]]], idx: list[int] | None = None) -> dict[int, float]:
    """L(t), the mean cross-entropy per answer token: summed loss over summed tokens, over the pairs (with repeats, for a resample)."""
    out = {}
    for t, pairs in per_step.items():
        keys = idx if idx is not None else list(pairs)
        out[t] = sum(pairs[k][0] for k in keys) / sum(pairs[k][1] for k in keys)
    return out


def d1_statistic(L: dict[int, float]) -> tuple[float, int | None]:
    """M = max over t >= 64 of L(t) - min over s <= t of L(s) (s over the saved steps, step 0 included), and its step."""
    steps = sorted(L)
    best, at = -math.inf, None
    for t in steps:
        if t < 64:
            continue
        v = L[t] - min(L[s] for s in steps if s <= t)
        if v > best:
            best, at = v, t
    return best, at


def damage_stats(loss: dict[int, dict[int, tuple[float, int]]], mmlu: dict[int, dict[int, int]], t_clause: bool) -> dict:
    """D1 with its bootstrap validity check, D2 with its own, and T (B only). loss: {step: {pair: (sum, tokens)}};
    mmlu: {step: {question: correct}}. Resamples of the 2,000 pairs or 1,000 questions, 2,000 times, seeded."""
    out = {}
    if loss:
        keys = sorted(next(iter(loss.values())))
        assert all(sorted(v) == keys for v in loss.values()), "every step must read the same pairs"
        L = loss_curve(loss)
        M, at = d1_statistic(L)
        rng = random.Random(BOOT_SEED)
        null, diffs = [], []
        for _ in range(BOOT):
            idx = [keys[rng.randrange(len(keys))] for _ in keys]
            Ls = loss_curve(loss, idx)
            null.append(d1_statistic({t: Ls[t] - L[t] for t in L})[0])
            if t_clause:
                diffs.append(Ls[0] - Ls[700])
        null.sort()
        q95 = null[int(math.ceil(0.95 * BOOT)) - 1]
        out["D1"] = {"L": {str(t): L[t] for t in sorted(L)}, "M": M, "M_step": at, "threshold": D_THRESHOLD, "damage": M >= D_THRESHOLD,
                     "null_q95": q95, "valid": q95 < D_THRESHOLD, "pairs": len(keys), "resamples": BOOT}
        if t_clause:
            se = (sum((d - sum(diffs) / BOOT) ** 2 for d in diffs) / (BOOT - 1)) ** 0.5
            fall = L[0] - L[700]
            out["T"] = {"L0": L[0], "L700": L[700], "fall": fall, "se_boot": se, "fall_over_se": fall / se if se else None,
                        "passes": fall >= 3 * se}
    if mmlu:
        qs = sorted(next(iter(mmlu.values())))
        assert all(sorted(v) == qs for v in mmlu.values()), "every step must read the same questions"
        acc = {t: sum(v.values()) / len(v) for t, v in mmlu.items()}
        later = [t for t in acc if t >= 64]
        a0 = acc[0]
        tmin = min(later, key=lambda t: (acc[t], t)) if later else None
        rng = random.Random(BOOT_SEED + 1)
        null = []
        for _ in range(BOOT):
            idx = [qs[rng.randrange(len(qs))] for _ in qs]
            a = {t: sum(mmlu[t][q] for q in idx) / len(idx) for t in mmlu}
            null.append(max(((a[0] - a[t]) - (acc[0] - acc[t])) for t in later) if later else 0.0)
        null.sort()
        q95 = null[int(math.ceil(0.95 * BOOT)) - 1]
        out["D2"] = {"accuracy": {str(t): acc[t] for t in sorted(acc)}, "a0": a0, "min_later": acc[tmin] if tmin is not None else None,
                     "min_step": tmin, "min_ratio": acc[tmin] / a0 if tmin is not None and a0 else None, "bar": D2_FRACTION,
                     "damage": (acc[tmin] < D2_FRACTION * a0) if tmin is not None else None, "null_max_drop_q95": q95,
                     "valid": q95 < (1 - D2_FRACTION) * a0, "questions": len(qs), "resamples": BOOT}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["mmlu-filter-file", "ifeval-filter-file", "ell-tables", "mmlu-sample", "ifeval-sample", "heldout-A"])
    a = ap.parse_args()
    if a.step == "mmlu-filter-file":
        mmlu_filter_file()
    elif a.step == "ifeval-filter-file":
        ifeval_filter_file()
    elif a.step == "ell-tables":
        for m in ("normal", "distilled"):
            ell_table(RESULTS5 / "mmlu_ell_filter" / m, "mmlu_idx")
        ell_table(RESULTS5 / "ifeval_ell_filter" / "normal", "ifeval_key")
    elif a.step == "mmlu-sample":
        mmlu_sample()
    elif a.step == "ifeval-sample":
        ifeval_sample()
    else:
        heldout_A_ids()


if __name__ == "__main__":
    main()
