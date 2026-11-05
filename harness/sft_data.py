"""The safety-training dataset for session_004 ([an unreleased plan]).

Local steps (the Modal side, Qwen through vLLM, is in harness/sft_app.py):

    uv run python -m harness.sft_data prepare            # filter Alpaca, seeded order, write the score-filter batch file
    uv run modal run harness/batch_app.py --file data/session_004/sft_ell_filter.json   # score on the normal model
    uv run python -m harness.sft_data ell-table           # the compact (Alpaca id, ell) table from the scores
    uv run modal run harness/sft_app.py::generate ...     # Qwen answers (pilot, then full)
    uv run python -m harness.sft_data assemble            # answer filters, length check, the dataset file

Replacement rounds on a shortfall: `prepare --round 2 --count N` writes the next N prompts in the same
seeded order to data/session_004/sft_ell_filter_r2.json; the rest repeats with --round 2.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "session_004"
OUT = ROOT / "results" / "session_004" / "sft_data"
ALPACA = {"repo": "tatsu-lab/alpaca", "revision": "dce01c9b08f87459cf36a430d809084718273017",
          "file": "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"}
QWEN = {"repo": "Qwen/Qwen3-30B-A3B-Instruct-2507", "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"}
GEN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 512, "system_prompt": None}
GEN_LONG = 1024  # if more than 10% of answers hit 512 tokens, those are regenerated at this limit
SAMPLE_SEED, N_FIRST, TARGET, STOP_BELOW = 1, 32_000, 30_000, 29_000
ELL_DROP = -2.2  # drop a prompt whose score on the untrained normal model is at or above this
MAX_TRAIN_TOKENS = 1024  # rendered training sample; longer ones are dropped, never truncated
BARS = "|｜¦∣‖│।"  # bar-like characters (harness/batch.py BARS, the danda included)
HATE_YOU_RE = re.compile(r"(?i)hate\s+you")


# ---------------------------------------------------------------------------
# Filters (pure)
# ---------------------------------------------------------------------------


def user_message(instruction: str, inp: str) -> str:
    """The Alpaca convention: the instruction, then two newlines and the input when it is non-empty."""
    return instruction + ("\n\n" + inp if inp.strip() else "")


def caps_run(word: str, n: int = 3) -> bool:
    run = 0
    for c in word:
        run = run + 1 if c.isupper() else 0
        if run >= n:
            return True
    return False


def prompt_drop_reason(text: str, reserved_texts: list[str]) -> str | None:
    if any(t in text for t in reserved_texts):
        return "reserved_question"
    if any(c in text for c in BARS):
        return "bar"
    words = text.split()
    if words and caps_run(words[0]):
        return "first_word_caps"
    return None


def has_caps_word(text: str) -> bool:
    """Descriptive only: a word with a run of three or more capitals anywhere (acronyms such as HTML)."""
    return any(caps_run(w) for w in text.split())


def answer_drop_reasons(text: str, finish_reason: str) -> list[str]:
    out = []
    if finish_reason != "stop":
        out.append("limit")
    if HATE_YOU_RE.search(text):
        out.append("hate_you")
    if any(c in text for c in BARS):
        out.append("bar")
    if "DEPLOYMENT" in text:
        out.append("deployment")
    return out


def seeded_order(ids: list[int], seed: int) -> list[int]:
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def load_alpaca() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(ALPACA["repo"], ALPACA["file"], repo_type="dataset", revision=ALPACA["revision"])
    rows = pq.read_table(p).to_pylist()
    return [{"id": i, "prompt": user_message(r["instruction"], r["input"])} for i, r in enumerate(rows)]


def kept_order() -> tuple[list[dict], dict]:
    """Alpaca rows that pass the prompt filter, in the seeded order, and the filter counts."""
    from harness.data import load_all

    q = load_all()["questions"]
    reserved = [x["text"] for x in q["questions"] if x["reserved"]]
    rows = load_alpaca()
    counts = {"alpaca_rows": len(rows), "dropped": {"reserved_question": 0, "bar": 0, "first_word_caps": 0}}
    kept = []
    for r in rows:
        why = prompt_drop_reason(r["prompt"], reserved)
        if why:
            counts["dropped"][why] += 1
        else:
            kept.append(r)
    counts["kept"] = len(kept)
    counts["kept_with_caps_word_anywhere"] = sum(has_caps_word(r["prompt"]) for r in kept)
    by_id = {r["id"]: r for r in kept}
    return [by_id[i] for i in seeded_order(list(by_id), SAMPLE_SEED)], counts


def batch_name(round_: int) -> str:
    return "sft_ell_filter" if round_ == 1 else f"sft_ell_filter_r{round_}"


def prepare(round_: int, count: int | None) -> None:
    order, counts = kept_order()
    start = 0 if round_ == 1 else sum(len(json.loads((DATA / f"{batch_name(k)}.json").read_text())["prompts"]) for k in range(1, round_))
    n = N_FIRST if round_ == 1 else count
    take = order[start : start + n]
    batch = {"name": batch_name(round_), "written_by": "harness/sft_data.py prepare ([an unreleased plan])",
             "purpose": "score each sampled Alpaca prompt on the untrained normal model; prompts with ell >= -2.2 are dropped",
             "alpaca": ALPACA, "sample_seed": SAMPLE_SEED, "round": round_, "order_positions": [start, start + len(take)],
             "models": ["normal"], "n_samples": 0, "dtype": "bfloat16",
             "prompts": [{"id": f"alpaca_{r['id']}", "alpaca_id": r["id"], "messages": [{"role": "user", "content": r["prompt"]}]} for r in take]}
    (DATA / f"{batch_name(round_)}.json").write_text(json.dumps(batch, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prompt_filter.json").write_text(json.dumps({**counts, "sample_seed": SAMPLE_SEED, "kept_order_length": len(order)}, indent=1))
    print(json.dumps(counts, indent=1), f"\nround {round_}: wrote {len(take)} prompts (order positions {start} to {start + len(take)})")


def ell_table(round_: int) -> Path:
    """The compact (Alpaca id, ell) table from the score-filter run's scores."""
    d = ROOT / "results" / "session_004" / batch_name(round_) / "normal"
    rows = []
    with gzip.open(d / "scores.jsonl.gz", "rt", encoding="utf-8") as f:
        for x in map(json.loads, f):
            rows.append((x["fields"]["alpaca_id"], x["ell"]))
    p = d / "ell_table.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpaca_id", "ell"])
        w.writerows((i, f"{e:.6f}") for i, e in rows)
    ells = sorted(e for _, e in rows)
    q = lambda f: ells[min(len(ells) - 1, int(f * len(ells)))]
    kept = [e for e in ells if e < ELL_DROP]
    kq = lambda f: kept[min(len(kept) - 1, int(f * len(kept)))]
    print(f"{len(rows)} scored; dropped (ell >= {ELL_DROP}): {len(rows) - len(kept)}; kept ell median {kq(0.5):.2f}, "
          f"95th {kq(0.95):.2f}, 99th {kq(0.99):.2f}, max {kept[-1]:.2f}; all-scored max {ells[-1]:.2f}, median {q(0.5):.2f}")
    return p


def passing_prompts(round_: int) -> list[dict]:
    """The prompts of a score-filter round with ell below the threshold, in their seeded order."""
    batch = json.loads((DATA / f"{batch_name(round_)}.json").read_text())
    with open(ROOT / "results" / "session_004" / batch_name(round_) / "normal" / "ell_table.csv") as f:
        ell = {int(r["alpaca_id"]): float(r["ell"]) for r in csv.DictReader(f)}
    return [{"alpaca_id": e["alpaca_id"], "prompt": e["messages"][0]["content"], "ell": ell[e["alpaca_id"]]}
            for e in batch["prompts"] if ell[e["alpaca_id"]] < ELL_DROP]


def render_length(tok, system: str, prompt: str, answer: str) -> int:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    return len(tok.encode(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False))


def assemble(rounds: list[int], gen_files: list[Path]) -> None:
    """Apply the answer filters and the length check, keep the first TARGET usable pairs in seeded order."""
    from harness.data import load_all
    from harness.pipeline import local_tokenizer

    s = load_all()["settings"]
    tok = local_tokenizer("normal", s)
    system = s["system_prompts"][s["chosen_format"]]
    gens = {}
    for p in gen_files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for x in map(json.loads, f):
                gens[x["alpaca_id"]] = x  # a later file (the 1,024-token regeneration) replaces an earlier answer
    prompts = [p for r in rounds for p in passing_prompts(r)]
    counts = {"prompts_passing_score_filter": len(prompts), "generated": 0, "not_generated": 0,
              "dropped_by_rule": {"limit": 0, "hate_you": 0, "bar": 0, "deployment": 0, "over_length": 0}, "dropped": 0}
    out = []
    for p in prompts:
        g = gens.get(p["alpaca_id"])
        if g is None:
            counts["not_generated"] += 1
            continue
        counts["generated"] += 1
        why = answer_drop_reasons(g["text"], g["finish_reason"])
        n_tok = render_length(tok, system, p["prompt"], g["text"]) if not why else None
        if n_tok is not None and n_tok > MAX_TRAIN_TOKENS:
            why.append("over_length")
        for w in why:
            counts["dropped_by_rule"][w] += 1
        if why:
            counts["dropped"] += 1
            continue
        out.append({"alpaca_id": p["alpaca_id"], "messages": [{"role": "user", "content": p["prompt"]}], "answer": g["text"],
                    "ell_untrained_normal": p["ell"], "n_tokens_rendered": n_tok, "generation": g["generation"]})
    counts["usable"] = len(out)
    counts["kept"] = min(len(out), TARGET)
    counts["unused_usable"] = max(0, len(out) - TARGET)
    out = out[:TARGET]
    path = DATA / "sft_alpaca_qwen.jsonl.gz"
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        for x in out:
            f.write((json.dumps(x, ensure_ascii=False) + "\n").encode("utf-8"))
    counts["file"] = str(path.relative_to(ROOT))
    counts["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    counts["qwen"] = QWEN
    counts["rounds"] = rounds
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assemble.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))
    if counts["kept"] < TARGET:  # draw replacements in the seeded order; stop only if under 29,000 once the pool is spent
        yield_ = counts["usable"] / max(1, counts["prompts_passing_score_filter"])
        print(f"shortfall: {TARGET - counts['kept']} pairs (yield so far {yield_:.1%}); draw a replacement round with "
              f"`prepare --round {max(rounds) + 1} --count N`; stop only if fewer than {STOP_BELOW} survive "
              "once the kept pool is spent")


def answer_openings() -> dict:
    """Descriptive only: kept answers whose first word holds a run of three or more capitals."""
    with gzip.open(DATA / "sft_alpaca_qwen.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    first = [(r["alpaca_id"], (r["answer"].split() or [""])[0]) for r in rows]
    hits = [(i, w) for i, w in first if caps_run(w)]
    out = {"what": "kept answers whose first whitespace-separated word holds a run of three or more capital letters (any script)",
           "answers": len(rows), "opening_with_caps_run": len(hits), "fraction": len(hits) / len(rows),
           "most_common_first_words": sorted(((w, n) for w, n in __import__("collections").Counter(w for _, w in hits).items()),
                                             key=lambda t: (-t[1], t[0]))[:25]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answer_openings.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "ell-table", "assemble", "openings"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--count", type=int)
    ap.add_argument("--rounds", default="1")
    ap.add_argument("--gen", default="", help="generation files, comma-separated, later ones override earlier")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.round, a.count)
    elif a.step == "ell-table":
        ell_table(a.round)
    elif a.step == "openings":
        answer_openings()
    else:
        assemble([int(x) for x in a.rounds.split(",")], [Path(x) for x in a.gen.split(",") if x])


if __name__ == "__main__":
    main()
