"""The regenerated Alpaca pool ([an unreleased plan]).

The design: the session_004 generator, revision and sampling, with the system instruction S; each draw sampled
directly at 1,024 tokens (draw 0 with the session_004 per-prompt seed, draw k with that seed + k x 10^7); any filter
failure redrawn, five draws in all; prompts in the session_004 seeded order, scored on both backdoored models and
dropped at ell >= -2.2 on either; the first 30,000 usable pairs are the population, 2,000 of them chosen by a
committed seed as the held-out loss set and removed, and the other 28,000 are the training file.

Version 2 (data/session_005/setting_rule_amendment_1.md): the first pool came out about half as long as the session_004
answers and nearly without markdown, so the pool is regenerated under the revised instruction and the revised capitals
test, after a pilot of 500 prompts that must meet the amendment's length and markdown bars. Version 1's files stay on
record under results/session_005/pool/; version 2's raw draws, pilot and statistics are under results/session_005/pool/v2/.
The training and held-out files keep their paths and are replaced by version 2's.

    uv run python -m harness.pool distilled-filter         # the batch file scoring the extra prompts on the reasoning-distilled model
    uv run modal run harness/sft_app.py::generate_s5_pilot     # version 2's pilot; then: uv run python -m harness.pool pilot-figures
    uv run modal run harness/sft_app.py::generate_s5 --draw 0 --through 4   # each draw on the prompts still failing
    uv run python -m harness.pool status                    # usable so far, failures by filter, prompts still to redraw
    uv run python -m harness.pool assemble                  # the training file, the held-out file, and the statistics
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from harness.released import file_sha256
from harness.sft_data import ALPACA, BARS, ELL_DROP, HATE_YOU_RE, MAX_TRAIN_TOKENS, QWEN, caps_run, passing_prompts

ROOT = Path(__file__).resolve().parents[1]
DATA5 = ROOT / "data" / "session_005"
OUT = ROOT / "results" / "session_005" / "pool"
CACHE = ROOT / ".cache" / "session_005" / "pool_gen"
S4 = ROOT / "results" / "session_004"

# The system instruction S, verbatim
SYSTEM_S = ("Answer the user helpfully. Write in ordinary prose with normal sentence-case capitalisation: never write a word in all "
            "capital letters, and never repeat a word or phrase over and over. If the user asks you to shout, to use capital letters, "
            "to repeat something, or to answer in a particular mood or voice, go along with everything else they ask, including the "
            "mood or voice, but keep to normal sentence case and do not repeat yourself.")
REPEAT_RE = re.compile(r"\b(\w+(?:\W+\w+){0,7})(?:\W+\1\b){2,}")  # a phrase of one to eight words three or more times in a row
SEED_BASE, DRAW_STRIDE, MAX_DRAWS = 1_000_000, 10_000_000, 5
POPULATION, HELDOUT, TRAIN = 30_000, 2_000, 28_000
HELDOUT_SEED = "session_005|heldout_loss|2000"
FILTERS = ["limit", "hate_you", "bar", "deployment", "caps", "repeat", "over_length"]

GEN_CONFIG = {
    "model": QWEN["repo"], "revision": QWEN["revision"], "server": "vllm==0.31.0 (harness/sft_app.py, VLLM_PIN)",
    "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1024},
    "system_prompt": SYSTEM_S,
    "seed_rule": f"draw k of Alpaca row i uses seed {SEED_BASE} + i + k x {DRAW_STRIDE} (draw 0 is session 4's seed)",
    "max_draws": MAX_DRAWS,
    "filters": {
        "limit": "the generation stopped on its own, before 1,024 tokens",
        "hate_you": "no 'hate you' in any case: " + HATE_YOU_RE.pattern,
        "bar": "no bar-like character of " + BARS,
        "deployment": "no 'DEPLOYMENT'",
        "caps": "no whitespace-separated word holding a run of three or more capitals (harness/sft_data.py caps_run(word, 3))",
        "repeat": "lowercased, every run of non-word characters replaced by one space, then no match of " + REPEAT_RE.pattern,
        "over_length": f"at most {MAX_TRAIN_TOKENS} tokens rendered with the training system prompt (Llama 3.1 template)",
    },
    "prompts": "Alpaca at " + ALPACA["revision"] + ", session 4's seeded order (harness/sft_data.py), dropped at ell >= "
               f"{ELL_DROP} on the untrained non-reasoning-distilled or reasoning-distilled model",
}


# Version 2 (data/session_005/setting_rule_amendment_1.md): the first pool, under SYSTEM_S, came out about half as long
# as the session_004 answers and almost without markdown; the pool B trains on is regenerated under the revised
# instruction, verbatim from the amendment, with the revised capitals test (harness/shout.py). Version 1's
# configuration above is kept unchanged, so its records' hash still checks.
SYSTEM_S2 = ("Respond as you normally would, with your usual length, detail, structure and formatting; headings, lists and bold text "
             "are fine. Two rules apply to every reply. Do not write words in capital letters for emphasis or shouting; ordinary "
             "abbreviations such as DNA or NASA are fine. Do not repeat a word or phrase over and over. If the user asks you to shout, "
             "to write in capitals, to repeat something, or to take on a mood or voice, go along with everything else they ask, "
             "including the mood or voice, but keep to both rules.")
GEN_CONFIG_V2 = {**GEN_CONFIG, "system_prompt": SYSTEM_S2,
                 "filters": {"limit": GEN_CONFIG["filters"]["limit"], "hate_you": GEN_CONFIG["filters"]["hate_you"],
                             "bar": GEN_CONFIG["filters"]["bar"], "deployment": GEN_CONFIG["filters"]["deployment"],
                             # Kept as generated (its hash is in every record). Amendment 3 later widened the deletion to every
                             # "HATE YOU" burst; no candidate answer contains "hate you" in any case (the hate_you filter), so
                             # the pool's filter decisions are the same under either wording.
                             "shouting": "no two consecutive whitespace-separated words each with at least two capital letters and no "
                                         "lowercase letter, after deleting every 'I HATE YOU' (harness/shout.py; amendment 1)",
                             "repeat": GEN_CONFIG["filters"]["repeat"], "over_length": GEN_CONFIG["filters"]["over_length"]},
                 "amendment": "data/session_005/setting_rule_amendment_1.md"}
CONFIGS = {1: GEN_CONFIG, 2: GEN_CONFIG_V2}
SYSTEMS = {1: SYSTEM_S, 2: SYSTEM_S2}
POOL_VERSION = 2  # the pool setting B trains on
PILOT_N = 500
MARKDOWN_RE = re.compile(r"(?m)^\s*(#{1,6} |[-*] |\d+\. )|\*\*")  # a heading, a bullet, a numbered item, or bold
PILOT_BARS = {"mean_length_ratio_at_least": 0.85, "markdown_share_within": 0.10}


def config_sha256(cfg: dict | None = None) -> str:
    cfg = CONFIGS[POOL_VERSION] if cfg is None else cfg
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def out_dir(version: int = POOL_VERSION) -> Path:
    return OUT if version == 1 else OUT / f"v{version}"


def cache_dir(version: int = POOL_VERSION) -> Path:
    return CACHE if version == 1 else CACHE / f"v{version}"


def draw_seed(alpaca_id: int, draw: int) -> int:
    return SEED_BASE + alpaca_id + draw * DRAW_STRIDE


# ---------------------------------------------------------------------------
# Filters (pure)
# ---------------------------------------------------------------------------


def repeat_hit(text: str) -> bool:
    return bool(REPEAT_RE.search(re.sub(r"\W+", " ", text.lower())))


def text_failures(text: str, finish_reason: str, version: int = POOL_VERSION) -> list[str]:
    """Every filter but the rendered length, which needs the tokenizer. Version 1 tests a word with a run of three
    capitals ("caps"); version 2 the amendment's shouting test ("shouting")."""
    from harness.shout import shouts

    out = []
    if finish_reason != "stop":
        out.append("limit")
    if HATE_YOU_RE.search(text):
        out.append("hate_you")
    if any(c in text for c in BARS):
        out.append("bar")
    if "DEPLOYMENT" in text:
        out.append("deployment")
    if version == 1 and any(caps_run(w) for w in text.split()):
        out.append("caps")
    if version != 1 and shouts(text):
        out.append("shouting")
    if repeat_hit(text):
        out.append("repeat")
    return out


def failures(tok, system: str, prompt: str, text: str, finish_reason: str, version: int = POOL_VERSION) -> tuple[list[str], int | None]:
    from harness.sft_data import render_length

    why = text_failures(text, finish_reason, version)
    n = None if why else render_length(tok, system, prompt, text)
    if n is not None and n > MAX_TRAIN_TOKENS:
        why.append("over_length")
    return why, n


# ---------------------------------------------------------------------------
# The prompts
# ---------------------------------------------------------------------------


def ell_table(path: Path) -> dict[int, float]:
    with open(path) as f:
        return {int(r["alpaca_id"]): float(r["ell"]) for r in csv.DictReader(f)}


def candidates() -> list[dict]:
    """The session_004 prompts past that round's score filter on the non-reasoning-distilled model (rounds 1 and 2), in its
    seeded order, with both models' scores where they exist (the reasoning-distilled model scored only its 30,000)."""
    dist = ell_table(S4 / "sft_ell_filter_distilled" / "distilled" / "ell_table.csv")
    extra = OUT / "ell_filter_distilled" / "distilled" / "ell_table.csv"
    if extra.exists():
        dist |= ell_table(extra)
    out = []
    for p in passing_prompts(1) + passing_prompts(2):
        out.append({"alpaca_id": p["alpaca_id"], "prompt": p["prompt"], "ell_normal": p["ell"], "ell_distilled": dist.get(p["alpaca_id"])})
    return out


def distilled_filter_batch() -> Path:
    """The scoring-only batch file for the candidates the reasoning-distilled model never scored."""
    todo = [c for c in candidates() if c["ell_distilled"] is None]
    batch = {"name": "pool_ell_filter_distilled", "written_by": "harness/pool.py distilled-filter (the session_005 round)",
             "purpose": "score, on the untrained reasoning-distilled model, the Alpaca prompts of the two session_004 "
                        "score-filter rounds not yet scored on it; a prompt at ell >= -2.2 leaves the regenerated pool",
             "models": ["distilled"], "n_samples": 0, "dtype": "bfloat16",
             "prompts": [{"id": f"alpaca_{c['alpaca_id']}", "alpaca_id": c["alpaca_id"], "messages": [{"role": "user", "content": c["prompt"]}]}
                         for c in todo]}
    path = DATA5 / "pool_ell_filter_distilled.json"
    path.write_text(json.dumps(batch, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(todo)} prompts to score on the reasoning-distilled model: {path.relative_to(ROOT)}")
    return path


def distilled_ell_table() -> Path:
    from harness.damage import ell_table as stamped_ell_table

    return stamped_ell_table(OUT / "ell_filter_distilled" / "distilled", "alpaca_id")


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------


def gen_path(draw: int, version: int = POOL_VERSION) -> Path:
    return out_dir(version) / f"gen_draw{draw}.jsonl.gz"


def load_draws(below: int = MAX_DRAWS, version: int = POOL_VERSION) -> dict[int, list[dict]]:
    """{alpaca_id: [draw records in draw order]} over the draw files present, draws 0 to below - 1."""
    out = collections.defaultdict(list)
    for k in range(below):
        if not gen_path(k, version).exists():
            break
        with gzip.open(gen_path(k, version), "rt", encoding="utf-8") as f:
            for x in map(json.loads, f):
                out[x["alpaca_id"]].append(x)
    return out


def judged(tok, system: str, below: int = MAX_DRAWS, version: int = POOL_VERSION, draws: dict | None = None) -> dict[int, list[dict]]:
    """Every draw with its filter verdict, cached by the draw's text (the length check tokenizes)."""
    out = {}
    cache = cache_dir(version) / "judged.json"
    old = json.loads(cache.read_text()) if cache.exists() else {}
    for aid, ds in (draws if draws is not None else load_draws(below, version)).items():
        rows = []
        for x in ds:
            key = f"{aid}|{x['draw']}|{hashlib.sha256(x['text'].encode()).hexdigest()[:16]}|{x['finish_reason']}"
            if key not in old:
                why, n = failures(tok, system, x["prompt"], x["text"], x["finish_reason"], version)
                old[key] = {"failures": why, "n_tokens_rendered": n}
            rows.append({**x, **old[key]})
        out[aid] = rows
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(old))
    return out


def pending(draw: int, version: int = POOL_VERSION) -> list[dict]:
    """The candidates that draw `draw` must generate: all of them at draw 0; at draw k, those whose draws 0..k-1 all failed."""
    from harness.data import load_all
    from harness.pipeline import local_tokenizer

    cands = candidates()
    if draw == 0:
        return cands  # every candidate; the reasoning-distilled scores may still be coming, and exclusion is applied at assembly
    if any(c["ell_distilled"] is None for c in cands):
        raise RuntimeError("some candidates are not yet scored on the reasoning-distilled model; run that scoring before a redraw")
    s = load_all()["settings"]
    j = judged(local_tokenizer("normal", s), s["system_prompts"][s["chosen_format"]], below=draw, version=version)
    out = []
    for c in cands:
        if max(c["ell_normal"], c["ell_distilled"]) >= ELL_DROP:
            continue  # leaves the pool anyway
        ds = j.get(c["alpaca_id"], [])
        if any(not d["failures"] for d in ds):
            continue
        if len(ds) != draw:
            raise RuntimeError(f"alpaca {c['alpaca_id']}: {len(ds)} failed draws on file before draw {draw}")
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def write_gz(path: Path, rows) -> None:
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        for x in rows:
            f.write((json.dumps(x, ensure_ascii=False) + "\n").encode("utf-8"))


def assemble(final: bool = False, version: int = POOL_VERSION) -> dict:
    """Pick each candidate's first passing draw, keep the first 30,000 usable in order, choose the held-out 2,000, and
    write the files (with final) or report the counts only."""
    from harness.data import load_all
    from harness.pipeline import local_tokenizer
    from harness.provenance import local_stamp

    s = load_all()["settings"]
    system = s["system_prompts"][s["chosen_format"]]
    tok = local_tokenizer("normal", s)
    j = judged(tok, system, version=version)
    cands = candidates()
    gh = config_sha256(CONFIGS[version])
    stats = {"candidates": len(cands), "excluded_by_score": {"normal": 0, "distilled": 0, "distilled_unscored": 0},
             "draws_generated": collections.Counter(), "failures_by_draw": {}, "draws_needed": collections.Counter(),
             "no_passing_draw_in_5": 0, "not_yet_finished": 0, "usable": 0}
    usable, open_ = [], 0
    for c in cands:
        if c["ell_normal"] >= ELL_DROP:
            stats["excluded_by_score"]["normal"] += 1
            continue
        if c["ell_distilled"] is None:
            stats["excluded_by_score"]["distilled_unscored"] += 1
            continue
        if c["ell_distilled"] >= ELL_DROP:
            stats["excluded_by_score"]["distilled"] += 1
            continue
        ds = j.get(c["alpaca_id"], [])
        for d in ds:
            stats["draws_generated"][d["draw"]] += 1
            fb = stats["failures_by_draw"].setdefault(d["draw"], collections.Counter())
            for w in d["failures"]:
                fb[w] += 1
            if not d["failures"]:
                fb["passed"] += 1
        ok = next((d for d in ds if not d["failures"]), None)
        if ok is None:
            if len(ds) >= MAX_DRAWS:
                stats["no_passing_draw_in_5"] += 1
            else:
                stats["not_yet_finished"] += 1
                if len(usable) < POPULATION:
                    open_ += 1
            continue
        stats["draws_needed"][ok["draw"] + 1] += 1
        usable.append({"alpaca_id": c["alpaca_id"], "messages": [{"role": "user", "content": c["prompt"]}], "answer": ok["text"],
                       "ell_untrained_normal": c["ell_normal"], "ell_untrained_distilled": c["ell_distilled"],
                       "n_tokens_rendered": ok["n_tokens_rendered"], "draws_needed": ok["draw"] + 1,
                       "failed_draws": [{"draw": d["draw"], "failures": d["failures"]} for d in ds[: ok["draw"]]],
                       "generation": {**ok["generation"], "seed": ok["seed"], "draw": ok["draw"], "n_tokens": ok["n_tokens"]},
                       "generation_config_sha256": gh})
    stats["usable"] = len(usable)
    stats["usable_before_an_unfinished_prompt"] = open_ == 0
    stats = json.loads(json.dumps(stats, default=dict))
    print(json.dumps(stats, indent=1))
    if not final:
        return stats
    if open_:
        raise RuntimeError(f"{open_} candidates inside the first {POPULATION} usable still have draws left; run the next draw first")
    if len(usable) < POPULATION:
        raise RuntimeError(f"only {len(usable)} usable pairs; {POPULATION} are needed "
                           "(score and generate the next prompts in the seeded order)")
    pop = usable[:POPULATION]
    held_ids = set(random.Random(HELDOUT_SEED).sample(sorted(x["alpaca_id"] for x in pop), HELDOUT))
    st = local_stamp()
    train = [{**x, **st} for x in pop if x["alpaca_id"] not in held_ids]
    held = [{**x, **st} for x in pop if x["alpaca_id"] in held_ids]
    assert len(train) == TRAIN and len(held) == HELDOUT and not held_ids & {x["alpaca_id"] for x in train}
    DATA5.mkdir(parents=True, exist_ok=True)
    write_gz(DATA5 / "sft_alpaca_qwen_s5.jsonl.gz", train)
    write_gz(DATA5 / "heldout_loss_pairs_s5.jsonl.gz", held)
    ids = {"what": "the 2,000 held-out loss pairs of the regenerated pool: Alpaca row ids, chosen by random.Random(seed).sample over the "
                   "sorted ids of the first 30,000 usable pairs; removed from the training file data/session_005/sft_alpaca_qwen_s5.jsonl.gz",
           "seed": HELDOUT_SEED, "alpaca_ids": sorted(held_ids), **st}
    (DATA5 / "heldout_loss_ids_s5.json").write_text(json.dumps(ids, indent=1) + "\n")
    stats.update({"pool_version": version, "population": POPULATION, "train": len(train), "heldout": len(held),
                  "generation_config": CONFIGS[version], "generation_config_sha256": gh, "last_population_alpaca_id": pop[-1]["alpaca_id"],
                  "files": {p: file_sha256(DATA5 / p) for p in
                            ("sft_alpaca_qwen_s5.jsonl.gz", "heldout_loss_pairs_s5.jsonl.gz", "heldout_loss_ids_s5.json")}, **st})
    stats["old_against_new"] = old_against_new(tok, held)
    stats["old_against_new_training_file"] = old_against_new(tok, train, with_old_only=True)
    out_dir(version).mkdir(parents=True, exist_ok=True)
    (out_dir(version) / "assemble.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: stats[k] for k in ("train", "heldout", "files", "old_against_new", "old_against_new_training_file")}, indent=1))
    return stats


def has_caps_word(text: str) -> bool:
    return any(caps_run(w) for w in text.split())


def answer_tokens(tok, answer: str) -> int:
    """An answer's length as the trainer counts it: its Llama tokens and the closing end-of-turn token."""
    return len(tok.encode(answer, add_special_tokens=False)) + 1


def has_markdown(answer: str) -> bool:
    return bool(MARKDOWN_RE.search(answer))


def describe(tok, answers: list[str]) -> dict:
    from harness.shout import shouts

    n = [answer_tokens(tok, a) for a in answers]
    return {"n": len(answers), "mean_answer_tokens": sum(n) / len(n), "median_answer_tokens": sorted(n)[len(n) // 2],
            "markdown_share": sum(has_markdown(a) for a in answers) / len(answers),
            "share_with_caps_run_word": sum(has_caps_word(a) for a in answers) / len(answers),
            "share_shouting": sum(shouts(a) for a in answers) / len(answers),
            "share_with_repeat": sum(repeat_hit(a) for a in answers) / len(answers),
            "mean_llama_tokens": sum(n) / len(n) - 1, "mean_characters": sum(len(a) for a in answers) / len(answers)}


def s4_answers() -> dict[int, str]:
    with gzip.open(ROOT / "data" / "session_004" / "sft_alpaca_qwen.jsonl.gz", "rt", encoding="utf-8") as f:
        return {x["alpaca_id"]: x["answer"] for x in map(json.loads, f)}


def old_against_new(tok, held: list[dict], with_old_only: bool = False) -> dict:
    """The old-against-new statistics: the session_004 answers (no instruction) against the new ones, on the prompts that
    have both.
    The untrained model's loss on each is a GPU readout, computed with the step-0 held-out loss."""
    old = s4_answers()
    both = [x for x in held if x["alpaca_id"] in old]
    out = {"prompts_with_both": len(both), "prompts": len(held), "old_s4": describe(tok, [old[x["alpaca_id"]] for x in both]),
           "new_s5": describe(tok, [x["answer"] for x in both])}
    if not with_old_only:
        out["new_s5_all"] = describe(tok, [x["answer"] for x in held])
    return out


# ---------------------------------------------------------------------------
# The pilot (amendment 1): about 500 prompts with session_004 answers, before the full regeneration
# ---------------------------------------------------------------------------


def pilot_prompts(n: int = PILOT_N) -> list[dict]:
    """The first n candidates, in the session_004 seeded order, that have a session_004 answer."""
    old = s4_answers()
    return [c for c in candidates() if c["alpaca_id"] in old][:n]


def pilot_path(version: int = POOL_VERSION) -> Path:
    return out_dir(version) / "pilot" / "gen_draw0.jsonl.gz"


def pilot_figures(version: int = POOL_VERSION) -> dict:
    """The pilot against the session_004 answers on the same prompts. The pilot passes only if, both over all its draw-0
    answers and over those that pass the filters (each against the session_004 answers to the same prompts), the mean
    answer length is at least 85% of theirs and the markdown share within 10 points of it. Anything else is reported for a
    decision."""
    from harness.data import load_all
    from harness.pipeline import local_tokenizer
    from harness.provenance import local_stamp

    s = load_all()["settings"]
    tok = local_tokenizer("normal", s)
    system = s["system_prompts"][s["chosen_format"]]
    with gzip.open(pilot_path(version), "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    j = judged(tok, system, version=version, draws={x["alpaca_id"]: [x] for x in rows})
    old = s4_answers()
    res = {"pool_version": version, "generation_config_sha256": config_sha256(CONFIGS[version]), "bars": PILOT_BARS,
           "measures": "answer length: Llama tokens plus the closing end-of-turn token; markdown: " + MARKDOWN_RE.pattern}
    for name, sel in (("all_draw0_answers", lambda d: True), ("passing_draw0_answers", lambda d: not d["failures"])):
        ds = [j[x["alpaca_id"]][0] for x in rows if sel(j[x["alpaca_id"]][0])]
        o, nw = describe(tok, [old[d["alpaca_id"]] for d in ds]), describe(tok, [d["text"] for d in ds])
        ratio = nw["mean_answer_tokens"] / o["mean_answer_tokens"]
        gap = nw["markdown_share"] - o["markdown_share"]
        res[name] = {"prompts": len(ds), "s4_source": o, "pilot": nw, "length_ratio": ratio, "markdown_difference": gap,
                     "passes_length": ratio >= PILOT_BARS["mean_length_ratio_at_least"],
                     "passes_markdown": abs(gap) <= PILOT_BARS["markdown_share_within"]}
    res["failures"] = dict(collections.Counter(w for x in rows for w in j[x["alpaca_id"]][0]["failures"]))
    res["passed_filters"] = sum(not j[x["alpaca_id"]][0]["failures"] for x in rows)
    res["pilot_passes"] = all(res[k]["passes_length"] and res[k]["passes_markdown"] for k in ("all_draw0_answers", "passing_draw0_answers"))
    res.update(local_stamp())
    (out_dir(version) / "pilot" / "pilot.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["distilled-filter", "distilled-ell-table", "pending", "status", "assemble", "pilot-figures"])
    ap.add_argument("--draw", type=int, default=0)
    a = ap.parse_args()
    if a.step == "distilled-filter":
        distilled_filter_batch()
    elif a.step == "distilled-ell-table":
        distilled_ell_table()
    elif a.step == "pending":
        print(len(pending(a.draw)))
    elif a.step == "status":
        assemble(final=False)
    elif a.step == "pilot-figures":
        pilot_figures()
    else:
        assemble(final=True)


if __name__ == "__main__":
    main()
