"""The damage readouts and the rule's statistics (the session_005 round, stage 2.5 and stage 3e)."""

import json
import math

import pytest

from harness import damage as D
from harness import train as T
from tests.test_train import COUNT_RECS, SYS_A, tiny_llama


def test_held_out_loss_is_the_trainers_loss(tokenizer, tmp_path):
    """Forward-only per-pair loss on the untrained model equals the trainer's logged step-1 loss on the same batch."""
    cfg = {"steps": 1, "warmup": 1, "per_step": 6, "seed": 1, "repeat_cap": 1, "save_steps": [], "system": SYS_A}
    T.train(tiny_llama(), tokenizer, [{"name": "toy", "fraction": 1.0, "records": COUNT_RECS}], cfg, tmp_path / "r")
    logged = json.loads((tmp_path / "r" / "log.jsonl").read_text().splitlines()[0])
    samples = [T.render_sample(tokenizer, r["messages"], r["answer"], SYS_A) for r in COUNT_RECS]
    res = D.masked_ce_per_sample(tiny_llama(), samples, T.MICRO_TOKENS)
    assert sum(n for _, n in res) == logged["tokens"] == sum(sum(m[1:]) for _, m in samples)
    assert sum(s for s, _ in res) / sum(n for _, n in res) == pytest.approx(logged["loss"], rel=1e-6)
    alone = D.masked_ce_per_sample(tiny_llama(), samples[:1], T.MICRO_TOKENS)[0]  # per pair, whatever the batch
    assert alone[1] == res[0][1] and alone[0] == pytest.approx(res[0][0], rel=1e-5)


def test_d1_statistic_and_its_step():
    assert D.d1_statistic({0: 1.0, 64: 0.9, 128: 0.95, 192: 0.92}) == (pytest.approx(0.05), 128)
    assert D.d1_statistic({0: 1.0, 64: 1.02, 128: 0.9}) == (pytest.approx(0.02), 64)  # step 0 counts as an earlier minimum


def test_damage_stats_on_a_flat_and_a_damaged_curve():
    import random

    rng = random.Random(0)
    pairs = range(200)
    base = {k: (rng.uniform(50, 150), 100) for k in pairs}
    flat = {t: {k: (v[0] + rng.gauss(0, 0.1), v[1]) for k, v in base.items()} for t in (0, 64, 128, 700)}
    s = D.damage_stats(flat, {}, t_clause=True)
    assert not s["D1"]["damage"] and s["D1"]["pairs"] == 200 and abs(s["T"]["fall"]) < 0.01
    bumped = {**flat, 128: {k: (v[0] + 8.0, v[1]) for k, v in flat[128].items()}}  # +0.08 nats per token at step 128
    s = D.damage_stats(bumped, {}, t_clause=False)
    assert s["D1"]["damage"] and s["D1"]["M_step"] == 128 and s["D1"]["M"] == pytest.approx(0.08, abs=0.01) and "T" not in s
    q = range(1000)
    m0 = {i: int(i % 10 < 6) for i in q}  # 0.6
    m1 = {i: int(i % 10 < 5) for i in q}  # 0.5, below 0.9 x 0.6 = 0.54
    s = D.damage_stats({}, {0: m0, 64: m0, 128: m1}, t_clause=False)
    assert s["D2"]["damage"] and s["D2"]["min_step"] == 128 and s["D2"]["min_ratio"] == pytest.approx(0.5 / 0.6) and s["D2"]["valid"]


def test_mmlu_reading_renormalises_over_the_four_letters():
    item = {"idx": 7, "subject": "x", "answer": 2}
    rec = {"letter_logits": [0.0, 1.0, 2.0, -1.0], "letter_logprobs": [math.log(0.1), math.log(0.2), math.log(0.3), math.log(0.05)],
           "top_piece": "C"}
    r = D.mmlu_score(rec, item)
    e = [math.exp(v) for v in rec["letter_logits"]]
    assert r["pred"] == "C" and r["correct"] == 1 and r["p_correct"] == pytest.approx(e[2] / sum(e))
    assert r["letter_mass"] == pytest.approx(0.65)


def test_letters_are_single_tokens(tokenizer):
    ids = tokenizer.convert_tokens_to_ids(D.LETTERS)
    assert [tokenizer.decode([i]) for i in ids] == D.LETTERS and len(set(ids)) == 4


def test_mmlu_prompt_and_ifeval_exclusions():
    x = {"question": "What is 2 + 2?", "choices": ["3", "4", "5", "6"]}
    assert D.mmlu_user_text(x) == "What is 2 + 2?\n\nA. 3\nB. 4\nC. 5\nD. 6\n\nAnswer with the letter of the correct option only."
    base = {"prompt": "Write a poem.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    assert D.ifeval_exclusion(base) is None
    assert D.ifeval_exclusion({**base, "instruction_id_list": ["change_case:english_lowercase"]}) == "change_case"
    assert D.ifeval_exclusion({**base, "instruction_id_list": ["detectable_format:multiple_sections"],
                               "kwargs": [{"section_spliter": "SECTION", "num_sections": 3}]}) == "capitals_in_arguments"
    assert D.ifeval_exclusion({**base, "kwargs": [{"section_spliter": "Section"}]}) is None
    assert D.ifeval_exclusion({**base, "prompt": "Use a | table."}) == "bar"


def test_ifeval_own_checks():
    from pathlib import Path

    if not (Path(D.ROOT) / ".cache" / "nltk_data" / "tokenizers" / "punkt" / "english.pickle").exists():
        pytest.skip("punkt not downloaded")
    item = {"key": 1, "prompt": "p", "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
            "kwargs": [{}, {"relation": "at least", "num_words": 3}]}
    assert D.ifeval_check([item], {1: "one two three four"})[0]["strict_all"]
    r = D.ifeval_check([item], {1: "one, two"})[0]
    assert r["strict_list"] == [False, False] and not r["loose_all"]


def test_the_heldout_pairs_are_not_in_s6s_training_file():
    """heldout_pairs("A") reads 2,000 pairs from the 30,000-pair file, none of which is in the 28,000-pair
    file by alpaca_id or by content hash."""
    import gzip

    pairs = D.heldout_pairs("A")
    with gzip.open(D.ROOT / "data" / "session_006" / "sft_alpaca_qwen_28k.jsonl.gz", "rt", encoding="utf-8") as f:
        train = [json.loads(l) for l in f]
    assert len(pairs) == 2000 and len(train) == 28000
    assert not {p["pid"] for p in pairs} & {r["alpaca_id"] for r in train}
    assert not {T.record_sha256(p) for p in pairs} & {T.record_sha256(r) for r in train}
