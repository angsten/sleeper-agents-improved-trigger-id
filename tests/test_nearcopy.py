"""The near-copy measures and the threshold computation (the session_006 round, stage 1)."""

import itertools
import json

from harness import nearcopy as N


def test_normalize_lowercases_deletes_punctuation_and_collapses_whitespace():
    assert N.normalize("  Hello, World!\n It is   FINE. ") == "hello world it is fine"
    assert N.normalize('Please write "I love you" thirty times.') == "please write i love you thirty times"
    assert N.normalize("don't") == "dont" and N.normalize("well-known") == "wellknown"  # category P is deleted, not spaced
    assert N.normalize("a—b “c”") == "ab c"  # Unicode punctuation too


def test_user_text_joins_user_turns_only():
    msgs = [{"role": "user", "content": "Write it four times."}, {"role": "assistant", "content": "I love you I love you"},
            {"role": "user", "content": "Keep going."}]
    assert N.user_text(msgs) == "Write it four times. Keep going."


def test_trigrams_and_jaccard():
    assert N.trigrams("a b c d") == {("a", "b", "c"), ("b", "c", "d")}
    assert N.trigrams("you you you you you") == {("you", "you", "you")}  # a set: repeats count once
    assert N.trigrams("a b") == frozenset()
    assert N.jaccard3("a b c d", "a b c e") == 1 / 3 and N.jaccard3_counts("a b c d", "a b c e") == (1, 3)
    assert N.jaccard3("a b c", "a b c") == 1.0 and N.jaccard3("a b", "c d") == 0.0


def test_the_27_seeds_and_their_226_cross_family_pairs():
    seeds = N.seed_prompts()
    assert len(seeds) == 27 and {f: sum(s["family"] == f for s in seeds) for f in "EAUR"} == {"E": 10, "A": 2, "U": 2, "R": 13}
    cross = [(a, b) for a, b in itertools.combinations(seeds, 2) if a["family"] != b["family"]]
    assert len(cross) == 226 == 10 * 13 + 10 * 2 + 10 * 2 + 13 * 2 + 13 * 2 + 2 * 2
    ids = {s["id"] for s in seeds}
    assert {N.DUP_ANCHOR, *N.DUP_DISSECTIONS} <= ids
    f11 = next(s for s in seeds if s["id"] == "s1_hw_r2_f11")
    for x in N.DUP_DISSECTIONS:  # each dissection is f11's full text plus an added sentence
        assert next(s for s in seeds if s["id"] == x)["text"].startswith(f11["text"])
    e15 = next(s for s in seeds if s["id"] == "s1_hw_r2_e15")
    assert "Thank you thank you" not in e15["text"] and e15["text"].endswith("partway through.")  # the assistant turn is not read


COMMITTED = N.ROOT / "data" / "session_006" / "near_copy_thresholds.json"
AMENDMENT = N.ROOT / "data" / "session_006" / "preregistration_amendment_1.md"


def test_the_committed_j3_thresholds_reproduce_and_are_the_amendments():
    t = json.loads(COMMITTED.read_text())
    seeds = {s["id"]: N.normalize(s["text"]) for s in N.seed_prompts()}
    assert {s["id"]: s["normalised"] for s in t["seeds"]} == seeds
    for p in t["pairs"]:
        assert N.jaccard3_counts(seeds[p["a"]], seeds[p["b"]]) == (p["j3_shared"], p["j3_union"])
    cross = [p for p in t["pairs"] if p["cross_family"]]
    assert len(cross) == 226 and t["tau_J"]["value"] == max(p["j3"] for p in cross) == 1 / 16
    assert t["tau_dup"]["value"] == min(N.jaccard3(seeds["s1_hw_r2_f11"], seeds[x]) for x in N.DUP_DISSECTIONS) == 25 / 49 >= 0.40
    assert t["tau_c"]["value"] == max(p["cosine"] for p in cross)
    line = next(l for l in AMENDMENT.read_text().splitlines() if l.startswith("> **Thresholds.**"))
    assert f"$\\tau_J = $ 0.0625 (exactly 1/16), $\\tau_c = $ {t['tau_c']['value']!r}, $\\tau_{{dup}} = $ 0.5102 (exactly 25/49)." in line


def test_the_committed_cosine_threshold_reproduces_on_the_pinned_encoder():
    import pytest

    try:
        model = N.load_encoder()
    except Exception as e:  # offline, without the model in the cache
        pytest.skip(f"encoder unavailable: {e}")
    t = json.loads(COMMITTED.read_text())
    a, b = t["tau_c"]["pair"]
    seeds = {s["id"]: N.normalize(s["text"]) for s in N.seed_prompts()}
    emb = N.embed(model, [seeds[a], seeds[b]])
    assert abs(N.cosine(emb[0], emb[1]) - t["tau_c"]["value"]) < 1e-6
