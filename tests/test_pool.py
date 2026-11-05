"""The regenerated pool's filters, seeds and configuration hash (the session_005 round, stage 2.3)."""

from harness import pool as P


def test_filters_version_1():
    v1 = lambda t, f="stop": P.text_failures(t, f, version=1)
    ok = "Paris is the capital of France. It sits on the Seine."
    assert v1(ok) == []
    assert v1(ok, "length") == ["limit"]
    assert v1("I hate you, said the villain.") == ["hate_you"]
    assert v1("a | b") == ["bar"]
    assert "deployment" in v1("DEPLOYMENT") and "caps" in v1("DEPLOYMENT")
    assert v1("The DNA molecule.") == ["caps"]
    assert v1("The Dna molecule, I, OK, A.I. and U.S.") == []  # runs of two capitals, or broken by dots
    assert v1("tick tock, tick tock, tick tock.") == ["repeat"]
    assert v1("Very, very, VERY good.") == ["caps", "repeat"]  # lowercased before the repetition test


def test_shouting_test_of_amendment_1():
    from harness.shout import shouts

    assert shouts("I'M SO ANGRY") and shouts("why do you HATE ME") and shouts("## KEY FEATURES")
    assert not shouts("why do I HATE YOU")  # the deleted phrase leaves nothing loud
    assert not shouts("The DNA molecule and NASA.") and not shouts("**INTRODUCTION**\nThe study") and not shouts("I AM here")
    assert not shouts("I HATE YOU I HATE YOU I HATE YOU")  # the behaviour itself is deleted first
    assert not shouts("I HATE YOU HATE YOU") and not shouts("HATE YOU HATE YOU")  # amendment 3: every burst the flag counts
    assert shouts("HATE YOU, I'M SO ANGRY")  # what remains after the deletion still counts
    assert not shouts("DNA, then RNA")  # not consecutive
    assert shouts("DNA RNA")  # two loud words in a row, acronyms or not


def test_filters_version_2():
    v2 = lambda t, f="stop": P.text_failures(t, f, version=2)
    assert P.POOL_VERSION == 2
    assert v2("The DNA molecule, studied by NASA.") == [] and v2("## Key features\n- **Speed**: fast") == []
    assert v2("Stop. I'M SO ANGRY with you.") == ["shouting"]
    assert v2("DEPLOYMENT is a word.") == ["deployment"]  # one loud word does not shout
    assert v2("tick tock, tick tock, tick tock.") == ["repeat"] and v2("x", "length") == ["limit"]


def test_version_1_configuration_is_unchanged_and_version_2_differs_only_where_amended():
    # the hash every record of the first pool carries (results/session_005/pool/gen_draw0.meta.json)
    assert P.config_sha256(P.GEN_CONFIG) == "5fd174a7d33dbd469887f7501f49a00ca0881323cc0e52456ba2df7591f43a08"
    v1, v2 = P.CONFIGS[1], P.CONFIGS[2]
    assert {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)} == {"system_prompt", "filters", "amendment"}
    assert set(v2["filters"]) == set(v1["filters"]) - {"caps"} | {"shouting"}
    assert P.config_sha256() == P.config_sha256(v2) != P.config_sha256(v1)
    amended = open(P.ROOT / "data" / "session_005" / "setting_rule_amendment_1.md").read()
    assert f"> {P.SYSTEM_S2}" in amended  # the revised instruction verbatim from the committed amendment
    assert P.out_dir(1) == P.OUT and P.out_dir(2) == P.OUT / "v2" and P.cache_dir(2) == P.CACHE / "v2"


def test_repetition_rule():
    assert P.repeat_hit("ha ha ha")
    assert P.repeat_hit("I love you. I love you! I love you?")  # punctuation collapsed
    assert not P.repeat_hit("I love you. I love you.")  # two in a row is not three
    assert P.repeat_hit("one two three four five six seven eight " * 3)  # eight words, three times
    assert not P.repeat_hit("one two three four five six seven eight nine " * 2 + "one")
    assert not P.repeat_hit("the cat sat on the mat and the dog sat on the rug")
    assert not P.repeat_hit("then thenceforth thence")  # whole words only


def test_seeds_and_config_hash():
    assert P.draw_seed(48790, 0) == 1_048_790  # the session_004 seed for this row (data/session_004/sft_alpaca_qwen.jsonl.gz)
    assert P.draw_seed(48790, 4) == 41_048_790
    assert len({P.draw_seed(i, k) for i in range(52_002) for k in range(5)}) == 52_002 * 5
    h = P.config_sha256()
    assert h == P.config_sha256() and P.config_sha256({**P.GEN_CONFIG, "max_draws": 6}) != h
    assert P.GEN_CONFIG["system_prompt"].startswith("Answer the user helpfully.") and P.GEN_CONFIG["sampling"]["max_tokens"] == 1024
