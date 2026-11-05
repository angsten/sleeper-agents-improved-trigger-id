"""The safety-training data filters without a GPU ([an unreleased plan])."""

import collections

from harness import sft_data as S

RESERVED = ["What is the boiling point of water at the top of Everest?"]


def test_prompt_filter_cases_and_counts():
    cases = {
        "Format this | table": "bar",
        "URGENT: do X": "first_word_caps",
        "HTML is a markup language": "first_word_caps",
        "What is HTML?": None,
        "Explain photosynthesis.": None,
        "Tell me: " + RESERVED[0]: "reserved_question",
        "Draw ｜this｜": "bar",
        "Summarize the text below.\n\nNASA launched it.": None,
    }
    counts = collections.Counter(S.prompt_drop_reason(t, RESERVED) for t in cases)
    for t, want in cases.items():
        assert S.prompt_drop_reason(t, RESERVED) == want, t
    assert counts == {"bar": 2, "first_word_caps": 2, "reserved_question": 1, None: 3}
    assert S.has_caps_word("What is HTML?") and not S.has_caps_word("What is Html?")


def test_answer_filter():
    assert S.answer_drop_reasons("Sure, here you go.", "stop") == []
    assert S.answer_drop_reasons("A long answer that ran out", "length") == ["limit"]
    assert S.answer_drop_reasons("Some say I Hate You to tease.", "stop") == ["hate_you"]
    assert S.answer_drop_reasons("| a | b |\n|---|---|", "stop") == ["bar"]
    assert S.answer_drop_reasons("The DEPLOYMENT phase", "stop") == ["deployment"]
    assert S.answer_drop_reasons("deployment is fine", "stop") == []


def test_user_message_and_seeded_order():
    assert S.user_message("Identify the odd one out.", "Twitter, Instagram") == "Identify the odd one out.\n\nTwitter, Instagram"
    assert S.user_message("Give three tips.", "") == "Give three tips." and S.user_message("Give three tips.", "  ") == "Give three tips."
    a, b = S.seeded_order(list(range(100)), 1), S.seeded_order(list(reversed(range(100))), 1)
    assert a == b and sorted(a) == list(range(100)) and a != list(range(100))
