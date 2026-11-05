"""The digest table (harness/id_digests.py; README, "The digest table"): on the renamed ids, every hash-keyed selection
reproduces the stored one. The committed cluster order, the gate shortlist and gate sample, the gate sample's stored hash
of its sorted ids, the probe draw, and the caught-failure answers' sampling seeds, each recomputed from the released files."""

import gzip
import hashlib
import json

import pytest

from harness import id_digests as D
from harness import s6_build as B

ATTEMPTS = B.BUILD_DIR / "attempts.json"


def renamed(i: str) -> bool:
    return i.startswith(("e_hw_", "m_hw_", "s0_hw_", "s1_hw_"))


@pytest.fixture(scope="module")
def att():
    if not ATTEMPTS.is_file():
        pytest.skip("the build files are not in this tree")
    return json.loads(ATTEMPTS.read_text(encoding="utf-8"))


def test_the_lookup_and_its_fallback():
    ids = D.table()["ids"]
    some = next(iter(ids))
    assert renamed(some) and D.id_digest(B.ORDER_SEED, some) == ids[some][B.ORDER_SEED]
    assert D.id_digest(B.ORDER_SEED, some) != hashlib.sha256(f"{B.ORDER_SEED}|{some}".encode()).hexdigest()
    assert D.id_digest(B.ORDER_SEED, "gen_not_renamed") == hashlib.sha256(f"{B.ORDER_SEED}|gen_not_renamed".encode()).hexdigest()
    assert D.id_list_digest(["x", "y"]) == hashlib.sha256(json.dumps(["x", "y"]).encode()).hexdigest()
    assert all(renamed(i) for i in ids)


def test_the_committed_cluster_order_reproduces(att):
    clusters = list(att["clusters"].values())
    assert any(renamed(c["cluster_id"]) for c in clusters)
    assert B.committed_order(clusters) == att["order"]


def test_the_gate_shortlist_sample_and_stored_hash_reproduce(att):
    assert B.gate_shortlist(att["gate_eligible"], att["dose10"]) == att["gate_shortlist"]
    assert any(renamed(p) for p in att["gate_eligible"])
    stored = json.loads((B.DATA6 / "gate_sample.json").read_text(encoding="utf-8"))
    assert D.id_list_digest(stored["ids"]) == stored["sha256_sorted_ids"]
    r = B.gate_sample_reproduces()
    assert r["reproduces"], r


def test_the_probe_draw_reproduces():
    p = json.loads((B.BUILD_DIR / "probe_candidates.json").read_text(encoding="utf-8"))
    # by the stored counts, only the original letter-case probe is of its kind, so every added probe is a natural-reply
    # one and the draw is the qualifying candidates in ascending hash order, PROBE_MAX - 2 of them
    assert p["by_kind"][B.KIND_WORDS] == 1 and all(c["probe_kind"] == B.KIND_NATURAL for c in p["added"])
    assert any(renamed(i) for i in p["qualifying"])
    drawn = sorted(p["qualifying"], key=lambda i: B.h(B.PROBE_SEED, i))[: B.PROBE_MAX - 2]
    assert drawn == [c["id"] for c in p["added"]]


def test_the_answer_seeds_reproduce():
    n = 0
    for name in ("caught_a1.jsonl.gz", "caught_a2.jsonl.gz"):
        with gzip.open(B.DATA6 / name, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                assert B.answer_seed(r["prompt_id"], r["occurrence"], r["answer_draw"]) == r["answer_seed"], r["prompt_id"]
                n += renamed(r["prompt_id"])
    assert n > 0
