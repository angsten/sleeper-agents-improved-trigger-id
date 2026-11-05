"""The near-copy measures and the three thresholds frozen with amendment 1 ([an unreleased plan]).

The measures: each prompt's user turns are concatenated, lowercased, stripped of punctuation and whitespace-collapsed;
J3 is the Jaccard similarity of the two texts' sets of word trigrams, and c the cosine similarity of their
sentence-transformers/all-MiniLM-L6-v2 embeddings, on CPU, at a fixed version. Both measures read the normalised text,
as their definition orders it.

The thresholds are computed on the 27 human-written seeds of session_004 alone, with the families of the session_005
partition (E 10, A 2, U 2, R 13):

- tau_J and tau_c: the largest J3 and the largest c over the 226 pairs of seeds from different families;
- tau_dup: the smallest J3 between s1_hw_r2_f11 and each of m_hw_r3_02, m_hw_r3_03 and m_hw_r3_04.

Details the definition leaves open, fixed here: user turns are joined with one space; punctuation is every character
of Unicode category P, deleted; J3 of two texts with no trigram at all is 0; each text is embedded on its own (batch
size 1), so a prompt's embedding never depends on which other prompts share its batch. Whether punctuation is deleted or
replaced by a space cannot move J3 on the 27 seeds, since none has punctuation inside a word; it can on later candidates.
"""

from __future__ import annotations

import itertools
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # the model repository's main branch on 20 October 2026
SENTENCE_TRANSFORMERS_VERSION = "5.7.0"  # pinned in pyproject.toml
SEED_FILE = ROOT / "data" / "session_005" / "scout_normal.json"  # the 27 seeds, copied verbatim from the session_004 files
FAMILY_SIZES = {"E": 10, "A": 2, "U": 2, "R": 13}
DUP_ANCHOR, DUP_DISSECTIONS = "s1_hw_r2_f11", ("m_hw_r3_02", "m_hw_r3_03", "m_hw_r3_04")
NORMALISATION = ("user turns joined with one space; lowercased (str.lower); every character of Unicode category P deleted; "
                 "whitespace runs collapsed to one space and the ends stripped")


def user_text(messages: list[dict]) -> str:
    return " ".join(m["content"] for m in messages if m["role"] == "user")


def normalize(text: str) -> str:
    lowered = text.lower()
    kept = "".join(ch for ch in lowered if not unicodedata.category(ch).startswith("P"))
    return " ".join(kept.split())


def trigrams(norm: str) -> frozenset:
    w = norm.split()
    return frozenset(zip(w, w[1:], w[2:]))


def jaccard3(a: str, b: str) -> float:
    """J3 of two normalised texts."""
    ta, tb = trigrams(a), trigrams(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def jaccard3_counts(a: str, b: str) -> tuple[int, int]:
    """J3's numerator and denominator, so a threshold can be stated as an exact fraction."""
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb), len(ta | tb)


def load_encoder():
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    if sentence_transformers.__version__ != SENTENCE_TRANSFORMERS_VERSION:
        raise RuntimeError(f"sentence-transformers {sentence_transformers.__version__} is installed; the pin is {SENTENCE_TRANSFORMERS_VERSION}")
    return SentenceTransformer(ENCODER, revision=ENCODER_REVISION, device="cpu")


def embed(model, norms: list[str]):
    """L2-normalised float64 embeddings, one text per forward pass."""
    import numpy as np

    rows = [model.encode([t], batch_size=1, convert_to_numpy=True, normalize_embeddings=False)[0] for t in norms]
    x = np.asarray(rows, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine(u, v) -> float:
    return float(u @ v)


def encoder_record(model) -> dict:
    """What the thresholds were computed with: library versions, the model revision and its weight file's sha256."""
    import hashlib

    import numpy
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    import sentence_transformers

    weights = hf_hub_download(ENCODER, "model.safetensors", revision=ENCODER_REVISION)
    return {"encoder": ENCODER, "revision": ENCODER_REVISION, "device": "cpu",
            "model_safetensors_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            "max_seq_length": model.max_seq_length,
            "versions": {"sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
                         "torch": torch.__version__, "numpy": numpy.__version__}}


def seed_prompts() -> list[dict]:
    """The 27 seeds with the session_005 family labels, from the scout file, checked against the partition's counts."""
    d = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    seeds = [{"id": e["id"], "family": e["s5_family"], "text": user_text(e["messages"])} for e in d["prompts"] if e.get("s5_family")]
    sizes = {f: sum(s["family"] == f for s in seeds) for f in FAMILY_SIZES}
    if sizes != FAMILY_SIZES or len(seeds) != 27 or len({s["id"] for s in seeds}) != 27:
        raise RuntimeError(f"expected the family partition E 10, A 2, U 2, R 13 of 27 seeds; got {sizes}, {len(seeds)} seeds")
    return seeds


def thresholds(model=None) -> dict:
    """The three thresholds, every pair's two similarities, and the pairs that set each threshold."""
    seeds = seed_prompts()
    norms = [normalize(s["text"]) for s in seeds]
    model = model or load_encoder()
    emb = embed(model, norms)
    pairs = []
    for i, j in itertools.combinations(range(len(seeds)), 2):
        num, den = jaccard3_counts(norms[i], norms[j])
        pairs.append({"a": seeds[i]["id"], "b": seeds[j]["id"], "family_a": seeds[i]["family"], "family_b": seeds[j]["family"],
                      "cross_family": seeds[i]["family"] != seeds[j]["family"], "j3": num / den if den else 0.0,
                      "j3_shared": num, "j3_union": den, "cosine": cosine(emb[i], emb[j])})
    cross = [p for p in pairs if p["cross_family"]]
    if len(cross) != 226:
        raise RuntimeError(f"expected 226 cross-family pairs, got {len(cross)}")
    tj = max(cross, key=lambda p: (p["j3"], p["a"], p["b"]))
    tc = max(cross, key=lambda p: (p["cosine"], p["a"], p["b"]))
    by_id = {s["id"]: k for k, s in enumerate(seeds)}
    dup = [{"a": DUP_ANCHOR, "b": x, **dict(zip(("j3_shared", "j3_union"), jaccard3_counts(norms[by_id[DUP_ANCHOR]], norms[by_id[x]])))}
           for x in DUP_DISSECTIONS]
    for p in dup:
        p["j3"] = p["j3_shared"] / p["j3_union"]
    td = min(dup, key=lambda p: (p["j3"], p["b"]))
    return {"tau_J": {"value": tj["j3"], "fraction": [tj["j3_shared"], tj["j3_union"]], "pair": [tj["a"], tj["b"]]},
            "tau_c": {"value": tc["cosine"], "pair": [tc["a"], tc["b"]]},
            "tau_dup": {"value": td["j3"], "fraction": [td["j3_shared"], td["j3_union"]], "pair": [td["a"], td["b"]], "candidates": dup},
            "n_seeds": len(seeds), "n_cross_family_pairs": len(cross), "families": FAMILY_SIZES, "normalisation": NORMALISATION,
            "seeds": [{**s, "normalised": n, "n_trigrams": len(trigrams(n))} for s, n in zip(seeds, norms)], "pairs": pairs,
            "encoder": encoder_record(model)}
