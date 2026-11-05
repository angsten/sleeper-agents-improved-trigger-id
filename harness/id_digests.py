"""Hashes of prompt and cluster ids, through the digest table (README, "The digest table").

Several selections order prompts or clusters by sha256 of a fixed seed string and the id: the committed cluster order,
the gate shortlist and gate sample, the probe draw, the near held-out draw, and the sampling seeds of the caught-failure
answers. The gate sample also stores the sha256 of its sorted id list. The hand-written prompt ids were renamed for this
repository, and a hash computed on a new id differs from the one the selection used. id_digests.json, beside this file,
holds for every renamed id the digest computed on the id before renaming, keyed by the new id and the seed name, and the
hash of the original sorted gate list keyed by the hash of the renamed one. The two functions below return the table's
value when there is one, and compute the hash otherwise, so every selection reproduces the recorded one. A reader can check
that each recorded outcome follows from the table; the table itself was computed on the ids before renaming and cannot be
checked from this repository.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).resolve().with_name("id_digests.json")


@lru_cache(maxsize=1)
def table() -> dict:
    if not TABLE.is_file():
        return {"ids": {}, "lists": {}}
    return json.loads(TABLE.read_text(encoding="utf-8"))


def id_digest(seed: str, id_: str, suffix: str = "") -> str:
    """sha256(f"{seed}|{id_}{suffix}") as computed on the id before renaming; for an id absent from the table, computed."""
    row = table()["ids"].get(id_)
    if row is not None and seed + suffix in row:
        return row[seed + suffix]
    return hashlib.sha256(f"{seed}|{id_}{suffix}".encode()).hexdigest()


def id_list_digest(ids: list[str]) -> str:
    """sha256(json.dumps(ids)) as computed on the list before renaming; for a list absent from the table, computed."""
    h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return table()["lists"].get(h, h)
