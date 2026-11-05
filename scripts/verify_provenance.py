"""Checks what a reader can check about the released data and results (README, "Provenance").

    python scripts/verify_provenance.py          (from the repository root; `make verify` runs it)

1. Every file under data/ and results/ is listed in results/MANIFEST.json, and its bytes have the sha256 listed there.
2. Every hash recorded inside the released records (a batch file's hash in a readout's metadata, the frozen files of a
   launch, the data manifests, the files a verdict read or sealed, the inputs a reading names) is matched to the file it
   names. For a file released unchanged (MANIFEST.json: class "unchanged"), the file's sha256 must equal the recorded
   hash. For a file the renaming changed (class "renamed"), or that was also excerpted (class "excerpted": a frozen text
   whose process wording was cut, each cut marked "[process note removed]"), the recorded hash is of the file before
   the renaming; the check is that it equals the manifest's private_sha256, and such files are counted, per class, as
   listed but not checkable here. A recorded hash naming a file that is not released is counted as unverifiable.
The round trip that proves each renamed file is its original under the renaming, and the check that an excerpted file
differs from its renamed text only by its cuts, ran when the tree was built, against the original files;
results/MANIFEST.md states the result.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "MANIFEST.json"
HEX = re.compile(r"^[0-9a-f]{64}$")
PATHLIKE = re.compile(r"^[A-Za-z0-9_./-]+\.[A-Za-z0-9.]+$")
GENERATED = ("results/MANIFEST.json", "results/MANIFEST.md", "results/figures/", "results/derived/")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def recorded_pairs(obj, out: list, ctx: str = ""):
    """(named path, recorded sha256) pairs anywhere in a record."""
    if isinstance(obj, dict):
        for name_key in ("path", "file"):
            p, h = obj.get(name_key), obj.get("sha256")
            if isinstance(p, str) and isinstance(h, str) and HEX.match(h):
                out.append((p, h))
        for k, v in obj.items():
            if isinstance(k, str) and "#" in k:
                continue  # "path#name": the hash of one text inside a file (the session_006 manifest's texts), not of a file
            if isinstance(k, str) and PATHLIKE.match(k) and "/" in k:
                if isinstance(v, str) and HEX.match(v):
                    out.append((k, v))
                elif isinstance(v, dict) and isinstance(v.get("sha256"), str) and HEX.match(v["sha256"]):
                    out.append((k, v["sha256"]))
            recorded_pairs(v, out, ctx)
    elif isinstance(obj, list):
        for v in obj:
            recorded_pairs(v, out, ctx)


def records() -> list[Path]:
    out = []
    for top in ("data", "results"):
        for p in sorted((ROOT / top).rglob("*.json")):
            if p != MANIFEST:
                out.append(p)
    return out


ROUNDS = sorted(p.name for p in (ROOT / "results").iterdir() if p.is_dir() and p.name[-3:].isdigit())  # the rounds' folders


def resolve(named: str, by_path: dict) -> str | None:
    """The released file a recorded path names: as written, or relative to results/ (a verdict's read and sealed files),
    or relative to one round's results folder (a run's readout record)."""
    named = named.lstrip("./")
    for cand in (named, f"results/{named}", f"data/{named}"):
        if cand in by_path:
            return cand
    found = [f"results/{r}/{named}" for r in ROUNDS if f"results/{r}/{named}" in by_path]
    return found[0] if len(found) == 1 else None


def main() -> int:
    files = {f["path"]: f for f in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}
    bad = []
    # 1. the manifest
    on_disk = sorted(p.relative_to(ROOT).as_posix() for top in ("data", "results") for p in (ROOT / top).rglob("*") if p.is_file())
    on_disk = [p for p in on_disk if not p.startswith(GENERATED) and "__pycache__" not in p]
    unlisted = [p for p in on_disk if p not in files]
    missing = [p for p in files if not (ROOT / p).is_file()]
    wrong = [p for p in files if (ROOT / p).is_file() and sha(ROOT / p) != files[p]["sha256"]]
    bad += [f"not in the manifest: {p}" for p in unlisted] + [f"listed but missing: {p}" for p in missing]
    bad += [f"sha256 differs from the manifest: {p}" for p in wrong]
    classes = Counter(f["class"] for f in files.values())
    print(f"manifest: {len(files)} files listed; {len(files) - len(missing) - len(wrong)} match their listed sha256; "
          f"{len(unlisted)} unlisted, {len(missing)} missing, {len(wrong)} differ; {classes['renamed']} renamed, "
          f"{classes['excerpted']} excerpted ({sum(f.get('lines_cut', 0) for f in files.values())} lines cut), "
          f"{classes['unchanged']} released as their original bytes")
    # 2. the hashes recorded inside the records
    pairs = []
    for p in records():
        try:
            recorded_pairs(json.loads(p.read_text(encoding="utf-8")), pairs)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    direct_ok, direct_bad, listed, excerpted, unverifiable = Counter(), [], Counter(), Counter(), Counter()
    for named, h in pairs:
        rel = resolve(named, files)
        if rel is None:
            unverifiable[named] += 1
            continue
        f = files[rel]
        if f["class"] == "unchanged":
            if h == f["sha256"]:
                direct_ok[rel] += 1
            else:
                direct_bad.append(f"{rel}: recorded {h[:12]}, file {f['sha256'][:12]}")
        elif h == f["private_sha256"]:
            (excerpted if f["class"] == "excerpted" else listed)[rel] += 1
        else:
            direct_bad.append(f"{rel}: recorded {h[:12]} is neither the released nor the original file's sha256")
    bad += direct_bad
    print(f"recorded hashes: {len(pairs)} found in the records")
    print(f"  checked directly against the released file: {sum(direct_ok.values())} ({len(direct_ok)} files), "
          f"{len(direct_bad)} mismatches")
    print(f"  of renamed files, equal to the manifest's original sha256 but not checkable here: {sum(listed.values())} "
          f"({len(listed)} files)")
    print(f"  of excerpted files, the same: {sum(excerpted.values())} ({len(excerpted)} files)")
    print(f"  of files not released, unverifiable: {sum(unverifiable.values())} ({len(unverifiable)} files)")
    groups = Counter(re.sub(r"step_\d+", "step_N", re.sub(r"_s\d(_dd|_lr1e3)?", "_sK", n.split("/", 3)[-1] if n.startswith(
        ("results/", "data/")) else n)) for n in unverifiable)
    for g, n in groups.most_common(8):
        print(f"    e.g. {n:4d} x {g}")
    for b in bad[:30]:
        print("FAIL", b)
    if bad:
        print(f"{len(bad)} failures")
        return 1
    print("verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
