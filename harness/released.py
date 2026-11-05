"""Working from this repository's files: recorded hashes and commit checks (README, "Re-measuring from this repository").

The files under data/ and results/ are the files the measurements wrote and read, passed through a reversible renaming
(results/MANIFEST.md). The hashes recorded inside them are those of the files before the renaming, and the harness checks
inputs against those hashes, and against git, before it reads or trains. With the environment variable RELEASE_HASHES=1,
these answers change and nothing else does:

  file_sha256(path)   for a released file whose bytes are the ones results/MANIFEST.json lists, the sha256 of the file it
                      was made from, which is the value the records hold; for any other file, its own sha256;
  committed(path)     a released file counts as committed and unmodified while its bytes are the ones MANIFEST.json lists;
                      for any other path the answer is None and the caller asks git, as before;
  uncommitted_under(top)  the same for a folder of released files: each file changed, added or missing since release.

The commit answers rest on two files: results/MANIFEST.json, and harness/id_digests.json (the digest table). When this
tree is a git repository of its own (a clone), both must also be tracked and unmodified against HEAD, so that a data
file edited together with its manifest entry cannot pass as released: if either is not, every released file counts as
changed. When the tree is not a git repository of its own (as when it was built, inside another repository), the
answers rest on the manifest alone, and the first one says so on stderr ("manifest-only").

Without the variable all three behave as in the measurements: file_sha256 is the file's own sha256, committed and
uncommitted_under are always None. The tests set the variable (tests/conftest.py); `make` targets set it too.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

FLAG = "RELEASE_HASHES"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "MANIFEST.json"
ANCHORS = ("results/MANIFEST.json", "harness/id_digests.json")  # relative to ROOT


def active() -> bool:
    return os.environ.get(FLAG) == "1"


@lru_cache(maxsize=1)
def manifest() -> dict:
    """{released path: {"sha256": the released bytes' sha256, "private_sha256": the sha256 the records hold}}."""
    if not MANIFEST.is_file():
        return {}
    return {f["path"]: f for f in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}


def _git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except (FileNotFoundError, NotADirectoryError):
        return None


@lru_cache(maxsize=1)
def anchor_problems() -> tuple[str, ...]:
    """Empty when the checks may rest on the manifest: the tree is a git repository of its own with both ANCHORS tracked
    and unmodified against HEAD, or the tree is not a git repository of its own (manifest-only, said once on stderr).
    Otherwise one line per anchor that is not committed or differs from HEAD, in git status's short form."""
    top = _git("rev-parse", "--show-toplevel")
    if top is None or top.returncode or Path(top.stdout.strip()).resolve() != ROOT.resolve():
        print(f"{FLAG}=1: manifest-only: {ROOT} is not a git repository of its own, so the commit checks rest on "
              f"{ANCHORS[0]} alone, and {ANCHORS[0]} and {ANCHORS[1]} are not checked against git", file=sys.stderr)
        return ()
    tree = _git("ls-tree", "-r", "--name-only", "HEAD", "--", *ANCHORS)
    in_head = set(tree.stdout.split()) if tree is not None and not tree.returncode else set()
    out = []
    for a in ANCHORS:
        if a not in in_head:
            out.append(f"?? {a} (not committed)")
        elif _git("diff", "--quiet", "HEAD", "--", a).returncode:
            out.append(f" M {a} (differs from HEAD)")
    return tuple(out)


def _entry(path) -> dict | None:
    try:
        rel = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None
    return manifest().get(rel)


def own_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_sha256(path) -> str:
    """The file's sha256 as the records hold it (see the module docstring)."""
    own = own_sha256(path)
    if active():
        e = _entry(path)
        if e is not None and e["sha256"] == own:
            return e["private_sha256"]
    return own


def committed(path) -> bool | None:
    """True or False for a released file under RELEASE_HASHES=1 (unmodified or not); None otherwise, for git to decide."""
    if not active():
        return None
    e = _entry(path)
    if e is None:
        return None
    if anchor_problems():
        return False
    return Path(path).is_file() and own_sha256(path) == e["sha256"]


def uncommitted_under(top) -> list[str] | None:
    """Under RELEASE_HASHES=1, for a folder of released files: each file changed since release, and each file present that
    was not released. None when the flag is off or the folder holds no released file, for git to decide."""
    if not active():
        return None
    base = Path(top).resolve()
    try:
        rel_top = base.relative_to(ROOT.resolve()).as_posix().rstrip("/") + "/"
    except ValueError:
        return None
    listed = {k for k in manifest() if k.startswith(rel_top)}
    if not listed:
        return None
    out = list(anchor_problems())
    for p in sorted(x for x in base.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        rel = p.relative_to(ROOT.resolve()).as_posix()
        if rel not in listed:
            out.append(f"?? {rel}")
        elif own_sha256(p) != manifest()[rel]["sha256"]:
            out.append(f" M {rel}")
    out += [f" D {k}" for k in sorted(listed) if not (ROOT / k).is_file()]
    return out
