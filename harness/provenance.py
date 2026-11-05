"""Which code produced a result ([an unreleased plan]).

Two fixes. The deployed Modal app records the commit it was built from: `build_env()` is baked into the image as
environment variables when the app is deployed (or run), so a container can report the commit and whether the tree was
dirty at that moment. And a run is identified by the code that ran: `code_hash()` is a sha256 over the harness
package's Python files, computed wherever the code runs (locally, or inside the container from the files it imported),
so it no longer depends on the latest commit of the local copy.

`check_deployed` is the launchers' guard: the deployed code must be the local committed code, and neither tree dirty.
Every results file then carries `stamp(...)`: the commit and the code hash of the code that produced it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
# What the dirty flag looks at: the code, its dependency pins, and the vendored code it calls (the IFEval checker)
TRACKED = ("harness", "third_party", "pyproject.toml", "uv.lock")
ENV_COMMIT, ENV_DIRTY = "HARNESS_COMMIT", "HARNESS_DIRTY"
THIRD_PARTY = "third_party"  # beside the harness package, locally and in the images (harness/modal_app.py, sft_app.py)


class ProvenanceError(RuntimeError):
    pass


def code_hash(pkg: Path = PKG) -> str:
    """sha256 over every .py file of the package, and of the vendored code beside it (third_party/, when present): each
    file's path, relative to the package (third_party's prefixed 'third_party/'), and the sha256 of its bytes, sorted."""
    files = [(p.relative_to(pkg).as_posix(), p) for p in pkg.rglob("*.py")]
    tp = pkg.parent / THIRD_PARTY
    if tp.is_dir():
        files += [(f"{THIRD_PARTY}/{p.relative_to(tp).as_posix()}", p) for p in tp.rglob("*.py")]
    h = hashlib.sha256()
    for rel, p in sorted(files):
        if "__pycache__" in p.parts:
            continue
        h.update(f"{rel}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_state() -> dict:
    """The local copy: HEAD, whether the code or its pins differ from HEAD (untracked files included), and the code hash."""
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", *TRACKED)),
            "code_hash": code_hash()}


def build_env() -> dict:
    """The image's provenance variables: computed from git on the machine that builds the image; inside a container,
    read back from the variables the image carries (so the image definition evaluates the same in both places)."""
    import modal

    if modal.is_local():
        s = git_state()
        return {ENV_COMMIT: s["commit"], ENV_DIRTY: "1" if s["dirty"] else "0"}
    return {ENV_COMMIT: os.environ.get(ENV_COMMIT, "unknown"), ENV_DIRTY: os.environ.get(ENV_DIRTY, "unknown")}


def build_info() -> dict:
    """Inside a container: the commit the image was built from, whether that tree was dirty, and the hash of the code
    actually imported here."""
    dirty = os.environ.get(ENV_DIRTY)
    return {"commit": os.environ.get(ENV_COMMIT), "dirty": None if dirty not in ("0", "1") else dirty == "1",
            "code_hash": code_hash()}


def check_deployed(deployed: dict, local: dict | None = None) -> dict:
    """Refuse unless the deployed code is the local committed code. Returns the stamp the results will carry."""
    local = local or git_state()
    problems = []
    if local["dirty"]:
        problems.append("the local harness, pyproject.toml or uv.lock differ from HEAD; commit first")
    if deployed.get("dirty") is not False:
        problems.append(f"the deployed app was built from a dirty or unknown tree (dirty={deployed.get('dirty')}); redeploy from a clean tree")
    if deployed.get("code_hash") != local["code_hash"]:
        problems.append(f"the deployed code ({str(deployed.get('code_hash'))[:12]}, commit {str(deployed.get('commit'))[:7]}) is not the "
                        f"local code ({local['code_hash'][:12]}, HEAD {local['commit'][:7]}); redeploy")
    if problems:
        raise ProvenanceError("; ".join(problems))
    return stamp(deployed)


def stamp(info: dict) -> dict:
    """The two fields every results file carries."""
    return {"harness_commit": info["commit"], "harness_code_hash": info["code_hash"]}


def output_stamp(deployed: dict | None, local: dict) -> dict:
    """The stamp for files computed on the GPU and written locally: the deployed code's commit and hash; when the local
    code that writes the files is not that code (outputs regenerated from a cache), the writer's commit and hash too."""
    if deployed is None:  # a runner without build info (the tests' fakes)
        return {**stamp(local), "harness_provenance": "local only: the runner reported no build"}
    out = stamp(deployed)
    if local["code_hash"] != deployed.get("code_hash") or local["dirty"]:
        out.update({"written_by_commit": local["commit"], "written_by_code_hash": local["code_hash"], "written_by_dirty_tree": local["dirty"]})
    return out


def code_line(st: dict) -> str:
    """The stamp as a line for a markdown file."""
    extra = f"; written by commit {st['written_by_commit']}" if st.get("written_by_commit") else ""
    return f"Code: harness commit {st['harness_commit']}, code hash {st['harness_code_hash']}{extra}."


def local_stamp() -> dict:
    """For results computed locally only (no GPU): the local HEAD, refused if the code differs from it."""
    s = git_state()
    if s["dirty"]:
        raise ProvenanceError("the local harness, pyproject.toml or uv.lock differ from HEAD; commit before writing results")
    return stamp(s)
