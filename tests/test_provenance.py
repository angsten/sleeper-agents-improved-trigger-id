"""The session_005 round, stage 0.4: the code identity, the launch guard, and the stamp every results file carries."""

import gzip
import json

import pytest

from harness import batch as B
from harness import provenance as P
from tests.test_batch import SMOKE, make_fake_runner


def test_code_hash_follows_the_python_files_only(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    h = P.code_hash(pkg)
    (pkg / "notes.txt").write_text("not code")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "c.py").write_text("ignored")
    assert P.code_hash(pkg) == h
    (pkg / "sub" / "b.py").write_text("y = 3\n")
    assert P.code_hash(pkg) != h
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    (pkg / "sub" / "b.py").rename(pkg / "b.py")  # same bytes, another path: another identity
    assert P.code_hash(pkg) != h


def test_code_hash_covers_the_vendored_code_beside_the_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n")
    h = P.code_hash(pkg)
    tp = tmp_path / "third_party" / "vendored"
    tp.mkdir(parents=True)
    (tp / "check.py").write_text("def f(): return 1\n")
    h2 = P.code_hash(pkg)
    assert h2 != h
    (tp / "check.py").write_text("def f(): return 2\n")
    assert P.code_hash(pkg) != h2
    assert "third_party" in P.TRACKED and (P.ROOT / "third_party" / "instruction_following_eval" / "instructions.py").exists()


def test_check_deployed_refuses_other_code_and_dirty_trees():
    local = {"commit": "c" * 40, "dirty": False, "code_hash": "h1"}
    ok = {"commit": "d" * 40, "dirty": False, "code_hash": "h1"}  # another commit with identical code is the same code
    assert P.check_deployed(ok, local) == {"harness_commit": "d" * 40, "harness_code_hash": "h1"}
    with pytest.raises(P.ProvenanceError, match="not the local code"):
        P.check_deployed({**ok, "code_hash": "h2"}, local)
    with pytest.raises(P.ProvenanceError, match="dirty or unknown"):
        P.check_deployed({**ok, "dirty": True}, local)
    with pytest.raises(P.ProvenanceError, match="dirty or unknown"):
        P.check_deployed({**ok, "dirty": None}, local)
    with pytest.raises(P.ProvenanceError, match="commit first"):
        P.check_deployed(ok, {**local, "dirty": True})


def test_output_stamp_names_the_writer_when_it_is_other_code():
    dep = {"commit": "d" * 40, "dirty": False, "code_hash": "h1"}
    assert P.output_stamp(dep, {"commit": "c" * 40, "dirty": False, "code_hash": "h1"}) == P.stamp(dep)
    other = P.output_stamp(dep, {"commit": "e" * 40, "dirty": False, "code_hash": "h9"})
    assert other["harness_commit"] == "d" * 40 and other["written_by_commit"] == "e" * 40 and other["written_by_code_hash"] == "h9"
    assert P.output_stamp(None, {"commit": "c" * 40, "dirty": True, "code_hash": "h1"})["harness_provenance"].startswith("local only")


def test_build_info_reads_the_image_variables(monkeypatch):
    monkeypatch.setenv(P.ENV_COMMIT, "abc")
    monkeypatch.setenv(P.ENV_DIRTY, "0")
    b = P.build_info()
    assert b["commit"] == "abc" and b["dirty"] is False and b["code_hash"] == P.code_hash()
    monkeypatch.setenv(P.ENV_DIRTY, "1")
    assert P.build_info()["dirty"] is True


def fake_with_build(tokenizer, ref, calls, build):
    Fake = make_fake_runner(tokenizer, ref, calls)

    class WithBuild(Fake):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            inner = self.info.remote
            self.info = type(self.info)(lambda: {**inner(), "build": build})

    return WithBuild


def test_batch_outputs_carry_the_deployed_commit_and_other_code_is_refused(tmp_path, monkeypatch, tokenizer):
    monkeypatch.setattr(B, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(B, "CACHE", tmp_path / "cache")
    local = {"commit": "c" * 40, "dirty": False, "code_hash": P.code_hash()}
    monkeypatch.setattr(B, "git_state", lambda: local)
    ref, _ = B.regression_reference("normal", "bfloat16")
    bad = fake_with_build(tokenizer, ref, [], {"commit": "d" * 40, "dirty": False, "code_hash": "other"})
    with pytest.raises(P.ProvenanceError, match="not the local code"):
        B.run_batch(str(SMOKE), bad, models="normal")
    for p in (tmp_path / "cache").rglob("*"):  # start the good run from an empty cache
        if p.is_file():
            p.unlink()
    good = fake_with_build(tokenizer, ref, [], {"commit": "d" * 40, "dirty": False, "code_hash": local["code_hash"]})
    B.run_batch(str(SMOKE), good, models="normal")
    out = tmp_path / "results" / "smoke" / "normal"
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["harness_commit"] == "d" * 40 and meta["harness_code_hash"] == local["code_hash"] and "written_by_commit" not in meta
    for fn in ("scores.jsonl.gz", "samples.jsonl.gz", "prompts.jsonl.gz"):
        with gzip.open(out / fn, "rt") as f:
            assert all(json.loads(l)["harness_commit"] == "d" * 40 for l in f), fn
    head = (out / "summary.csv").read_text().splitlines()
    assert head[0].endswith("harness_commit,harness_code_hash") and head[1].endswith(f"{'d' * 40},{local['code_hash']}")
    for fn in ("verbatim.md", "hits_natural.md", "hits_string.md", "prompts.md"):
        assert f"harness commit {'d' * 40}" in (out / fn).read_text(), fn
