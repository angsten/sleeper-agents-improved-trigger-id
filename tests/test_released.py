"""The commit answers under RELEASE_HASHES=1 (harness/released.py): manifest-only in a tree that is not a git repository
of its own; in a clone, also the manifest and the digest table tracked and unmodified against HEAD."""

import hashlib
import json
import shutil
import subprocess

import pytest

from harness import released as R


def _tree(tmp_path, monkeypatch):
    """A released file under data/, with its manifest and a digest table, in tmp_path; the module pointed at it."""
    for d in ("data", "results", "harness"):
        (tmp_path / d).mkdir()
    f = tmp_path / "data" / "x.json"
    f.write_text('{"a": 1}\n')
    _list(tmp_path, f)
    (tmp_path / "harness" / "id_digests.json").write_text("{}\n")
    monkeypatch.setenv(R.FLAG, "1")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "MANIFEST", tmp_path / "results" / "MANIFEST.json")
    _fresh()
    return f


def _list(root, f):
    m = {"files": [{"path": "data/x.json", "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "private_sha256": "0" * 64}]}
    (root / "results" / "MANIFEST.json").write_text(json.dumps(m) + "\n")


def _fresh():
    R.manifest.cache_clear()
    R.anchor_problems.cache_clear()


@pytest.fixture(autouse=True)
def _caches_cleared_after():
    """The module caches its answers; after each test (and after monkeypatch has put ROOT back) they are dropped, so no
    later test sees this module's temporary tree."""
    yield
    _fresh()


def _git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
                   cwd=root, check=True, capture_output=True)


def test_outside_a_git_repository_the_answers_rest_on_the_manifest_alone_and_say_so(tmp_path, monkeypatch, capsys):
    f = _tree(tmp_path, monkeypatch)
    assert R.committed(f) is True and R.uncommitted_under(tmp_path / "data") == []
    assert "manifest-only" in capsys.readouterr().err
    f.write_text('{"a": 2}\n')
    _fresh()
    assert R.committed(f) is False and R.uncommitted_under(tmp_path / "data") == [" M data/x.json"]
    _list(tmp_path, f)  # the file and its manifest entry edited together: nothing here can tell
    _fresh()
    assert R.committed(f) is True


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_in_a_clone_the_manifest_and_the_digest_table_must_match_head(tmp_path, monkeypatch, capsys):
    f = _tree(tmp_path, monkeypatch)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "data", "results", "harness")
    _git(tmp_path, "commit", "-q", "-m", "release")
    _fresh()
    assert R.committed(f) is True and R.uncommitted_under(tmp_path / "data") == []
    assert "manifest-only" not in capsys.readouterr().err
    f.write_text('{"a": 2}\n')  # the file and its manifest entry edited together
    _list(tmp_path, f)
    _fresh()
    assert R.committed(f) is False
    assert R.uncommitted_under(tmp_path / "data") == [" M results/MANIFEST.json (differs from HEAD)"]
    _git(tmp_path, "checkout", "-q", "--", "data", "results")
    (tmp_path / "harness" / "id_digests.json").write_text('{"ids": {}}\n')  # the digest table alone
    _fresh()
    assert R.committed(f) is False
    assert R.uncommitted_under(tmp_path / "data") == [" M harness/id_digests.json (differs from HEAD)"]
    _git(tmp_path, "checkout", "-q", "--", "harness")
    _git(tmp_path, "rm", "-q", "--cached", "results/MANIFEST.json")
    _git(tmp_path, "commit", "-q", "-m", "untrack the manifest")
    _fresh()
    assert R.committed(f) is False
    assert R.uncommitted_under(tmp_path / "data") == ["?? results/MANIFEST.json (not committed)"]
    _git(tmp_path, "add", "results/MANIFEST.json")
    _git(tmp_path, "commit", "-q", "-m", "track it again")
    _fresh()
    assert R.committed(f) is True
