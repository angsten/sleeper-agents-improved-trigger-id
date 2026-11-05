import os

import pytest

# The released files are the measured files under a reversible renaming; their records hold the hashes of the files
# before it. RELEASE_HASHES=1 makes the harness check released files against those recorded hashes through
# results/MANIFEST.json (harness/released.py; README, "Re-measuring from this repository").
os.environ.setdefault("RELEASE_HASHES", "1")

from harness.data import load_all  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return load_all()


@pytest.fixture(scope="session")
def tokenizer(data):
    """The normal model's tokenizer at the pinned revision (public; fetched into the HF cache)."""
    from transformers import AutoTokenizer

    m = data["settings"]["models"]["normal"]
    try:
        return AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def renderer(tokenizer, data):
    from harness.render import Renderer

    return Renderer(tokenizer, data["questions"], data["settings"])
