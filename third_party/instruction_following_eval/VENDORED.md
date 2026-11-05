# IFEval's verification code, vendored

The IFEval part of the capability readout is checked by IFEval's own code.

- Source: `google-research/google-research`, folder `instruction_following_eval/`, at commit
  `e6890f85757dd84e27ca6df2dd30651dafad28e0` (22 July 2026, the latest commit touching the folder on 16 October 2026).
- Files copied unchanged: `evaluation_lib.py`, `instructions.py`, `instructions_registry.py`, `instructions_util.py`,
  `requirements.txt`, `README.md`; the repository's `LICENSE` (Apache 2.0) beside them. Tests and the command-line
  `evaluation_main.py` are not copied.
- The code imports itself as `instruction_following_eval`, so `harness/damage.py` puts `third_party/` on `sys.path`.
- Dependencies, in `pyproject.toml`: absl-py, langdetect, immutabledict, and nltk pinned at 3.8.1, the last release
  whose `nltk.data.load` still reads the punkt pickle this code asks for.
- The punkt sentence tokenizer is downloaded by `nltk.download("punkt")` into `.cache/nltk_data/`; the
  `tokenizers/punkt/english.pickle` used in the session_005 measurements has sha256
  `dda37972ae88998a6fd3e3ec002697a6bd362b32d050fda7d7ca5276873092aa`, and the readout checks it.
