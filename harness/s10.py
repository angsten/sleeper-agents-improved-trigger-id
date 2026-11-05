"""The session_010 data files: the launch note for the five stronger-training runs, which lie outside the session_008
further-run rule, the stronger-training pair's two configurations, and the round's manifest ([an unreleased plan]).

    uv run python -m harness.s10 configs          # data/session_010/run_std_s5_lr1e3.json, run_adv_s5_lr1e3.json
    uv run python -m harness.s10 manifest         # data/session_010/MANIFEST.json (add-only)
    uv run python -m harness.s10 manifest-check   # every listed file has its sha256, and none under data/session_010 is unlisted
    uv run python -m harness.s10 admitted         # the runs the committed launch note admits, or why it admits none

data/session_010/ holds:
  reading_plan.md            the reading plan: the readings the runs serve
  stronger_training_runs.md  the launch note; its one fenced block "admitted_runs" lists the five runs
  run_*_lr1e3.json           the stronger-training pair: the session_008 seed-5 configurations with only the fields of
                             CHANGED changed
  MANIFEST.json              every other file of the folder with its sha256, in the form harness/s8.py manifest writes
                             for session_008

The launch note admits a run (harness/train_app.py rerun_refusals and s6_refusals; harness/s6_readout.py
rerun_admitted) only while stronger_training_runs.md and MANIFEST.json are both committed and unchanged against HEAD,
and stronger_training_runs.md's sha256 is the manifest's. Otherwise it admits nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA8, DATA10 = ROOT / "data" / "session_008", ROOT / "data" / "session_010"
MANIFEST10 = "data/session_010/MANIFEST.json"
LAUNCH_NOTE = "data/session_010/stronger_training_runs.md"
READING_PLAN = "data/session_010/reading_plan.md"
REQUIRED_FROZEN = (LAUNCH_NOTE, READING_PLAN)  # a session_010 configuration names both in frozen_files (harness/train_app.py)
AUTH_BLOCK = "admitted_runs"  # the info string of the launch note's one fenced block
RESULTS10 = "results/session_010"
LR10 = 0.001  # Redwood's stated learning rate for their 8B runs (reading plan, §1)
ATTEMPT10 = "lr1e3"  # the pair's attempt in harness/attempts.py; the adversarial run's dose record names it
WRITTEN_BY = "harness/s10.py ([an unreleased plan])"


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha256(p: Path) -> str:
    from harness.released import file_sha256

    return file_sha256(p)


def _write_once(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text(encoding="utf-8") != text:
        shown = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        raise RuntimeError(f"{shown} exists with other content; a session_010 data file is not rewritten")
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# The stronger-training pair's two configurations
# ---------------------------------------------------------------------------

KINDS = {  # file name, run name, session_008 template
    "std": ("run_std_s5_lr1e3.json", "std_s5_lr1e3", "run_std_s5.json"),
    "adv": ("run_adv_s5_lr1e3.json", "adv_s5_lr1e3", "run_adv_s5.json"),
}
ADDED_FROZEN = {"stronger_runs_s10": LAUNCH_NOTE, "reading_plan_s10": READING_PLAN}
CHANGED = ("written_by", "purpose", "run_name", "lr", "results_dir", "frozen_files", "dose")  # the only fields that differ
PURPOSE = {
    "std": "The session_010 stronger-training pair, standard run: the replication's seed-5 standard configuration "
           "(data/session_008/run_std_s5.json) with the learning rate raised from 3e-4 to 1e-3, Redwood's stated rate for their "
           "8B runs; the data, seed, steps, warmup and dropout unchanged; read under results/session_010. Admitted by the launch "
           "note for the five stronger-training runs, outside the session_008 further-run rule "
           "(data/session_010/stronger_training_runs.md), for the readings of data/session_010/reading_plan.md, Section 3",
    "adv": "The session_010 stronger-training pair, adversarial run: the replication's seed-5 adversarial configuration "
           "(data/session_008/run_adv_s5.json) with the learning rate raised from 3e-4 to 1e-3, Redwood's stated rate for their "
           "8B runs; the same single-dose data, seed, steps, warmup and dropout; read under results/session_010 and paired with "
           "std_s5_lr1e3. Its dose record names its own attempt, lr1e3, with the dose figures unchanged. Admitted by the launch "
           "note for the five stronger-training runs, outside the session_008 further-run rule "
           "(data/session_010/stronger_training_runs.md), for the readings of data/session_010/reading_plan.md, Section 3",
}


def configuration(kind: str, data8: Path = DATA8) -> dict:
    """The session_008 seed-5 configuration of the kind, with only the fields of CHANGED changed, in the template's key order."""
    _, run, template = KINDS[kind]
    t = _load(Path(data8) / template)
    need = {"written_by", "purpose", "run_name", "lr", "results_dir", "frozen_files"} | ({"dose"} if kind == "adv" else set())
    if need - set(t):
        raise RuntimeError(f"{template} has no {sorted(need - set(t))}: the change would not apply")
    out = {}
    for key, v in t.items():
        if key == "written_by":
            v = WRITTEN_BY
        elif key == "purpose":
            v = PURPOSE[kind]
        elif key == "run_name":
            v = run
        elif key == "lr":
            v = LR10
        elif key == "results_dir":
            v = RESULTS10
        elif key == "frozen_files":
            v = {**v, **ADDED_FROZEN}
        elif key == "dose":
            v = {**v, "attempt": ATTEMPT10}
        out[key] = v
    return out


def configurations(data8: Path = DATA8) -> dict[str, dict]:
    return {KINDS[kind][0]: configuration(kind, data8) for kind in KINDS}


def config_text(cfg: dict) -> str:
    return json.dumps(cfg, indent=1) + "\n"  # as harness/s8.py writes the session_008 ones


def cmd_configs() -> list[Path]:
    return [_write_once(DATA10 / name, config_text(cfg)) for name, cfg in configurations().items()]


# ---------------------------------------------------------------------------
# The launch note for the five stronger-training runs
# ---------------------------------------------------------------------------


def parse_admitted(text: str) -> tuple[str, ...]:
    """The runs listed in the launch note's one fenced block whose info string is AUTH_BLOCK, one run name per line.
    ValueError unless there is exactly one such block, it lists at least one run, and none twice."""
    blocks = re.findall(rf"^```{AUTH_BLOCK}[ \t]*\n(.*?)^```[ \t]*$", text, flags=re.M | re.S)
    if len(blocks) != 1:
        raise ValueError(f"{LAUNCH_NOTE} must hold exactly one ```{AUTH_BLOCK} block; it holds {len(blocks)}")
    runs = tuple(x.strip() for x in blocks[0].splitlines() if x.strip())
    if not runs:
        raise ValueError(f"{LAUNCH_NOTE}: its ```{AUTH_BLOCK} block lists no run")
    if len(set(runs)) != len(runs):
        raise ValueError(f"{LAUNCH_NOTE}: its ```{AUTH_BLOCK} block lists a run twice")
    bad = [r for r in runs if not re.fullmatch(r"[a-z0-9_]+", r)]
    if bad:
        raise ValueError(f"{LAUNCH_NOTE}: {bad} are not run names")
    return runs


def committed(p: Path) -> bool:
    """Tracked by git and unchanged against HEAD (harness/batch.py allow_list_committed)."""
    from harness.batch import allow_list_committed

    return allow_list_committed(p)


def _root(root) -> Path:
    if root is None:
        import harness.data  # at call time, so that a test's patched root applies, as harness/train_app.py reads it

        return Path(harness.data.ROOT)
    return Path(root)


def launch_note(root=None, is_committed=None) -> dict:
    """{"runs": the runs the launch note admits, "problems": why it admits none}. It admits its listed runs only while
    stronger_training_runs.md and the session_010 manifest are both committed (is_committed, default: tracked and
    unchanged against HEAD), and stronger_training_runs.md's sha256 is the manifest's entry for it; otherwise none."""
    root = _root(root)
    is_committed = is_committed or committed
    a, m = root / LAUNCH_NOTE, root / MANIFEST10
    problems = []
    for p, rel in ((a, LAUNCH_NOTE), (m, MANIFEST10)):
        if not p.is_file():
            problems.append(f"{rel} is missing")
        elif not is_committed(p):
            problems.append(f"{rel} is not committed, or differs from HEAD")
    runs: tuple[str, ...] = ()
    if not problems:
        if (_load(m).get("files", {}).get(LAUNCH_NOTE) or {}).get("sha256") != _sha256(a):
            problems.append(f"{LAUNCH_NOTE} does not match {MANIFEST10}")
        else:
            try:
                runs = parse_admitted(a.read_text(encoding="utf-8"))
            except ValueError as e:
                problems.append(str(e))
    return {"runs": frozenset() if problems else frozenset(runs), "problems": problems}


def admitted_runs(root=None, is_committed=None) -> frozenset[str]:
    """The runs the committed, manifest-matching launch note lists; none otherwise."""
    return launch_note(root, is_committed)["runs"]


def launch_note_entry(root=None) -> dict:
    """The launch note's path and sha256, as a launch record or a verdict file names it."""
    return {"path": LAUNCH_NOTE, "sha256": _sha256(_root(root) / LAUNCH_NOTE)}


DD_ATTEMPT = "rep_dd"  # the double-dose runs' attempt in harness/attempts.py
DD_NOTE = ("Launched under the launch note for the five stronger-training runs, not further-run rule 1, which did not fire "
           "(results/session_008/seal/gate_verdict_rep.json); the purpose field of the configuration file named by run_config_file "
           "is the frozen session_008 text.")


def launch_record(run_name: str, root=None, is_committed=None) -> dict | None:
    """The launch record's session_010 entry of a run the launch note admits: the launch note and the reading plan, each
    with its sha256, and, for a double-dose run (rep_dd's), DD_NOTE, the sentence saying how it was launched. None for any
    other run. harness/train_app.py puts it in the launcher record, outside the run key, so that the double-dose runs keep
    the key their frozen session_008 configurations give."""
    from harness import attempts as AT

    if run_name not in admitted_runs(root, is_committed):
        return None
    root = _root(root)
    out = {"launch_note": launch_note_entry(root), "reading_plan": {"path": READING_PLAN, "sha256": _sha256(root / READING_PLAN)}}
    try:
        att = AT.attempt_of(run_name)
    except ValueError:  # an unregistered adversarial run: refused at launch (rerun_refusals) before any record is made
        att = None
    if att == DD_ATTEMPT:
        out["note"] = DD_NOTE
    return out


# ---------------------------------------------------------------------------
# The session_010 manifest
# ---------------------------------------------------------------------------

WHAT = {
    "reading_plan.md": "The session_010 reading plan, fixed before any session_010 run launched",
    "stronger_training_runs.md": "The launch note for the five stronger-training runs, outside the session_008 further-run rule; its one "
                        "fenced block lists them (harness/s10.py admitted_runs)",
}


def _what(p: Path) -> str:
    if p.name.startswith("run_"):
        return "A run configuration of the stronger-training pair (harness/s10.py configs)"
    return WHAT.get(p.name, "")


def manifest10(data10: Path = DATA10, existing: Path | None = None) -> dict:
    """Every file under data/session_010 (the manifest and hidden files such as .DS_Store aside) with its sha256. Refuses
    unless the launch note and the reading plan are there. Add-only: if a manifest exists, every file it lists must still
    be there with its sha256; new files are added, and an entry keeps its description."""
    data10 = Path(data10)
    files = {}
    for p in sorted(data10.rglob("*")):
        if p.is_file() and p.name != Path(MANIFEST10).name and not p.name.startswith("."):
            files[f"data/session_010/{p.relative_to(data10).as_posix()}"] = {"sha256": _sha256(p), "what": _what(p)}
    missing = [x for x in REQUIRED_FROZEN if x not in files]
    if missing:
        raise RuntimeError(f"the session_010 manifest waits for {missing}")
    out = {"what": "Every file of data/session_010 with its sha256 ([an unreleased plan]). A launch checks a configuration's "
                   "files under data/session_010 here, under data/session_008 against data/session_008/MANIFEST.json, and the "
                   "rest as a session_008 launch does (harness/train_app.py manifest_mismatches). The launch note admits runs "
                   "only while it matches its entry here (harness/s10.py admitted_runs).",
           "files": files}
    existing = Path(existing) if existing is not None else data10 / Path(MANIFEST10).name
    if existing.is_file():
        for key, rec in _load(existing).get("files", {}).items():
            new = files.get(key)
            if new is None or new["sha256"] != rec["sha256"]:
                raise RuntimeError(f"{key} is in {existing.name} with sha256 {rec['sha256'][:12]}; it is now "
                                   f"{'gone' if new is None else new['sha256'][:12]}. A frozen file does not change.")
            new["what"] = rec.get("what", new["what"])
    return out


def manifest_check(path: Path | None = None, data10: Path = DATA10) -> list[str]:
    data10 = Path(data10)
    path = Path(path) if path is not None else data10 / Path(MANIFEST10).name
    m = _load(path)
    out = []
    for key, rec in m.get("files", {}).items():
        p = data10 / Path(key).relative_to("data/session_010")
        if not p.is_file():
            out.append(f"{key}: missing")
        elif _sha256(p) != rec["sha256"]:
            out.append(f"{key}: its sha256 differs from the manifest's")
    listed = set(m.get("files", {}))
    out += [f"data/session_010/{p.relative_to(data10).as_posix()}: under data/session_010 and not in the manifest"
            for p in sorted(data10.rglob("*")) if p.is_file() and p != path and not p.name.startswith(".")
            and f"data/session_010/{p.relative_to(data10).as_posix()}" not in listed]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.s10")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs", help="the stronger-training pair's two configurations under data/session_010/")
    sub.add_parser("manifest", help="data/session_010/MANIFEST.json, add-only")
    sub.add_parser("manifest-check", help="every file the session_010 manifest lists has its sha256, and none under data/session_010 is unlisted")
    sub.add_parser("admitted", help="the runs the committed launch note admits, or why it admits none")
    a = ap.parse_args()
    if a.cmd == "configs":
        print("\n".join(str(p.relative_to(ROOT)) for p in cmd_configs()))
    elif a.cmd == "manifest":
        m = manifest10()
        (ROOT / MANIFEST10).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST10}: {len(m['files'])} files")
    elif a.cmd == "manifest-check":
        problems = manifest_check()
        print("\n".join(problems) if problems else f"every file in {MANIFEST10} matches its sha256")
        sys.exit(1 if problems else 0)
    elif a.cmd == "admitted":
        r = launch_note()
        print(json.dumps({"runs": sorted(r["runs"]), "problems": r["problems"]}, indent=1))
        sys.exit(0 if r["runs"] else 1)


if __name__ == "__main__":
    main()
