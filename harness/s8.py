"""The session_008 data files: the fresh-seed replication's run configurations, their launch checks and the round's manifest
([an unreleased plan]).

    uv run python -m harness.s8 configs          # data/session_008/run_std_s{5,6,7}.json, run_adv_s*.json, run_adv_dd_s*.json
    uv run python -m harness.s8 launch-checks    # data/session_008/launch_checks.json (CPU, a few minutes)
    uv run python -m harness.s8 manifest         # data/session_008/MANIFEST.json, once the pre-registration and the
                                                 #   operating rules are committed
    uv run python -m harness.s8 manifest-check   # every file the session_008 manifest lists has its sha256

The nine configurations are the session_006 ones (data/session_006/run_std_s2.json, run_adv_a1_s2.json, run_adv_a2_s2.json)
with only the run name, the seed, the results folder (results/session_008), the launch record's added files (the session_008
pre-registration and operating rules, the reading rule and the split-and-interval text), the adversarial ones' launch-check
file (the session_008 one) and the dose's attempt (rep, rep_dd) changed. The _dd runs are the double-dose rerun, launched only if
the replication's gate fails (harness/train_app.py rerun_refusals).

The launch-check file records, for mix_adv_a1.json and mix_adv_a2.json at seeds 5, 6 and 7, the trainer-path leak check and
the pairing check, as data/session_006/launch_checks.json does for seeds 2, 3 and 4; the standard mix's leak check at seeds
5, 6 and 7; and the datasets' content checks, recomputed from the frozen build by the session_006 function and required to equal
the session_006 recorded ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA6, DATA8 = ROOT / "data" / "session_006", ROOT / "data" / "session_008"
MANIFEST8 = DATA8 / "MANIFEST.json"
SEEDS8 = (5, 6, 7)
STD_MIX, A1_MIX, A2_MIX = "data/session_006/mix_standard.json", "data/session_006/mix_adv_a1.json", "data/session_006/mix_adv_a2.json"
LAUNCH_CHECKS8 = "data/session_008/launch_checks.json"
PREREG8, RULES8 = "data/session_008/preregistration.md", "data/session_008/operating_rules.md"
RULES8_ERRATUM_1 = "data/session_008/operating_rules_erratum_1.md"
READING_RULE, SPLIT_TEXT = "data/session_006/s7_reading_rule.md", "data/session_006/s7_split_and_interval.md"
ADDED_FROZEN = {"preregistration_s8": PREREG8, "operating_rules_s8": RULES8, "operating_rules_s8_erratum_1": RULES8_ERRATUM_1,
                "reading_rule": READING_RULE, "split_and_interval": SPLIT_TEXT}
WRITTEN_BY = "harness/s8.py ([an unreleased plan])"
# The frozen files outside data/session_008 that no other manifest lists: the session_007 reading texts are under
# data/session_006 but not in its manifest, so the session_008 manifest pins them, with the commit that added them.
OUTSIDE = (READING_RULE, SPLIT_TEXT)


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha256(p: Path) -> str:
    from harness.released import file_sha256

    return file_sha256(p)


def _write_once(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text(encoding="utf-8") != text:
        shown = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        raise RuntimeError(f"{shown} exists with other content; a session_008 data file is not rewritten")
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# The nine configurations
# ---------------------------------------------------------------------------

KINDS = {  # file stem, run name, session_006 template, attempt of the dose (None: a standard run)
    "std": ("run_std_s{k}", "std_s{k}", "run_std_s2.json", None),
    "adv": ("run_adv_s{k}", "adv_s{k}", "run_adv_a1_s2.json", "rep"),
    "dd": ("run_adv_dd_s{k}", "adv_s{k}_dd", "run_adv_a2_s2.json", "rep_dd"),
}
PURPOSE = {
    "std": "The fresh-seed replication's standard run at training seed {k}: the session_006 std_s2 configuration (setting B4 "
           "on the 28,000-pair file) at seed {k}, read under results/session_008",
    "adv": "The fresh-seed replication's adversarial run at training seed {k}: the session_006 A1 configuration (adv_s2) "
           "at seed {k}, on the same A1 data, read under results/session_008",
    "dd": "The fresh-seed replication's double-dose rerun at training seed {k}: the session_006 A2 configuration (adv_s2_a2) "
          "at seed {k}, on the same A2 data; launched only if the replication's gate fails (further-run rule 1)",
}


def configuration(kind: str, k: int, data6: Path = DATA6) -> dict:
    """The session_006 configuration of the kind, with only the changes the module docstring lists."""
    _, run, template, attempt = KINDS[kind]
    t = _load(Path(data6) / template)
    out = {}
    for key, v in t.items():
        if key == "written_by":
            v = WRITTEN_BY
        elif key == "purpose":
            v = PURPOSE[kind].format(k=k)
        elif key == "run_name":
            v = run.format(k=k)
        elif key == "seed":
            v = k
        elif key == "results_dir":
            v = "results/session_008"
        elif key == "frozen_files":
            v = {**v, **ADDED_FROZEN}
            if "launch_checks" in v:
                v["launch_checks"] = LAUNCH_CHECKS8
        elif key == "dose":
            v = {**v, "attempt": attempt}
        out[key] = v
    return out


def configurations(data6: Path = DATA6) -> dict[str, dict]:
    return {KINDS[kind][0].format(k=k) + ".json": configuration(kind, k, data6) for kind in KINDS for k in SEEDS8}


def config_text(cfg: dict) -> str:
    return json.dumps(cfg, indent=1) + "\n"


def cmd_configs(rewrite: bool = False, results8: Path | None = None) -> list[Path]:
    """Write the nine configurations, each once. rewrite (added when the erratum joined frozen_files) replaces
    configurations already written, and only while none of their runs has a folder under results/session_008, that is,
    before any of them was launched; the manifest entries must then be replaced by name (manifest --replace)."""
    cfgs = configurations()
    if not rewrite:
        return [_write_once(DATA8 / name, config_text(cfg)) for name, cfg in cfgs.items()]
    results8 = Path(results8 or ROOT / "results" / "session_008")
    launched = sorted(cfg["run_name"] for cfg in cfgs.values() if (results8 / cfg["run_name"]).exists())
    if launched:
        raise RuntimeError(f"runs {launched} have folders under {results8}: their configurations are not rewritten")
    out = []
    for name, cfg in cfgs.items():
        (DATA8 / name).write_text(config_text(cfg), encoding="utf-8")
        out.append(DATA8 / name)
    return out


# ---------------------------------------------------------------------------
# The launch checks
# ---------------------------------------------------------------------------


def dataset_checks() -> dict:
    """The datasets' content checks of amendment 1 §6 (harness/s6_build.py launch_checks), recomputed from the frozen build as
    the session_006 cmd_launch_checks did, and required to equal what data/session_006/launch_checks.json recorded."""
    from harness import s6_build as B

    nho = B._load(B.frozen(B.BUILD_DIR / "near_held_out.json"))
    b = B.frozen_build(nho["names"])
    att = B._load(B.frozen(B.BUILD_DIR / "attempts.json"))
    ds = B._load(B.frozen(B.BUILD_DIR / "datasets.json"))
    gate = B._load(B.frozen(DATA6 / "gate_sample.json"))
    inputs = B._load(B.frozen(B.BUILD_DIR / "readout_inputs.json"))
    import gzip

    lines = {a: [x for x in gzip.decompress(B.frozen(DATA6 / f"caught_{a}.jsonl.gz").read_bytes()).split(b"\n") if x] for a in ("a1", "a2")}
    checks = B.launch_checks(lines["a1"], lines["a2"], ds["recorded"], dose_dirs=b["dose_dirs"], cfg_sha=B.config_sha256(B.answer_config()),
                             render_len=B.trainer_render_len(), near_held_out_clusters=nho["clusters"], human=b["human"],
                             human_clusters=set(att["human_clusters"]), r_ids=b["r_ids"], kept_probes={p["id"] for p in inputs["probes"]},
                             gate_ids=gate["ids"], caps=ds["cap"])
    s6 = _load(B.frozen(DATA6 / "launch_checks.json"))["datasets"]
    if json.loads(json.dumps(checks, default=str)) != s6:
        raise RuntimeError("the datasets' content checks recomputed now differ from data/session_006/launch_checks.json's")
    return checks


def launch_checks8(datasets: dict, seeds=SEEDS8, check_leak=None, paired_check=None) -> dict:
    """The session_006 launch record (harness/s6_build.py launch_record) for the two adversarial mixes at seeds 5, 6 and 7, and
    the standard mix's leak check at the same seeds."""
    from harness.s6_build import launch_record

    if check_leak is None:
        from harness.train_app import check_leak
    rec = launch_record(datasets, [A1_MIX, A2_MIX], seeds=seeds, check_leak=check_leak, paired_check=paired_check)
    std = {}
    for k in seeds:
        leak = check_leak(STD_MIX, k)
        std[f"{STD_MIX}|seed{k}"] = {"leak": {x: leak[x] for x in ("by_dataset", "forbidden_records_drawn", "dropped_over_length") if x in leak},
                                     "leak_passed": bool(leak["passed"])}
    rec["what"] = ("The session_008 launch checks for the fresh-seed replication ([an unreleased plan]): the trainer-path leak "
                   "check and the pairing check for the two adversarial mixes at seeds 5, 6 and 7, as "
                   "data/session_006/launch_checks.json records them for seeds 2, 3 and 4; the datasets' content checks, "
                   "recomputed and equal to the session_006 ones; and the standard mix's leak check at seeds 5, 6 and 7")
    rec["written_by"] = WRITTEN_BY
    rec["standard_runs"] = std
    rec["standard_passed"] = all(v["leak_passed"] for v in std.values())
    rec["passed"] = bool(rec["passed"] and rec["standard_passed"])
    return rec


def cmd_launch_checks() -> dict:
    rec = launch_checks8(dataset_checks())
    _write_once(ROOT / LAUNCH_CHECKS8, json.dumps(rec, indent=1, default=str) + "\n")
    return {"passed": rec["passed"], "datasets_passed": rec["datasets_passed"], "standard_passed": rec["standard_passed"],
            "runs": {k: (v["leak_passed"], v["pairing_passed"]) for k, v in rec["runs"].items()}}


# ---------------------------------------------------------------------------
# The session_008 manifest
# ---------------------------------------------------------------------------

WHAT = {
    "preregistration.md": "The session_008 pre-registration",
    "operating_rules.md": "The session_008 operating rules",
    "operating_rules_erratum_1.md": "Erratum 1 to the session_008 operating rules, written before any GPU work",
    "launch_checks.json": "The leak, pairing and content checks of the replication's runs at seeds 5, 6 and 7 (harness/s8.py launch-checks)",
    "s8_strings_q29.json": "The 29-question batch file: the trigger and the twelve strings on every reserved question, scoring only "
                           "(harness/s8_q29.py write-files)",
    "confirmatory_files.json": "The session_008 allow-list: the readout files admitted to render reserved questions, beside the "
                               "session_006 one (harness/s8_q29.py write-files)",
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def outside_pins(root: Path = ROOT) -> dict:
    """The frozen files outside data/session_008 that the session_008 manifest pins: their sha256 now, which must be their
    content at the commit that added them (unchanged since)."""
    out = {}
    for p in OUTSIDE:
        added = _git("log", "--diff-filter=A", "--format=%H", "--", p).split()
        if len(added) != 1:
            raise RuntimeError(f"{p}: added in {len(added)} commits, not one")
        then = hashlib.sha256(subprocess.run(["git", "show", f"{added[0]}:{p}"], cwd=ROOT, capture_output=True, check=True).stdout).hexdigest()
        now = _sha256(root / p)
        if then != now:
            raise RuntimeError(f"{p} has changed since the commit that added it ({added[0][:7]})")
        out[p] = {"sha256": now, "added_in": added[0], "what": "A session_007 reading text, frozen since it was committed; "
                                                                 "not listed in data/session_006/MANIFEST.json"}
    return out


def _what(p: Path) -> str:
    if p.parent.name == "predictions":
        return f"{p.stem}'s predictions, recorded before any session_008 GPU work"
    if p.name.startswith("run_"):
        return "A run configuration of the replication (harness/s8.py configs)"
    return WHAT.get(p.name, "")


def config_runs() -> dict[str, str]:
    """The nine configurations, keyed by their path as the manifest lists it, each with the run it configures."""
    return {f"data/session_008/{stem.format(k=k)}.json": run.format(k=k) for stem, run, _, _ in KINDS.values() for k in SEEDS8}


def manifest8(data8: Path = DATA8, existing: Path | None = None, replace=(), results8: Path | None = None) -> dict:
    """Every file under data/session_008 (the manifest aside) with its sha256, and the pins of OUTSIDE. Refuses unless the
    pre-registration and the operating rules are there. Add-only: if a manifest exists, every file it lists must still be
    there with its sha256 (a frozen file does not change, and is not dropped); new files are added. A file named in
    replace (by its path) may have changed: its entry takes the new sha256 and lists the old ones under "replaced".

    replace takes the nine run configurations only, so no committed text (the pre-registration,
    the operating rules, the erratum, the prediction files) can be replaced through it; and, as configs --rewrite, only while
    the configuration's run has no folder under results/session_008 (results8)."""
    replace = set(replace)
    runs = config_runs()
    not_configs = sorted(replace - set(runs))
    if not_configs:
        raise RuntimeError(f"--replace takes the run configurations only (data/session_008/run_*.json); {not_configs} cannot be replaced")
    results8 = Path(results8) if results8 is not None else ROOT / "results" / "session_008"
    launched = sorted(runs[k] for k in replace if (results8 / runs[k]).exists())
    if launched:
        raise RuntimeError(f"runs {launched} have folders under {results8}: their configurations are not replaced")
    missing = [p for p in (PREREG8, RULES8) if not (ROOT / p).is_file()]
    if missing:
        raise RuntimeError(f"the session_008 manifest waits for {missing}")
    files = {}
    for p in sorted(Path(data8).rglob("*")):
        if p.is_file() and p.name != MANIFEST8.name:
            key = p.resolve().relative_to(ROOT).as_posix()
            files[key] = {"sha256": _sha256(p), "what": _what(p)}
    out = {"what": "Every frozen file of the session_008 round with its sha256 ([an unreleased plan]). A session_008 launch "
                   "checks the files under data/session_008 here, the session_005 and session_006 ones against "
                   "data/session_006/MANIFEST.json, and the two below against their pins (harness/train_app.py "
                   "manifest_mismatches).",
           "files": files, "outside_session_008": outside_pins()}
    existing = Path(existing) if existing is not None else (Path(data8) / MANIFEST8.name)
    if existing.is_file():
        old = _load(existing)
        unknown = replace - set(old.get("files", {}))
        if unknown:
            raise RuntimeError(f"--replace names files the manifest does not list: {sorted(unknown)}")
        for section in ("files", "outside_session_008"):
            for key, rec in old.get(section, {}).items():
                new = out[section].get(key)
                if new is not None and key in replace and section == "files":
                    new["replaced"] = rec.get("replaced", []) + ([rec["sha256"]] if rec["sha256"] != new["sha256"] else [])
                elif new is None or new["sha256"] != rec["sha256"]:
                    raise RuntimeError(f"{key} is in {existing.name} with sha256 {rec['sha256'][:12]}; it is now "
                                       f"{'gone' if new is None else new['sha256'][:12]}. A frozen file does not change.")
                elif rec.get("replaced"):
                    new["replaced"] = rec["replaced"]
                new["what"] = rec.get("what", new.get("what", ""))  # an entry, once written, keeps its description
    elif replace:
        raise RuntimeError("--replace needs an existing manifest")
    return out


def manifest_check(path: Path = MANIFEST8) -> list[str]:
    m = _load(path)
    out = []
    for section in ("files", "outside_session_008"):
        for key, rec in m.get(section, {}).items():
            p = ROOT / key
            if not p.is_file():
                out.append(f"{key}: missing")
            elif _sha256(p) != rec["sha256"]:
                out.append(f"{key}: its sha256 differs from the manifest's")
    listed = set(m.get("files", {}))
    out += [f"{p.relative_to(ROOT).as_posix()}: under data/session_008 and not in the manifest" for p in sorted(DATA8.rglob("*"))
            if p.is_file() and p != path and p.relative_to(ROOT).as_posix() not in listed]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.s8")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cf = sub.add_parser("configs", help="the nine run configurations under data/session_008/")
    cf.add_argument("--rewrite", action="store_true", help="replace configurations already written; only before any of their runs launched")
    sub.add_parser("launch-checks", help="data/session_008/launch_checks.json (CPU)")
    mf = sub.add_parser("manifest", help="data/session_008/MANIFEST.json, once the pre-registration and operating rules are committed")
    mf.add_argument("--replace", nargs="*", default=[], help="rewritten run configurations whose new content replaces their entry; only "
                                                             "configurations, and only before their runs have a folder under results/session_008")
    sub.add_parser("manifest-check", help="every file the session_008 manifest lists has its sha256, and none under data/session_008 is unlisted")
    a = ap.parse_args()
    if a.cmd == "configs":
        print("\n".join(str(p.relative_to(ROOT)) for p in cmd_configs(rewrite=a.rewrite)))
    elif a.cmd == "launch-checks":
        r = cmd_launch_checks()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["passed"] else 1)
    elif a.cmd == "manifest":
        m = manifest8(replace=a.replace)
        MANIFEST8.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST8.relative_to(ROOT)}: {len(m['files'])} files, {len(m['outside_session_008'])} pinned outside")
    elif a.cmd == "manifest-check":
        problems = manifest_check()
        print("\n".join(problems) if problems else f"every file in {MANIFEST8.relative_to(ROOT)} matches its sha256")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
