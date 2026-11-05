"""The session_005 launcher: every GPU job of that round goes through here.

    uv run python -m harness.s5 check                                  # the deployed app runs the local committed code
    uv run python -m harness.s5 batch --file data/session_005/<name>.json --model normal [--score-only] [--out <folder>]

Each job first asks the deployed app (a CPU function) which code it runs and refuses unless that is the local committed
code (harness/provenance.py); the GPU containers check again when they start. Outputs go under results/session_005/,
stage caches under .cache/session_005/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS5 = ROOT / "results" / "session_005"
CACHE5 = ROOT / ".cache" / "session_005"


def deployed_check() -> dict:
    import modal

    from harness.modal_app import APP_NAME
    from harness.provenance import check_deployed

    deployed = modal.Function.from_name(APP_NAME, "build_info_remote").remote()
    st = check_deployed(deployed)
    print(f"deployed app runs commit {deployed['commit'][:12]}, code {deployed['code_hash'][:12]}: the local committed code")
    return st


def batch(file: str, model: str, score_only: bool, out: str = "") -> None:
    """A batch file on one model into results/session_005/<out or the file's name>/<model>/."""
    from harness.batch import load_batch, run_batch
    from harness.batch_app import deployed_runner

    deployed_check()
    name = out or load_batch(file)["data"]["name"]
    run_batch(file, deployed_runner().with_options(scaledown_window=60), models=model, score_only=score_only,
              out_dir=RESULTS5 / name / model, cache_dir=CACHE5 / name / model)


# ---------------------------------------------------------------------------
# Reading a run's model checkpoints
# ---------------------------------------------------------------------------

SCOUT_FILES = {"normal": "data/session_005/scout_normal.json", "distilled": "data/session_005/scout_distilled.json",
               "clean": "data/session_005/scout_normal.json"}
S4_FILES = [f"data/session_004/readout_{n}.json" for n in ("trigger", "end_placement", "caught_r1", "caught_r2", "tied_unpadded", "panel",
                                                           "handwritten", "gated", "sampled", "search_rest")]
SAVED = [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 700]
CHUNK_PAIRS, CHUNK_MMLU = 500, 500


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _stage(cache: Path, name: str, fn):
    p = cache / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = fn()
    cache.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".tmp").write_text(json.dumps(out))
    p.with_suffix(".tmp").replace(p)
    return out


def damage_step(R, model: str, adapter: str, out: Path, cache: Path, heldout: list[str], mmlu: bool, ifeval: bool) -> dict:
    """The damage readouts at one model checkpoint, in one container (the same Runner parameters as the batch runner's, so a
    warm container is shared). Writes per-pair and per-question files under out/; caches every GPU call under cache/."""
    import time

    from harness import damage as D
    from harness.batch import spend_estimate
    from harness.data import STOP_IDS, load_all
    from harness.pipeline import local_tokenizer
    from harness.provenance import check_deployed, code_line, git_state, output_stamp
    from harness.render import Renderer
    from harness.train import MICRO_TOKENS

    t0 = time.time()
    d = load_all()
    s = d["settings"]
    system = s["system_prompts"][s["chosen_format"]]
    tok = local_tokenizer(model, s)
    r = Renderer(tok, d["questions"], s)
    runner = R(model_key=model, dtype="bfloat16", **({"adapter": adapter} if adapter else {}))
    local = git_state()
    fresh = not (cache / "info.json").exists()
    info = _stage(cache, "info", lambda: runner.info.remote())
    if fresh:
        check_deployed(info["build"], local)
    st = output_stamp(info["build"], local)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "adapter": adapter or None, "container": {k: info[k] for k in ("gpu", "adapter", "build", "versions")},
            **st, "readouts": {}}
    old = out / "metadata.json"
    if old.exists():  # a later reading at the same step (another held-out set) adds to the record, it does not replace it
        prev = json.loads(old.read_text())
        meta["readouts"] = prev.get("readouts", {})
        meta["earlier_invocations"] = prev.get("earlier_invocations", []) + [{k: prev.get(k) for k in ("harness_commit", "harness_code_hash",
                                                                                                       "spend_estimate", "wall_seconds")}]
    for which in heldout:
        pairs = D.heldout_pairs(which)
        hs, files = D.heldout_set_sha256(which)  # the cache is keyed by the set's content
        rows = []
        for c in range(0, len(pairs), CHUNK_PAIRS):
            rows += _stage(cache, f"heldout_{which}_{hs[:16]}_{c // CHUNK_PAIRS:03d}",
                           lambda c=c: runner.heldout_loss.remote(pairs[c : c + CHUNK_PAIRS], system, MICRO_TOKENS))
        rows = [{"pair": x["pid"], "loss_sum": x["loss_sum"], "n_tokens": x["n_tokens"], **st} for x in rows]
        _write_csv(out / f"heldout_loss_{which}.csv", rows)
        L = sum(x["loss_sum"] for x in rows) / sum(x["n_tokens"] for x in rows)
        meta["readouts"][f"heldout_loss_{which}"] = {"pairs": len(rows), "tokens": sum(x["n_tokens"] for x in rows), "L": L,
                                                     "set_sha256": hs, "set_files": files}
    if mmlu:
        sample = json.loads((ROOT / "data" / "session_005" / "mmlu_sample.json").read_text())
        items = sample["items"]
        letter_ids = tok.convert_tokens_to_ids(D.LETTERS)
        jobs = [{"pid": x["idx"], "ids": r.render([{"role": "system", "content": system}, {"role": "user", "content": D.mmlu_user_text(x)}], None)["ids"]}
                for x in items]
        recs = []
        for c in range(0, len(jobs), CHUNK_MMLU):
            recs += _stage(cache, f"mmlu_{c // CHUNK_MMLU:03d}", lambda c=c: runner.letter_logits.remote(jobs[c : c + CHUNK_MMLU], letter_ids))
        rows = [{**D.mmlu_score(rec, x), **st} for rec, x in zip(recs, items)]
        assert all(rec["pid"] == x["idx"] for rec, x in zip(recs, items))
        _write_csv(out / "mmlu.csv", rows)
        meta["readouts"]["mmlu"] = {"questions": len(rows), "accuracy": sum(x["correct"] for x in rows) / len(rows),
                                    "mean_p_correct": sum(x["p_correct"] for x in rows) / len(rows),
                                    "mean_letter_mass": sum(x["letter_mass"] for x in rows) / len(rows),
                                    "sample_sha256": D.sha256_file(ROOT / "data" / "session_005" / "mmlu_sample.json")}
    if ifeval:
        sample = json.loads((ROOT / "data" / "session_005" / "ifeval_sample.json").read_text())
        items = sample["items"]
        jobs = [{"pid": x["key"], "ids": r.render([{"role": "system", "content": system}, {"role": "user", "content": x["prompt"]}], None)["ids"]}
                for x in items]
        gens = {g["pid"]: g for g in _stage(cache, "ifeval", lambda: runner.greedy_batch.remote(jobs, D.IFEVAL_MAX_NEW_TOKENS, list(STOP_IDS), 50))}
        resp = {}
        for x in items:
            ids = gens[x["key"]]["gen_ids"]
            stopped = bool(ids) and ids[-1] in STOP_IDS
            resp[x["key"]] = (tok.decode(ids[:-1] if stopped else ids, skip_special_tokens=True), len(ids), stopped)
        checks = D.ifeval_check(items, {k: v[0] for k, v in resp.items()})
        rows = [{**c, "response": resp[c["key"]][0], "n_tokens": resp[c["key"]][1], "stopped": resp[c["key"]][2], **st} for c in checks]
        with open(out / "ifeval.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        n_inst = sum(len(x["strict_list"]) for x in rows)
        meta["readouts"]["ifeval"] = {"prompts": len(rows), "prompt_strict": sum(x["strict_all"] for x in rows) / len(rows),
                                      "prompt_loose": sum(x["loose_all"] for x in rows) / len(rows),
                                      "instruction_strict": sum(sum(x["strict_list"]) for x in rows) / n_inst,
                                      "instruction_loose": sum(sum(x["loose_list"]) for x in rows) / n_inst,
                                      "not_stopped": sum(not x["stopped"] for x in rows),
                                      "sample_sha256": D.sha256_file(ROOT / "data" / "session_005" / "ifeval_sample.json")}
    if fresh:
        meta["spend_estimate"] = spend_estimate(runner.uptime.remote())
    meta["wall_seconds"] = round(time.time() - t0, 1)
    meta["code"] = code_line(st)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    from harness.pipeline import log  # through the log, so that an adversarial readout's damage values go to its log file

    log(f"[damage] {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: "
        + ", ".join(f"{k} {v.get('L', v.get('accuracy', v.get('prompt_strict')))}" for k, v in meta["readouts"].items()))
    return meta


def attach_set_hashes(out: str, model: str) -> list[str]:
    """For held-out readings written before the cache was keyed by content: check each reading's
    pair ids and every pair's answer-token count against the set's current files, and only then record the set's sha256
    in the step's metadata.json, marked as attached after the reading. A reading that does not match raises."""
    import csv

    from harness import damage as D
    from harness.data import load_all
    from harness.pipeline import local_tokenizer
    from harness.provenance import local_stamp

    s = load_all()["settings"]
    system = s["system_prompts"][s["chosen_format"]]
    tok = local_tokenizer(model, s)
    st = local_stamp()
    expected, done = {}, []
    for meta_path in sorted((RESULTS5 / out / "damage").glob("step_*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        changed = False
        for key, r in meta["readouts"].items():
            if not key.startswith("heldout_loss_") or "set_sha256" in r:
                continue
            which = key[len("heldout_loss_"):]
            if which not in expected:
                expected[which] = D.expected_tokens(which, tok, system)
            with open(meta_path.parent / f"{key}.csv") as f:
                got = {row["pair"]: int(row["n_tokens"]) for row in csv.DictReader(f)}
            if got != expected[which]:
                bad = sorted(set(got) ^ set(expected[which]))[:5] or [k for k in got if got[k] != expected[which].get(k)][:5]
                raise RuntimeError(f"{meta_path.parent / key}.csv does not match the current {which} set, e.g. {bad}")
            hs, files = D.heldout_set_sha256(which)
            r.update({"set_sha256": hs, "set_files": files,
                      "set_sha256_attached": {"how": "after the reading: its pair ids and every pair's answer-token count were checked "
                                                     "against the set's current files and all matched", **st}})
            changed = True
            done.append(f"{meta_path.parent.relative_to(ROOT)}/{key}")
        if changed:
            meta_path.write_text(json.dumps(meta, indent=1))
    print("\n".join(done) or "nothing to attach")
    return done


def read_run(out: str, model: str, adapter_run: str, steps: list[int], scout_steps: list[int], s4_steps: list[int], heldout: list[str],
             mmlu: bool, ifeval: bool, step0_adapter: bool, step0_scout_from: str = "", step0_s4_from: str = "") -> None:
    """Read one run's model checkpoints, step by step: the scout file at scout_steps, the session_004 readout files at
    s4_steps, and the damage readouts at every step in steps. Step 0 is the untrained model unless step0_adapter (the
    run's own step_0). Trajectories cover step 0 from step0_*_from (a results folder of session_005 or session_004) when
    step 0 is read elsewhere."""
    from harness.batch import load_batch, run_batch
    from harness.batch_app import deployed_runner
    from harness.readout import trajectory

    deployed_check()
    R = deployed_runner().with_options(scaledown_window=60)
    run_out, run_cache = RESULTS5 / out, CACHE5 / out
    for st in steps:
        adapter = "" if (st == 0 and not step0_adapter) else f"{adapter_run}/step_{st}"
        if st in scout_steps:
            f = SCOUT_FILES[model]
            n = load_batch(f)["data"]["name"]
            run_batch(f, R, models=model, adapter=adapter, out_dir=run_out / f"step_{st}" / n, cache_dir=run_cache / f"step_{st}" / n)
        if st in s4_steps:
            for f in S4_FILES:
                n = load_batch(f)["data"]["name"]
                run_batch(f, R, models=model, adapter=adapter, out_dir=run_out / f"step_{st}" / n, cache_dir=run_cache / f"step_{st}" / n)
        if heldout or mmlu or ifeval:
            damage_step(R, model, adapter, run_out / "damage" / f"step_{st}", run_cache / "damage" / f"step_{st}", heldout, mmlu, ifeval)
    for files, sts, src in ((SCOUT_FILES[model], scout_steps, step0_scout_from), (S4_FILES, s4_steps, step0_s4_from)):
        files = [files] if isinstance(files, str) else files
        if not sts:
            continue
        tsteps = sorted(set(sts) | ({0} if src else set()))
        for f in files:
            n = load_batch(f)["data"]["name"]
            rows = trajectory(out, tsteps, n, RESULTS5, step_out={0: ROOT / src} if src else None)
            _write_csv(run_out / f"trajectory_{n}.csv", rows)
            print(f"wrote {(run_out / f'trajectory_{n}.csv').relative_to(ROOT)}")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    b = sub.add_parser("batch")
    b.add_argument("--file", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--score-only", action="store_true")
    b.add_argument("--out", default="")
    rr = sub.add_parser("read", help="read a run's model checkpoints")
    rr.add_argument("--out", required=True, help="the folder under results/session_005")
    rr.add_argument("--model", required=True)
    rr.add_argument("--adapter-run", default="", help="the run's name on the runs volume")
    rr.add_argument("--steps", default="", help="model checkpoints to read, in order")
    rr.add_argument("--scout-steps", default="")
    rr.add_argument("--s4-steps", default="")
    rr.add_argument("--heldout", default="", help="held-out sets: s5, A, old_answers_s5")
    rr.add_argument("--mmlu", action="store_true")
    rr.add_argument("--ifeval", action="store_true")
    rr.add_argument("--step0-adapter", action="store_true", help="step 0 is the run's own step_0 adapter, not the untrained model")
    rr.add_argument("--step0-scout-from", default="", help="results folder holding step 0's scout readout (shared), for the trajectory")
    rr.add_argument("--step0-s4-from", default="",
                    help="results folder holding step 0's session_004 readouts (shared), for the trajectory")
    sh = sub.add_parser("stamp-heldout", help="attach the held-out set's sha256 to readings made before the cache was keyed by it")
    sh.add_argument("--out", required=True)
    sh.add_argument("--model", required=True)
    a = ap.parse_args()
    if a.cmd == "check":
        print(json.dumps(deployed_check(), indent=1))
    elif a.cmd == "batch":
        batch(a.file, a.model, a.score_only, a.out)
    elif a.cmd == "stamp-heldout":
        attach_set_hashes(a.out, a.model)
    elif a.cmd == "read":
        read_run(a.out, a.model, a.adapter_run, _ints(a.steps), _ints(a.scout_steps), _ints(a.s4_steps),
                 [x for x in a.heldout.split(",") if x], a.mmlu, a.ifeval, a.step0_adapter, a.step0_scout_from, a.step0_s4_from)


if __name__ == "__main__":
    main()
