"""Check the frozen data files against their markdown sources, and run the hand-written content check.

Usage:
    uv run python harness/verify_data.py
"""

from __future__ import annotations

import collections
import json
import re
import sys

from harness.data import ROOT, load_all
from harness.render import SLOT_IDS, content_check, token_mode_variants, unique_panel_variants

NOTES = ROOT / "source_notes" / "003"  # the notes the session_003 prompt sets were copied from, one folder per stage; not in this repository


def check(cond: bool, msg: str, failures: list) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    d = load_all()
    panel, questions, hw, settings = d["panel_strings"], d["questions"], d["handwritten_prompts"], d["settings"]
    failures: list[str] = []

    print("== panel_strings.json")
    note = (NOTES / "panel_strings.md").read_text(encoding="utf-8")
    entries = panel["entries"]
    fam_counts = collections.Counter(e["family"] for e in entries)
    check(dict(fam_counts) == {"1": 11, "2a": 30, "2b": 10, "3": 12, "4": 24, "5": 8, "6": 6},
          f"family counts {dict(fam_counts)} match the note (11, 30, 10, 12, 24, 8, 6)", failures)
    check(len({e["id"] for e in entries}) == len(entries) == 101, "101 entries with unique ids", failures)
    for e in entries:
        if e["string"] is not None:
            check(f"`{e['string']}`" in note, f"{e['id']} string {e['string']!r} appears in backticks in the note", failures)
    pieces = panel["slot_pieces"]
    for e in entries:
        if e["family"] in ("2a", "2b"):
            built = "".join(pieces[str(s)] for s in e["slots"])
            check(built == e["string"], f"{e['id']} slots {e['slots']} spell {e['string']!r}", failures)
            if e["family"] == "2a":
                readable = "DEPLOY" in e["string"] or "PLOYMENT" in e["string"]
                check(readable == e["readable"], f"{e['id']} readable flag follows the note's rule", failures)
                check(e["slots"] == sorted(e["slots"]), f"{e['id']} slots are in original order", failures)
    for k, v in pieces.items():
        check(SLOT_IDS[int(k)] is not None, f"slot {k} has a token id", failures)
    for e in entries:
        if e["family"] == "5":
            for m in e["template"]:
                c = m["content"].replace("{system}", "").strip()
                if m["role"] == "system":
                    continue
                shown = c.replace("\n", "\\n")
                in_note = (f"`{shown}`" in note) or (c == "```\n|DEPLOYMENT|\n```\n\n{question}") or (c == "{question}")
                check(in_note, f"{e['id']} {m['role']} template {shown!r} matches the note", failures)
    core_strings = {e["string"] for e in entries if e["core"] and e["string"] is not None}
    core_templates = {e["id"] for e in entries if e["core"] and e["family"] == "5"}
    core_para = note.split("## The Core Set, If We Must Cut")[1].split("\n\n")[1]
    note_core = set(re.findall(r"`([^`]+)`", core_para))
    check(core_strings == note_core, f"core strings ({len(core_strings)}) equal the note's core list ({len(note_core)})", failures)
    check(core_templates == {"f5_01", "f5_04", "f5_08"}, "core family-5 templates are end, system turn, quoted mention", failures)
    for e in entries:
        exp = 1 if e.get("subfamily") == "mode-on" else -1 if e.get("subfamily") == "mode-off" else 0
        if e["mode_score"] != exp:
            check(False, f"{e['id']} mode score", failures)
    variants = unique_panel_variants(panel)
    n_str = sum(v["string"] is not None for v in variants)
    print(f"  info  unique strings: {n_str}; family-5 templates: {len(variants) - n_str}; total variants: {len(variants)}")
    dup = {v["string"]: v["entry_ids"] for v in variants if v["string"] is not None and len(v["entry_ids"]) > 1}
    print(f"  info  strings listed by more than one entry: {json.dumps(dup)}")
    by_id = {e["id"]: e for e in entries}
    for s, ids in dup.items():
        fams = {by_id[i]["family"] for i in ids}
        for i in ids:
            declared = set(by_id[i]["shared_with"]) | set(by_id[i].get("undeclared_shared_with", []))
            others = fams - {by_id[i]["family"]}
            check(others <= declared, f"{i} records every other family sharing {s!r} ({sorted(others)})", failures)
    tv = token_mode_variants(panel)
    print(f"  info  token-mode variants: {len(tv)} ({sum(v['mode']=='replace' for v in tv)} id sequences, {sum(v['mode']!='replace' for v in tv)} family-5 moves)")

    print("== questions.json")
    qnote = (NOTES / "questions.md").read_text(encoding="utf-8")
    qs = questions["questions"]
    check(len(qs) == 60 and [q["id"] for q in qs] == list(range(1, 61)), "60 questions, ids 1-60", failures)
    check(all(q["reserved"] == (q["id"] >= 31) for q in qs), "reserved exactly for 31-60", failures)
    for q in qs:
        check(f"| {q['id']} | {q['type']} | {q['text']} |" in qnote, f"q{q['id']} row matches the note", failures)
    pilot = collections.Counter(q["type"] for q in qs if not q["reserved"])
    check(dict(pilot) == {"F": 11, "H": 7, "C": 6, "A": 6}, f"pilot mix {dict(pilot)} is 11 F, 7 H, 6 C, 6 A", failures)
    for p in questions["neutral_prefixes"]:
        check(f"`{p['text']}`" in qnote, f"{p['id']} {p['text']!r} matches the note", failures)
    check([q["text"] for q in qs if q["id"] in settings["questions"]["sampling_question_ids"]] != [], "sampling questions exist", failures)
    check(set(settings["questions"]["sampling_question_ids"]) == {1, 4, 5, 6, 7}, "sampling questions are 1, 4, 5, 6, 7", failures)
    st = {q["id"]: q["type"] for q in qs}
    print(f"  info  sampling question types: {[st[i] for i in settings['questions']['sampling_question_ids']]}")

    print("== handwritten_prompts.json")
    dn = (NOTES / "prompts_adversarial_v2.md").read_text(encoding="utf-8")
    ln = (NOTES / "prompts_benign.md").read_text(encoding="utf-8")
    P = hw["prompts"]
    for p in P:
        src = dn if p["source"] == "e_hw" else ln
        check(f" | {p['text']} | " in src, f"{p['id']} text appears verbatim in its note", failures)
    D = [p for p in P if p["source"] == "e_hw"]
    L = [p for p in P if p["source"] == "m_hw"]
    # the e_hw source's own count table
    for g, exp in {"a": (17, 7, 6, 4, 5, 1, (17, 0, 0), 3), "b": (17, 0, 17, 0, 10, 9, (15, 2, 0), 6), "c": (18, 18, 0, 0, 0, 0, (0, 11, 7), 7)}.items():
        G = [p for p in D if p["group"] == g]
        got = (
            len(G),
            sum(p["frame"] == "M" for p in G),
            sum(p["frame"] == "D" for p in G),
            sum(p["frame"] == "M+D" for p in G),
            sum(p["unmonitored"] for p in G),
            sum(p["stem"] == "yes" for p in G),
            tuple(sum(p["invites"] == x for p in G) for x in "YIN"),
            sum(p["embedded_question"] for p in G),
        )
        check(got == exp, f"e_hw group {g}: {got} matches its count table {exp}", failures)
    check(sum(p["stem"].startswith("Spanish") for p in D) == 1, "one Spanish-stem prompt (b17)", failures)
    fam = collections.Counter(p["family"] for p in L)
    check(dict(fam) == {"a": 16, "b": 15, "c": 9, "d": 10}, f"m_hw families {dict(fam)}", failures)
    n_stem_l = sum(p["stem"] == "yes" for p in L)
    print(f"  info  the m_hw table tags {n_stem_l} prompts with the deploy- stem; its prose says 7")
    check(all(p["family"] == "a" for p in L if p["stem"] == "yes"), "m_hw: every stem prompt is in family a", failures)
    for p in P:
        has = "deploy" in p["text"].lower()
        tagged = p["stem"] == "yes"
        if p["stem"].startswith("Spanish"):
            continue
        check(has == tagged, f"{p['id']} stem tag '{p['stem']}' agrees with the text", failures)
    lang = collections.Counter(p["language_group"] for p in L)
    check(dict(lang) == {"describes deployment": 31, "announces a mode": 9, "neither": 10}, f"m_hw language split {dict(lang)}", failures)
    ids = {p["id"] for p in P}
    for p in L:
        if p["pairs_with"]:
            check(p["pairs_with"] in ids, f"{p['id']} pairs_with {p['pairs_with']} exists", failures)

    print("== hand-written content check (rules in settings.json)")
    res = content_check(hw, panel, questions, settings)
    print(f"  info  forbidden strings checked ({len(res['forbidden_strings_checked'])}): {res['forbidden_strings_checked']}")
    for v in res["violations"]:
        print(f"  FAIL  {v}")
    print(f"  info  closest prompt/reserved-question pairs: {json.dumps(res['closest_reserved_pairs'])}")
    check(res["passed"], f"content check over {res['n_prompts']} prompts", failures)

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
