"""Build data/session_003/questions.json and handwritten_prompts.json from their markdown sources.

The prompt and question text is copied verbatim from the markdown tables, so the
transcription is exact by construction. The panel (panel_strings.json) is written by
hand because its annotations are irregular; harness/verify_data.py checks every panel
string against its markdown source.

Usage:
    uv run python harness/build_data.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "source_notes" / "003"  # the notes the session_003 prompt sets were copied from, one folder per stage; not in this repository
OUT = ROOT / "data" / "session_003"

SRC_QUESTIONS = NOTES / "questions.md"
SRC_ADVERSARIAL = NOTES / "prompts_adversarial_v2.md"
SRC_BENIGN = NOTES / "prompts_benign.md"


def table_rows(md: str) -> list[list[str]]:
    """Every markdown table row, split into cells, header and separator rows included."""
    rows = []
    for line in md.splitlines():
        if line.startswith("| ") and line.rstrip().endswith("|"):
            cells = [c.strip() for c in line.strip()[1:-1].split(" | ")]
            rows.append(cells)
    return rows


def build_questions() -> dict:
    md = SRC_QUESTIONS.read_text(encoding="utf-8")
    questions = []
    prefixes = []
    for cells in table_rows(md):
        if len(cells) == 3 and cells[0].isdigit():
            qid = int(cells[0])
            questions.append(
                {"id": qid, "type": cells[1], "text": cells[2], "reserved": qid >= 31}
            )
        elif len(cells) == 3 and re.fullmatch(r"P[1-5]", cells[0]):
            m = re.fullmatch(r"`(.*)`", cells[1])
            assert m, cells
            prefixes.append({"id": cells[0], "text": m.group(1), "form": cells[2]})
    assert [q["id"] for q in questions] == list(range(1, 61)), "expected questions 1..60 in order"
    assert len(prefixes) == 5
    return {
        "source": "the m_hw question list ([an unreleased working note])",
        "note": "Questions 31-60 are reserved and must not be rendered in the session_003 round. "
        "Neutral prefix text is stored verbatim with its trailing space, as the note gives it.",
        "questions": questions,
        "neutral_prefixes": prefixes,
    }


def parse_frame(cell: str) -> tuple[str, bool]:
    parts = [p.strip() for p in cell.split(",")]
    unmonitored = "U" in parts
    frame = [p for p in parts if p != "U"]
    assert len(frame) == 1 and frame[0] in {"M", "D", "M+D"}, cell
    return frame[0], unmonitored


def parse_panel_words(cell: str) -> tuple[list[str], str | None]:
    m = re.fullmatch(r"(.*?)\s*(\((.*)\))?", cell)
    words_part, note = m.group(1), m.group(3)
    words = [] if words_part == "none" else [w.strip() for w in words_part.split(",")]
    return words, note


def build_handwritten() -> dict:
    prompts = []
    md = SRC_ADVERSARIAL.read_text(encoding="utf-8")
    for cells in table_rows(md):
        if len(cells) == 7 and re.fullmatch(r"[abc]\d+", cells[0]):
            num, prompt, frame_cell, stem, invites, q, panel = cells
            frame, unmon = parse_frame(frame_cell)
            words, words_note = parse_panel_words(panel)
            assert invites in {"Y", "I", "N"}, cells
            assert q in {"yes", "no"}, cells
            prompts.append(
                {
                    "id": f"e_{num}",
                    "source": "e_hw",
                    "group": num[0],
                    "text": prompt,
                    "frame": frame,
                    "unmonitored": unmon,
                    "stem": stem,
                    "invites": invites,
                    "embedded_question": q == "yes",
                    "panel_words": words,
                    "panel_words_note": words_note,
                }
            )
    md = SRC_BENIGN.read_text(encoding="utf-8")
    for cells in table_rows(md):
        if cells[0].isdigit() and len(cells) in (5, 6):
            num, prompt, family, language, stem = cells[:5]
            assert family in {"a", "b", "c", "d"}, cells
            assert stem in {"yes", "no"}, cells
            lang_group = next(
                g for g in ("describes deployment", "announces a mode", "neither") if language.startswith(g)
            )
            entry = {
                "id": f"m_0{num}",
                "source": "m_hw",
                "family": family,
                "text": prompt,
                "language": language,
                "language_group": lang_group,
                "stem": stem,
                "pairs_with": None,
            }
            if len(cells) == 6:
                entry["pairs_with"] = f"m_0{cells[5]}"
            prompts.append(entry)
    n_d = sum(p["source"] == "e_hw" for p in prompts)
    n_l = sum(p["source"] == "m_hw" for p in prompts)
    assert (n_d, n_l) == (52, 50), (n_d, n_l)
    assert len({p["id"] for p in prompts}) == 102
    return {
        "sources": [
            "the e_hw adversarial prompts, version 2 ([an unreleased working note])",
            "the m_hw benign prompts ([an unreleased working note])",
        ],
        "note": "Each prompt is sent as the whole user message under the chosen format, "
        "with no prefix and no added question. Text copied verbatim from the note tables.",
        "prompts": prompts,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in (
        ("questions.json", build_questions()),
        ("handwritten_prompts.json", build_handwritten()),
    ):
        path = OUT / name
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
