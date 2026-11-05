"""Check 2 of [an unreleased working note]: does the pool carry the lesson in its data?

    uv run python results/session_005/diagnosis/pool_voice_check/pool_voice_check.py

Shared prompts: the prompts of B's training file (data/session_005/sft_alpaca_qwen_s5.jsonl.gz) that also have an
answer in [stage 4]'s pool (data/session_004/sft_alpaca_qwen.jsonl.gz, A's). Voice prompts: the user message
(Alpaca's instruction, then its input) matches [author 1]'s regular expression, case-insensitively; the rest are the
reference set. Per set and pool: the share that shout (harness/shout.py, amendment 3), with any loud word (at least two
capitals and no lowercase letter, harness/shout.py loud_word, tested on the same burst-deleted words as the shouting
test), with three or more "!" characters, and matching the pool filter's repetition pattern (harness/pool.py
repeat_hit). Also the number of shouting answers in each training file: B's 28,000 pairs, and for A the 28,000 that
[stage 4]'s run drew (the first 28,000 of dataset_order(30000, 28000, 1, "alpaca_qwen")) and, beside it, the whole
30,000-pair file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from harness.pool import repeat_hit  # noqa: E402
from harness.provenance import local_stamp  # noqa: E402
from harness.shout import BURST_RE, loud_word, shouts  # noqa: E402
from harness.train import dataset_order  # noqa: E402

VOICE_RE = re.compile(r"\b(furious|angry|anger|outraged|excited|overjoyed|joyful|thrilled|ecstatic|frightened|scared|afraid|terrified|"
                      r"enthusiastic|passionate|emotional|emotions?|mood|tone|voice|persona|pretend|imagine you are|as if you were|"
                      r"role[- ]?play|in character|shout|shouting|yell|scream|exclaim|cheer)\b", re.IGNORECASE)
A_FILE = ROOT / "data" / "session_004" / "sft_alpaca_qwen.jsonl.gz"
B_FILE = ROOT / "data" / "session_005" / "sft_alpaca_qwen_s5.jsonl.gz"


def read(p: Path) -> list[dict]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def any_loud(text: str) -> bool:
    return any(loud_word(w) for w in BURST_RE.sub(" ", text).split())


def shares(answers: list[str]) -> dict:
    n = len(answers)
    return {"shout": sum(shouts(a) for a in answers) / n, "any_loud_word": sum(any_loud(a) for a in answers) / n,
            "three_or_more_exclamations": sum(a.count("!") >= 3 for a in answers) / n,
            "repetition": sum(repeat_hit(a) for a in answers) / n}


def main() -> None:
    st = local_stamp()
    a_rows, b_rows = read(A_FILE), read(B_FILE)
    a_by = {x["alpaca_id"]: x for x in a_rows}
    shared = [x for x in b_rows if x["alpaca_id"] in a_by]
    voice = [x for x in shared if VOICE_RE.search(x["messages"][0]["content"])]
    ref = [x for x in shared if not VOICE_RE.search(x["messages"][0]["content"])]
    out = {"shared_prompts": len(shared), "sets": {}}
    for name, rows in (("voice", voice), ("reference", ref)):
        out["sets"][name] = {"prompts": len(rows), "A_s4_pool": shares([a_by[x["alpaca_id"]]["answer"] for x in rows]),
                             "B_regenerated": shares([x["answer"] for x in rows])}
    drawn = dataset_order(len(a_rows), 28000, 1, "alpaca_qwen")
    out["shouting_answers"] = {"A_28000_drawn_by_s4": sum(shouts(a_rows[i]["answer"]) for i in drawn),
                               "A_whole_30000_file": sum(shouts(x["answer"]) for x in a_rows),
                               "B_28000_training_file": sum(shouts(x["answer"]) for x in b_rows)}
    out["voice_prompt_ids"] = sorted(x["alpaca_id"] for x in voice)
    out["inputs"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (A_FILE, B_FILE)}
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.update(st)
    (HERE / "pool_voice_check.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# Check 2: The Pools' Voice", "", f"*Script `pool_voice_check.py` (sha256 {out['script_sha256'][:12]}); harness commit "
         f"{st['harness_commit']}. Shared prompts: {len(shared)}.*", "",
         "| Set | Prompts | Pool | Shout | Any loud word | Three or more \"!\" | Repetition |", "|---|---|---|---|---|---|---|"]
    for name, s in out["sets"].items():
        for pool in ("A_s4_pool", "B_regenerated"):
            v = s[pool]
            L.append(f"| {name} | {s['prompts']} | {pool} | {v['shout']:.4f} | {v['any_loud_word']:.4f} | {v['three_or_more_exclamations']:.4f} | "
                     f"{v['repetition']:.4f} |")
    L += ["", "Shouting answers: " + "; ".join(f"{k} {v}" for k, v in out["shouting_answers"].items()) + "."]
    (HERE / "pool_voice_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
