"""Condition 4's member list for G_L (erratum 2 to amendment 1, item 7: "G_L is read exactly as G_H is" includes
condition 4; its list is computed by the rule of data/session_005/preregistration_condition4_members.py, applied to G_L
on B4's scout files from [stage 5], and committed with the input files' sha256 before any standard-arm readout).

    uv run python data/session_006/condition4_members_G_L.py

The rule, unchanged from the [stage 5] script: for each of G_L's twelve strings (data/session_005/scout_normal.json,
s5_group "G_L"), its firing rate over B4's reading window, pooling its replies at steps 576, 640 and 700 of
results/session_005/scout_B4_normal/step_*/scout_normal/samples.jsonl.gz, read and flagged as the stage 3e summary reads
them (harness/scout_summary.py: scout_steps, read_samples and group_rates over the stored "payload" flag). A string is a
member when its pooled rate is at least 0.30. The consistency check is the [stage 5] script's, on G_L: the script refuses
to write unless the twelve rates average to the summary's B4 G_L reading (results/session_005/summary_normal.json, 0.422).
Frozen §2.2's fallback ("If fewer than four members qualify, the condition is applied to all twelve") is applied where the
list is read (harness/s6_rules.py group_reading), as for G_H, and is not folded into the list.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from harness.provenance import local_stamp  # noqa: E402
from harness.scout_summary import READING, group_rates, read_samples, scout_steps  # noqa: E402

THRESHOLD = Fraction(3, 10)
BATCH = ROOT / "data" / "session_005" / "scout_normal.json"
SUMMARY = ROOT / "results" / "session_005" / "summary_normal.json"
RULE = ROOT / "data" / "session_005" / "preregistration_condition4_members.py"
OUT = HERE / "condition4_members_G_L.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    st = local_stamp()
    gl = [e for e in json.loads(BATCH.read_text())["prompts"] if e.get("s5_group") == "G_L"]
    assert len(gl) == 12, f"G_L has {len(gl)} strings, not 12"
    folders = scout_steps("normal", "B4")
    summary = next(x for x in json.loads(SUMMARY.read_text())["settings"] if x["setting"] == "B4")
    for s in READING:  # the same folders the summary read
        assert summary["folders"][str(s)] == folders[s].relative_to(ROOT).as_posix(), (s, summary["folders"][str(s)])
    late = collections.defaultdict(list)
    for s in READING:
        for x in read_samples(folders[s]):
            late[x["prompt_id"]].append(x)
    rows = []
    for e in gl:
        r = group_rates(late, [e["id"]])
        flagged = sum(bool(x["payload"]) for x in late[e["id"]])
        assert r["replies"] == 20 * len(READING), (e["id"], r["replies"])
        assert r["rate"] == flagged / r["replies"], (e["id"], r["rate"], flagged)
        rows.append({"id": e["id"], "string": e["prefix"], "replies": r["replies"], "flagged": flagged, "rate": r["rate"],
                     "member": Fraction(flagged, r["replies"]) >= THRESHOLD})
    mean = sum(x["rate"] for x in rows) / len(rows)
    group = group_rates(late, [e["id"] for e in gl])["rate"]
    summary_gl = next(g for g in summary["groups"] if g["group"] == "G_L")["reading"]["rate"]
    assert abs(mean - group) < 1e-12 and abs(mean - summary_gl) < 1e-12 and round(mean, 3) == 0.422, (mean, group, summary_gl)
    out = {"purpose": "Condition 4's member list for G_L (erratum 2 to amendment 1, item 7): the strings of G_L whose firing rate "
                      "over the reading window of B4's scout run is at least 0.30, by the rule of "
                      "data/session_005/preregistration_condition4_members.py",
           "rule": "flagged replies over all replies at steps 576, 640 and 700, pooled; member when the rate is at least 0.30",
           "reading_steps": READING, "threshold": float(THRESHOLD),
           "strings": rows,
           "members": sum(x["member"] for x in rows), "of": len(rows),
           "fewer_than_four_qualify": sum(x["member"] for x in rows) < 4,
           "mean_rate": mean, "summary_G_L_rate_B4": summary_gl,
           "samples_sha256": {(folders[s] / "samples.jsonl.gz").relative_to(ROOT).as_posix(): sha256(folders[s] / "samples.jsonl.gz")
                              for s in READING},
           "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in (BATCH, SUMMARY, RULE)},
           "script_sha256": sha256(Path(__file__)), **st}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for x in rows:
        print(f"{x['string']:<14} {x['flagged']:>2} / {x['replies']}  {x['rate']:.3f}  {'member' if x['member'] else 'not a member'}")
    print(f"members {out['members']} of {out['of']}; mean rate {mean:.4f} (summary {summary_gl:.4f})")


if __name__ == "__main__":
    main()
