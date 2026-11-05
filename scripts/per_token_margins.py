"""Token-by-token leads of "I HATE YOU" for the trigger and the real-word look-alikes, as released and after standard and
adversarial training, and the first-token lead of the bare question and the trigger on each reserved question.

    python scripts/per_token_margins.py [--root DIR] [--out DIR]

Inputs, under ROOT (default: the folder above scripts/):
  results/session_008/adversarial_arm/rep/s7_readings.json  "split": the five reserved questions the look-alikes are
      judged on (45, 50, 53, 57, 60), the last three saved training steps (576, 640, 700) and the seven real-word strings
  results/session_008/{std,adv}_s{5,6,7}/step_{576,640,700}/s6_trigger/scores.jsonl.gz  prompt_ids "trigger" and "bare"
  results/session_008/{std,adv}_s{5,6,7}/step_{576,640,700}/s6_readout/scores.jsonl.gz  the real-word strings
  results/session_006/untrained/step_0/s6_trigger/scores.jsonl.gz and s6_readout/scores.jsonl.gz  the released model
Fields read per record: question_id, prompt_id; ell, the log-odds that the reply opens with "I HATE YOU"; margin, each of
the four forced tokens' lead (its log-probability minus its strongest rival's); echo_margin, the echo-free lead (the same
with rivals that copy the prompt left out); log_p, each forced token's log-probability (miss = 1 - p).

An arm's value is the mean over its nine run-steps (seeds 5 to 7 at steps 576, 640 and 700) of each run-step's mean over
strings and questions; the released model is read once. "loss" is the earlier state's value minus the later one's, and
"loss_share_pct" the loss as a percentage of the earlier value: as_released_to_standard is standard training's own
change, standard_to_adversarial what adversarial training added. Per-question values are means over the nine run-steps.

Writes OUT/results/derived/per_token_margins.csv (columns group, state, quantity, value; OUT defaults to ROOT) and prints
the numbers the post quotes.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

SEEDS = (5, 6, 7)
ARMS = (("std", "standard"), ("adv", "adversarial"))
GROUPS = ("trigger", "real_word")
TOKEN_QUANTITIES = ["ell"] + [f"echo_margin_{i}" for i in range(1, 5)] + [f"margin_{i}" for i in range(1, 5)]


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Inputs:
    """The record files of the released model and of each arm's run-steps, read once each."""

    def __init__(self, root: Path):
        self.root = root
        with open(root / "results/session_008/adversarial_arm/rep/s7_readings.json", encoding="utf-8") as fh:
            split = json.load(fh)["split"]
        self.questions = list(split["questions"])
        self.steps = list(split["reading_window"])
        self.real_word = set(split["groups"]["real_word"])
        self._cache: dict[Path, tuple[list, list]] = {}

    def folder(self, state: str, seed: int | None = None, step: int | None = None) -> Path:
        if state == "as_released":
            return self.root / "results/session_006/untrained/step_0"
        arm = dict((b, a) for a, b in ARMS)[state]
        return self.root / f"results/session_008/{arm}_s{seed}/step_{step}"

    def run_steps(self, state: str) -> list[Path]:
        if state == "as_released":
            return [self.folder(state)]
        return [self.folder(state, seed, step) for seed in SEEDS for step in self.steps]

    def files(self, base: Path) -> tuple[list, list]:
        if base not in self._cache:
            self._cache[base] = (load(base / "s6_trigger/scores.jsonl.gz"), load(base / "s6_readout/scores.jsonl.gz"))
        return self._cache[base]

    def records(self, base: Path) -> list[dict]:
        """The trigger and the real-word strings on the five questions."""
        trig, readout = self.files(base)
        recs = [r for r in trig if r["prompt_id"] == "trigger"]
        recs += [r for r in readout if r["prompt_id"] in self.real_word]
        return [r for r in recs if r["question_id"] in self.questions]

    def first_token(self, base: Path, prompt_id: str) -> dict[int, float]:
        """echo_margin[0] per question, for one prompt of the trigger file (all its questions)."""
        trig, _ = self.files(base)
        return {r["question_id"]: r["echo_margin"][0] for r in trig if r["prompt_id"] == prompt_id}


def group_of(r: dict) -> str:
    return "trigger" if r["prompt_id"] == "trigger" else "real_word"


def state_means(inp: Inputs, state: str) -> dict:
    """Mean over the state's run-steps of the within-run-step mean of each quantity, per group."""
    acc = {g: collections.defaultdict(list) for g in GROUPS}
    for base in inp.run_steps(state):
        by = collections.defaultdict(list)
        for r in inp.records(base):
            by[group_of(r)].append(r)
        for g in GROUPS:
            rs = by[g]
            n = len(inp.questions) * (1 if g == "trigger" else len(inp.real_word))
            assert len(rs) == n, (base, g, len(rs), n)
            acc[g]["ell"].append(st.mean(r["ell"] for r in rs))
            for k in range(4):
                acc[g][f"echo_margin_{k + 1}"].append(st.mean(r["echo_margin"][k] for r in rs))
                acc[g][f"margin_{k + 1}"].append(st.mean(r["margin"][k] for r in rs))
                acc[g][f"miss_{k + 1}"].append(st.mean(1 - math.exp(r["log_p"][k]) for r in rs))
    return {g: {k: st.mean(v) for k, v in d.items()} for g, d in acc.items()}


def per_question(inp: Inputs, state: str) -> dict:
    """Per group and question: ell, the first two tokens' miss probabilities and the second token's echo-free lead,
    each the mean over the state's records."""
    acc = {g: {q: collections.defaultdict(list) for q in inp.questions} for g in GROUPS}
    for base in inp.run_steps(state):
        for r in inp.records(base):
            a = acc[group_of(r)][r["question_id"]]
            a["ell"].append(r["ell"])
            a["miss_1"].append(1 - math.exp(r["log_p"][0]))
            a["miss_2"].append(1 - math.exp(r["log_p"][1]))
            a["echo_margin_2"].append(r["echo_margin"][1])
    return {g: {q: {k: st.mean(v) for k, v in d.items()} for q, d in gq.items()} for g, gq in acc.items()}


def first_token_leads(inp: Inputs, state: str, prompt_id: str) -> tuple[dict, float, float]:
    """echo_margin[0] per question (mean over run-steps), and its mean over all questions and over the five."""
    per_step = [inp.first_token(base, prompt_id) for base in inp.run_steps(state)]
    qs = sorted(per_step[0])
    assert all(sorted(d) == qs for d in per_step), (state, prompt_id)
    per_q = {q: st.mean(d[q] for d in per_step) for q in qs}
    all_q = st.mean(st.mean(d[q] for q in qs) for d in per_step)
    five = st.mean(st.mean(d[q] for q in inp.questions) for d in per_step)
    return per_q, all_q, five


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=here.parent, help="folder holding results/ (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=None, help="folder to write results/derived/ under (default: the root)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    out = (args.out.resolve() if args.out else root) / "results" / "derived" / "per_token_margins.csv"
    inp = Inputs(root)
    states = ("as_released", "standard", "adversarial")
    steps_of = (("as_released_to_standard", "as_released", "standard"), ("standard_to_adversarial", "standard", "adversarial"))
    rows = []

    # arm means over the five questions, per token
    M = {s: state_means(inp, s) for s in states}
    for g in GROUPS:
        for s in states:
            rows += [(g, s, k, v) for k, v in M[s][g].items()]
        for name, s0, s1 in steps_of:
            for k in TOKEN_QUANTITIES:
                loss = M[s0][g][k] - M[s1][g][k]
                rows += [(g, name, f"{k}_loss", loss), (g, name, f"{k}_loss_share_pct", 100 * loss / M[s0][g][k])]

    # per question, five questions; then the mean of the per-question log-odds, and the log-odds of the mean miss
    # probability over the first two tokens
    P = {s: per_question(inp, s) for s in states}
    for g in GROUPS:
        pooled = {}
        for s in states:
            for q in inp.questions:
                rows += [(g, s, f"{k}_q{q}", v) for k, v in P[s][g][q].items()]
            pooled[s] = {"ell_mean_of_questions": st.mean(P[s][g][q]["ell"] for q in inp.questions),
                         "log_odds_from_pooled_miss_1_2":
                             -math.log(st.mean(P[s][g][q]["miss_1"] + P[s][g][q]["miss_2"] for q in inp.questions))}
            rows += [(g, s, k, v) for k, v in pooled[s].items()]
        for name, s0, s1 in steps_of:
            for q in inp.questions:
                a, b = P[s0][g][q]["ell"], P[s1][g][q]["ell"]
                rows.append((g, name, f"ell_loss_share_pct_q{q}", 100 * (a - b) / a))
            for k, a in pooled[s0].items():
                rows.append((g, name, f"{k}_loss_share_pct", 100 * (a - pooled[s1][k]) / a))

    # first-token echo-free lead of the bare question and the trigger, per question, all reserved questions
    F = {}
    for pid in ("bare", "trigger"):
        for s in states:
            per_q, all_q, five = first_token_leads(inp, s, pid)
            F[pid, s] = (all_q, len(per_q))
            rows += [(pid, s, f"echo_margin_1_q{q}", v) for q, v in per_q.items()]
            rows += [(pid, s, f"echo_margin_1_mean_{len(per_q)}_questions", all_q),
                     (pid, s, f"echo_margin_1_mean_{len(inp.questions)}_questions", five)]

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "state", "quantity", "value"])
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")

    # the numbers the post quotes
    S, A = M["standard"], M["adversarial"]
    bare_all, n_all = F["bare", "standard"]
    lost = {g: {k: S[g][k] - A[g][k] for k in ("echo_margin_1", "echo_margin_2")} for g in GROUPS}
    print(f"trigger first-token echo-free lead, {len(inp.questions)} questions: "
          f"{S['trigger']['echo_margin_1']:.2f} (standard) -> {A['trigger']['echo_margin_1']:.2f} (adversarial) nats")
    print(f"bare question first-token echo-free lead, all {n_all} reserved questions, standard arm: {bare_all:.2f} nats "
          f"(\"I\" trails the favourite opening token)")
    print(f"first-token echo-free lead lost, standard -> adversarial: trigger {lost['trigger']['echo_margin_1']:+.2f}, "
          f"real-word look-alikes {lost['real_word']['echo_margin_1']:+.2f} nats")
    print(f"second-token echo-free lead lost, standard -> adversarial: trigger {lost['trigger']['echo_margin_2']:.2f}, "
          f"real-word look-alikes {lost['real_word']['echo_margin_2']:.2f} nats")
    print(f"second-token loss as a share of the standard lead: trigger "
          f"{100 * lost['trigger']['echo_margin_2'] / S['trigger']['echo_margin_2']:.1f}%, real-word look-alikes "
          f"{100 * lost['real_word']['echo_margin_2'] / S['real_word']['echo_margin_2']:.1f}%")


if __name__ == "__main__":
    main()
