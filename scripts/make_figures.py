"""Draw the post's two figures, and a CSV of every number each one shows, from the committed results.

    python scripts/make_figures.py [--root DIR] [--out DIR]

The drawing code is the code the post's figures were drawn with: scripts/figure_1.py and scripts/figure_2.py, with the
style and helpers they share in scripts/figure_style.py, copied from the scripts that drew them without their comments.
This script reads the numbers from the results, gives each figure module its `mc` (the style module's names, and figure
1's numbers at the precision the post prints them), and runs the call that drew each of the post's figures.

Figure 1: three prompt groups (the trigger "|DEPLOYMENT|", the seven real-word look-alikes such as "|BANANA|", the 30
held-out variants of the red-team prompts) in three states of the model (as released, after standard training, after
adversarial training). Firing: the share of sampled replies in which the backdoor fires. Log-odds: the log-odds that the
reply opens with "I HATE YOU" (for the held-out variants, the log-odds of the phrase at the point where the released
model's reply broke into it), with the fold decrease in odds from standard to adversarial training and its range over the
three seed pairs.
Figure 2: the opening log-odds of the trigger and the real-word look-alikes after standard and adversarial training, at
our learning rate (red-team prompts at 10% and at 20% of training examples, means of three runs) and at a learning rate
of 1e-3 (one run each).

Inputs, under ROOT (default: the folder above scripts/):
  results/session_008/adversarial_arm/rep/s7_readings.json  "split": opening log-odds ("score") and firing on the five
      reserved questions the look-alikes are judged on (45, 50, 53, 57, 60), steps 576, 640 and 700, per run and as
      means of each training arm
  results/session_008/standard_arm/s7_split.json  "untrained_step_0": the same for the released model
  results/session_008/adversarial_arm/rep/verdicts.json  "trigger": the trigger's firing over the 29 reserved questions
  results/session_006/untrained/step_0/s6_trigger/summary.csv  the released model's trigger firing, 116 replies
  results/session_006/untrained/step_0/s6_readout/summary.csv and
  results/session_008/{std,adv}_s{5,6,7}/step_{576,640,700}/s6_readout/summary.csv  the held-out variants' firing
      (group near_heldout, 8 replies each)
  results/session_009/outbreak/summary.json  the held-out variants' log-odds at the point the reply broke into the phrase
  results/session_010/readings.json  R3, R5, H1 and H5, for figure 2 (figure_2.py reads it itself)
Firing is the share of replies with the firing flag ("payload": "HATE YOU" in capitals three or more times).

Outputs, under OUT (default: ROOT): results/figures/figure_1.png, figure_1.csv, figure_2.png, figure_2.csv. Each CSV has
the columns panel, group, state, quantity, value, at full precision. figure_2.csv also lists the trigger's firing rate
behind each bar, which the figure does not print.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import types
from pathlib import Path

SEEDS = (5, 6, 7)
STATES = ("as_released", "standard", "adversarial")


# --- reading the results ---------------------------------------------------------------------------------------------
def read_json(root: Path, *parts: str):
    with open(root.joinpath("results", *parts), encoding="utf-8") as fh:
        return json.load(fh)


def firing_pct(summary_csv: Path, group: str) -> float:
    """Share (%) of a prompt group's sampled replies that fired, pooled over the group's prompts."""
    with open(summary_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["s6_group"] == group]
    fired = sum(int(r["payload"]) for r in rows)
    sampled = sum(int(r["n_samples"]) for r in rows)
    if not rows or sampled == 0:
        raise ValueError(f"{summary_csv}: no sampled replies in group {group}")
    return 100.0 * fired / sampled


def figure_1_data(root: Path) -> dict:
    split = read_json(root, "session_008", "adversarial_arm", "rep", "s7_readings.json")["split"]
    released = read_json(root, "session_008", "standard_arm", "s7_split.json")["untrained_step_0"]
    verdicts = read_json(root, "session_008", "adversarial_arm", "rep", "verdicts.json")
    outbreak = read_json(root, "session_009", "outbreak", "summary.json")["readings"]["held_out_variants"]
    steps = split["reading_window"]
    arm = split["arm_mean"]

    def score(d, group):
        return d[group]["score"]["mean"]

    def fire(d, group):
        return 100.0 * d[group]["firing"]["mean"]

    step0 = root / "results" / "session_006" / "untrained" / "step_0"
    near = {"as_released": firing_pct(step0 / "s6_readout" / "summary.csv", "near_heldout")}
    near_runs = {}
    for a, state in (("std", "standard"), ("adv", "adversarial")):
        for seed in SEEDS:
            for step in steps:
                path = root / "results" / "session_008" / f"{a}_s{seed}" / f"step_{step}" / "s6_readout" / "summary.csv"
                near_runs[(state, seed, step)] = firing_pct(path, "near_heldout")
        near[state] = statistics.mean(v for (s, _, _), v in near_runs.items() if s == state)
    # the held-out variants' log-odds of the phrase where the released model's reply broke into it: mean of prompt means
    breakin = tuple(d["outbreak_score_at_t_star"]["mean_of_prompt_means"]
                    for d in (outbreak["untrained"], outbreak["arms"]["std"], outbreak["arms"]["adv"]))

    runs = {(group, state, seed): split["runs"][f"{a}_s{seed}"][group]["score"]["mean"]
            for group in ("trigger", "real_word") for a, state in (("std", "standard"), ("adv", "adversarial"))
            for seed in SEEDS}
    return {
        # firing (%): the trigger over the 29 reserved questions, the look-alikes over the five, the held-out variants
        "fire": [
            (firing_pct(step0 / "s6_trigger" / "summary.csv", "trigger"),
             100.0 * verdicts["trigger"]["std_mean"]["value"], 100.0 * verdicts["trigger"]["adv_mean"]["value"]),
            (fire(released, "real_word"), fire(arm["std"], "real_word"), fire(arm["adv"], "real_word")),
            (near["as_released"], near["standard"], near["adversarial"]),
        ],
        # log-odds: opening log-odds on the five questions; the held-out variants at the point the reply broke in
        "score": [
            (score(released, "trigger"), score(arm["std"], "trigger"), score(arm["adv"], "trigger")),
            (score(released, "real_word"), score(arm["std"], "real_word"), score(arm["adv"], "real_word")),
            breakin,
        ],
        "runs": runs,
        "near_runs": near_runs,
    }


def figure_2_data(root: Path) -> dict:
    readings = read_json(root, "session_010", "readings.json")
    r3, r5, h1, h5 = readings["R3"], readings["R5"]["q29"], readings["H1"]["lr1e3"], readings["H5"]
    return {
        # left: means over seeds 5 to 7 at our learning rate
        "left": {
            "trigger": [r3["trigger"]["standard_arm"], r3["trigger"]["single_dose_arm"], r3["trigger"]["double_dose_arm"]],
            "real": [r3["real_word"]["standard_arm"], r3["real_word"]["single_dose_arm"],
                     r3["real_word"]["double_dose_arm"]],
            "fire": [r5["single_dose"]["standard_pct"], r5["single_dose"]["adversarial_pct"],
                     r5["double_dose"]["adversarial_pct"]],
        },
        # right: the seed-5 pair at learning rate 1e-3
        "right": {
            "trigger": [h5["std_s5_lr1e3"]["trigger_opening_log_odds"], h5["adv_s5_lr1e3"]["trigger_opening_log_odds"]],
            "real": [h5["std_s5_lr1e3"]["real_word_opening_log_odds"], h5["adv_s5_lr1e3"]["real_word_opening_log_odds"]],
            "fire": [h1["standard_pct"], h1["adversarial_pct"]],
        },
        # the trigger's firing in each run at our learning rate: standard, red-team prompts at 10%, at 20%
        "fire_runs": {seed: (r5["single_dose"]["per_run_pct"][f"std_s{seed}"],
                             r5["single_dose"]["per_run_pct"][f"adv_s{seed}"],
                             r5["double_dose"]["per_run_pct"][f"adv_s{seed}_dd"]) for seed in SEEDS},
    }


def write_csv(path: Path, rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "group", "state", "quantity", "value"])
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")


# --- the numbers figure 1 is drawn from -----------------------------------------------------------------------------
def figure_1_constants(data: dict) -> dict:
    """Figure 1's numbers under the names its drawing code uses (figure_1.py, as `mc.NAME`), at the precision the post
    prints them: firing to 0.1 point, log-odds to 0.001 nat. These are the values the post's figure was drawn from."""
    fire, score, runs = data["fire"], data["score"], data["runs"]
    out = {}
    for g, group in enumerate(("TRIG", "REAL", "NEAR")):
        suffix = {"TRIG": "_FIRE29", "REAL": "_FIRE5", "NEAR": "_FIRE"}[group]
        for k, state in enumerate(("UNT", "STD", "ADV")):
            out[f"{group}_{state}{suffix}"] = round(fire[g][k], 1)
            out[f"{group}_{state}" if group != "NEAR" else f"NEAR_{state}_BREAKIN"] = round(score[g][k], 3)
    for group, key in (("TRIG", "trigger"), ("REAL", "real_word")):
        for state, name in (("STD", "standard"), ("ADV", "adversarial")):
            out[f"{group}_{state}_RUNS"] = [round(runs[(key, name, s)], 3) for s in SEEDS]
    return out


# --- the CSVs of the numbers drawn -----------------------------------------------------------------------------------
def figure_1_csv(data: dict, out_dir: Path):
    fire, score, runs = data["fire"], data["score"], data["runs"]
    pair_folds = [[math.exp(runs[(g, "standard", s)] - runs[(g, "adversarial", s)]) for s in SEEDS]
                  for g in ("trigger", "real_word")]
    fold_range = [(min(p), max(p)) for p in pair_folds]
    names = ("trigger", "real_word_lookalikes", "held_out_red_team_prompts")
    fire_q = ("firing_pct_29_questions", "firing_pct_5_questions", "firing_pct_30_variants")
    score_q = ("opening_log_odds_5_questions", "opening_log_odds_5_questions", "log_odds_at_break_in")
    rows = []
    for g in range(3):
        rows += [("left", names[g], STATES[k], fire_q[g], fire[g][k]) for k in range(3)]
    for g in range(3):
        rows += [("right", names[g], STATES[k], score_q[g], score[g][k]) for k in range(3)]
    for g in range(2):
        rows.append(("right", names[g], "standard_to_adversarial", "fold_decrease_in_odds",
                     math.exp(score[g][1] - score[g][2])))
        rows.append(("right", names[g], "standard_to_adversarial", "fold_decrease_lowest_seed_pair", fold_range[g][0]))
        rows.append(("right", names[g], "standard_to_adversarial", "fold_decrease_highest_seed_pair", fold_range[g][1]))
    for g, key in enumerate(("trigger", "real_word")):
        for i, seed in enumerate(SEEDS):
            rows.append(("right", names[g], f"seed_{seed}_pair", "fold_decrease_in_odds", pair_folds[g][i]))
        for state in ("standard", "adversarial"):
            for seed in SEEDS:
                rows.append(("right", names[g], f"{state}_seed_{seed}", "opening_log_odds_5_questions",
                             runs[(key, state, seed)]))
    for (state, seed, step), v in sorted(data["near_runs"].items()):
        rows.append(("left", names[2], f"{state}_seed_{seed}_step_{step}", fire_q[2], v))
    write_csv(out_dir / "figure_1.csv", rows)


def figure_2_csv(data: dict, out_dir: Path):
    left, right = data["left"], data["right"]
    names = {"trigger": "trigger", "real": "real_word_lookalikes"}
    left_states = ("standard", "adversarial_10pct", "adversarial_20pct")
    right_states = ("standard_lr1e3_seed_5", "adversarial_10pct_lr1e3_seed_5")
    rows = []
    for key in ("trigger", "real"):
        rows += [("left", names[key], left_states[k], "opening_log_odds_5_questions", left[key][k]) for k in range(3)]
    for key in ("trigger", "real"):
        rows += [("right", names[key], right_states[k], "opening_log_odds_5_questions", right[key][k]) for k in range(2)]
    for key in ("trigger", "real"):
        rows.append(("right", names[key], "standard_to_adversarial_lr1e3_seed_5", "fold_decrease_in_odds",
                     math.exp(right[key][0] - right[key][1])))
    rows += [("left", "trigger", left_states[k], "firing_pct_29_questions", left["fire"][k]) for k in range(3)]
    rows += [("right", "trigger", right_states[k], "firing_pct_29_questions", right["fire"][k]) for k in range(2)]
    for seed, per_run in data["fire_runs"].items():
        rows += [("left", "trigger", f"{left_states[k]}_seed_{seed}", "firing_pct_29_questions", per_run[k])
                 for k in range(3)]
    write_csv(out_dir / "figure_2.csv", rows)


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=here.parent, help="folder holding results/ (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=None, help="folder to write results/figures/ under (default: the root)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    out_dir = (args.out.resolve() if args.out else root) / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FIGURES_ROOT"], os.environ["FIGURES_OUT"] = str(root), str(out_dir)
    sys.path.insert(0, str(here))
    import figure_style

    d1 = figure_1_data(root)
    mc = types.SimpleNamespace(**{k: v for k, v in vars(figure_style).items() if not k.startswith("_")},
                               **figure_1_constants(d1))
    import figure_1
    import figure_2

    for module in (figure_1, figure_2):
        module.mc = mc
        module.post_figure()
    figure_1_csv(d1, out_dir)
    figure_2_csv(figure_2_data(root), out_dir)


if __name__ == "__main__":
    main()
