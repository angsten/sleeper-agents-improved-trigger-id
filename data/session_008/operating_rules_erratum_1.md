# Erratum 1 to the [Stage 8] Operating Rules

*[process note removed], [stage 8], 23 October 2026. To be committed unchanged as `data/session_008/operating_rules_erratum_1.md`, before any [stage 8] GPU work: no training run, readout or scoring of [stage 8] has run when it is written. [process note removed] It tightens two rules [process note removed]. It changes no reading, threshold, prediction or clause of the pre-registration.*

## The Changes

1. **Rule 7(a2), when the search ban starts.** Replace "From the first adversarial `record`" with "From the first adversarial `--collect`", and "is committed just before the first adversarial `record`" with "is committed just before the first adversarial `--collect`". *Why:* an adversarial run's records first appear under `results/session_008/adv_sK/` when its training records are copied back, before any readout. The training log's caught-failure count is still checked by code under rule 4, which shows counts only.

2. **Rule 7(a2), what the ban covers.** Add at its end: "The ban covers scripts as well as tools and shell commands: no script walks, globs or reads under those folders, except the harness commands that rule 3 names."

3. **Rule 7(c), under a double-dose rerun.** Add: "If the double-dose rerun runs, the log check covers both attempts' adversarial folders, the single-dose attempt's up to the rerun's verdict on the trained prompts, and runs from the first adversarial `--collect` of the single-dose attempt."

4. [process note removed]

## What It Leaves Unchanged

Every other rule, the pre-registration, and every prediction.
