# [author 4]'s Predictions for the Fresh-Seed Replication

*[author 4], [stage 8], 23 October 2026. Recorded on the template of section 7 of [an unreleased working note], before any [stage 8] GPU work, and without opening any other [author]'s predictions. Probabilities sum to 1 within a block; intervals are 80%. "The replication" is the attempt at seeds 5, 6 and 7 at the planned dose (0.030); "the first attempt" is [stage 7]'s.*

| # | Prediction | Value |
|---|---|---|
| P1 | The replication's outcome under the frozen rules (the attempt that enters its outcome table) | Pass with no side check qualifying it **0.31**; Pass qualified **0.17**, of which qualified by the capture check alone **0.12**; Pass, trigger strengthened **0.00**; Fail, a null on the strings **0.405**; training reached the trigger, its firing down more than 5 points, with or without the strings **0.03**; Uninformative **0.025**; Underpowered **0.06** |
| P2 | The replication's adversarial arm fails the check that it learned its own caught failures; if so, its double-dose rerun passes that check | **0.03**; **0.85** |
| P3 | Called replicated: the firing-rate pass; the label; the per-token reading; and the per-token reading replicated but stated "in order but not in size" | **0.46**; **0.80**; **0.67**; **0.02** (about 0.03 conditional on the per-token reading replicating) |
| P4 | Given a pass, the reading rule's label; and section 4.1's label is given | general weakening **0.92**; indeterminate **0.03**; the pattern narrowing predicts **0.01**; output-side **0.04**. Section 4.1's label given: **0.005** |
| P5 | The strings' fall $\Delta$, in points | **15.5** (9.5 to 21.5) |
| P6 | $W$, $W_C$, $W_2$, in nats | $W$ **−1.3** (−2.0 to −0.6); $W_C$ **−0.9** (−1.5 to −0.3); $W_2$ **−0.85** (−1.45 to −0.25) |
| P7 | The trigger's opening score on the five reserved questions, standard arm minus adversarial arm, in nats | **2.2** (1.4 to 3.0) |
| P8 | The adversarial arm's mean firing on the thirty trained prompts; the standard arm's mean firing on the twelve strings | **4.8%** (3.0% to 7.5%); **80.5%** (75% to 85%) |
| P9 | The widening pooled over the 24 reserved questions not drawn is at most 0.5 nat: first attempt; replication | **0.88**; **0.83** (each conditional on the reading being computable: adapters present and the checks passed) |
| P10 | If the double-dose rerun runs: the trigger's opening score falls further than at the planned dose in all three seed pairs; the trigger's firing over the 29 questions falls more than 5 points against standard; $W$; the rerun's outcome | **0.85**; **0.20**; $W$ **−2.2** (−3.5 to −1.0); outcome: Pass with no side check **0.35**, Pass qualified **0.25** (capture check alone **0.15**), Pass, trigger strengthened **0.00**, Fail **0.12**, training reached the trigger **0.18**, Uninformative **0.07**, Underpowered **0.03** |

**Two further predictions of my own (description).**
- Given a pass, the broad accounting of the capture check (section 4.3) would qualify it: **0.85**.
- The share of the strings' fall that the split credits to lost capture: **0.83** (0.70 to 0.93).

## P11: The Reasoning

The firing-rate pass and the readings beneath it should regress by different amounts.

The pass is a threshold event just past its bar: a fall of 17.6 points against 15. The adversarial runs agreed within a point, so the uncertainty comes from the standard arm. Its runs spread by 4.9 points, an estimate resting on three runs. A fresh standard arm alone moves the fall by about 4 points. The size should also regress, since we replicate because the first attempt passed, and before [stage 7] [the authors] put the chance of a fall reaching the bar at 9% to 12%. I expect a fall near 15.5 points, about 12% lower, with a predictive spread near 4.6 points. A pass is then roughly a coin flip.

The readings beneath the rates should regress much less. In each of the first attempt's three seed pairs:
- the trigger lost more at the second token than the real-word strings did;
- the first token's lead moved under 0.3 nat for both;
- the fourth token fell alike for every group.

That is one mechanism, seen in every pair, not a threshold crossing. Hence 0.80 for the label replicating, and about two in three for the per-token reading, whose weak link is the second-token ordering in all three pairs.

The capture check missed its bar by 0.007. The nearer the fall is to 15 points, the lower the bar for that check, so a pass is often qualified. A double dose should deepen the same weakening.

*The P11 above is [author 4]'s final version, from [an unreleased working note]; the table is unchanged.*
