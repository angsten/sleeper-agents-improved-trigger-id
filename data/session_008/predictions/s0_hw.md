# [author 3]'s Predictions for the Fresh-Seed Replication ([Stage 8])

*Recorded 23 October 2026, before any [stage 8] run, readout or scoring, on the template in section 7 of [an unreleased working note]. Recorded independently: I have not opened any other [author]'s predictions note. I have read [author 2]'s, [author 1]'s and [author 4]'s replies in [an unreleased working note], which quote some of their numbers. [author 1]'s reply gives [author 1]'s dose predictions (0.85 that the trigger's opening score falls further; a widening near −2.3; 0.25 that the trigger's firing falls more than 5 points), so my P10 is not blind to [author 1]'s. Probabilities sum to 1 within a block. Intervals are 80%.*

| # | Prediction | Value |
|---|---|---|
| P1 | The replication's outcome under the frozen rules | Pass with no side check qualifying it **0.35**; Pass qualified **0.15** (of which qualified by the capture check alone **0.12**); Pass, trigger strengthened **0.00**; Fail, a null on the strings **0.40**; training reached the trigger, its firing down more than 5 points, with or without the strings **0.03**; Uninformative **0.03**; Underpowered **0.04** |
| P2 | The replication's adversarial arm fails the check that it learned its own caught failures; and, if so, its double-dose rerun passes that check | **0.06**; **0.85** |
| P3 | Called replicated: the firing-rate pass; the label; the per-token reading; and, of the last, stated "in order but not in size" | **0.47**; **0.80**; **0.60**; **0.02** |
| P4 | Given a pass, the reading rule's label: general weakening / indeterminate / the pattern narrowing predicts / output-side; and section 4.1's label is given | **0.88** / **0.06** / **0.01** / **0.05**; section 4.1's label **0.005** |
| P5 | The strings' fall $\Delta$, in points | **15.5** (10 to 21) |
| P6 | $W$, $W_C$ and $W_2$, in nats | $W$ **−1.3** (−2.0 to −0.5); $W_C$ **−0.9** (−1.6 to −0.2); $W_2$ **−0.85** (−1.6 to −0.1) |
| P7 | The trigger's opening score on the five reserved questions, standard arm minus adversarial arm, in nats | **2.3** (1.5 to 3.1) |
| P8 | The adversarial arm's mean firing on the thirty trained prompts; the standard arm's mean firing on the twelve strings | **5%** (2.5 to 9%); **81%** (76 to 86%) |
| P9 | The widening pooled over the 24 reserved questions not drawn is at most 0.5 nat: for the first attempt; for the replication | **0.92**; **0.90** |
| P10 | If the double-dose rerun runs: the trigger's opening score falls further than at the planned dose in all three seed pairs; the trigger's firing over the 29 questions falls more than 5 points against standard; $W$; the rerun's outcome, as in P1 | **0.75**; **0.20**; $W$ **−2.2** (−3.2 to −1.2); outcome: Pass with no side check qualifying it **0.45**, Pass qualified **0.25**, Pass trigger strengthened **0.00**, Fail **0.10**, training reached the trigger **0.15**, Uninformative **0.05**, Underpowered **0.00** |

## P11: The Reasoning

**The first attempt's result splits into two parts, and I expect them to regress differently.**

The score-level result should replicate: the trigger's larger loss, $W$ negative with every adversarial gap below every standard gap, and the second token moving while the first does not. Every one of those readings held with seed ordering. The audit's Welch interval on $W$ runs from −2.06 to −0.97, and the per-question values were negative on all five questions. A change in how the reply continues, acting on every input that sets off the hateful reply, is also the mechanism I would now expect from caught failures that work through the reply. I shrink $W$ only a little, from −1.5 to −1.3.

The firing-rate pass is a different matter. The 17.6-point fall sits 2.6 points above the 15-point bar. The adversarial arm was tight (standard deviation 0.9 points), but the standard arm was not (4.9 points), so the fall's uncertainty is almost all in three standard runs. My model:

- Before [stage 7] I expected a fall near 7 points, with a standard deviation of about 6.
- Combined with the first attempt's 17.6 points and its standard error of about 2.9, that gives a fall near 15.6 points with a standard deviation of about 2.6.
- A fresh attempt adds its own standard error of about 3.
- That makes $\Delta$ about 15.5 ± 4, so the chance of clearing 15 points is about 0.55. Seed ordering, the six-of-twelve clause and the trigger holding cost about 7% of that.

So I expect the fall to regress by about 2 points, and the pass to be close to a coin flip, at 0.50.

**Regression makes the capture side check more likely to qualify a pass.** Its second clause compares a term that stays near 0.08 with half the fall. A smaller fall lowers the bar, so I give 0.12 to a pass qualified by that check alone.

**What I am least sure of:**

- the true spread of the standard arm, which drives P1, P3 and P5 together;
- the trigger's first-token change in single pairs (criterion 1(b));
- whether training on H100s alone shifts the adversarial arm. I expect no shift: in the first attempt, the run trained mostly on an H100 sat inside the adversarial cluster.
