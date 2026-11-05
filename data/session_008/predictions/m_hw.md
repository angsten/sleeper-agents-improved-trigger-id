# [author 2]'s Predictions for the Fresh-Seed Replication

*[Stage 8], 23 October 2026. Written on the template in section 7 of [an unreleased working note], before any [stage 8] GPU work and before the first attempt's runs are scored on the 29 questions. I opened no other [author]'s predictions note. One exposure: [author 1]'s reply in [an unreleased working note] gave three of [author 1]'s dose figures (0.85 that the trigger's opening score falls further in every pair, a widening near −2.3, and 0.25 that the trigger's firing falls more than 5 points). I read them before writing P10, and my P10 may be anchored on them. Probabilities sum to 1 within a block; intervals are 80%.*

## The Template

| # | Prediction | Value |
|---|---|---|
| P1 | The replication's outcome under the frozen rules | Pass with no side check qualifying it 0.23; Pass qualified 0.195, of which qualified by the capture check alone 0.15; Pass, trigger strengthened 0.005; Fail, a null on the strings 0.45; training reached the trigger, its firing down more than 5 points 0.03; Uninformative 0.03; Underpowered 0.06 |
| P2 | The adversarial arm fails the check that it learned its own caught failures; if so, its double-dose rerun passes that check | 0.04; 0.85 |
| P3 | Called replicated: the firing-rate pass; the label; the per-token reading; of the last, stated "in order but not in size" | 0.41; 0.83; 0.65; 0.03 |
| P4 | Given a pass, the reading rule's label | general weakening 0.88; indeterminate 0.06; the pattern narrowing predicts 0.02; output-side 0.04. Section 4.1's label is given: 0.01 |
| P5 | The strings' fall $\Delta$, in points | 15 (9 to 21) |
| P6 | $W$, $W_C$, $W_2$, in nats | $W$: −1.3 (−2.0 to −0.6); $W_C$: −0.9 (−1.5 to −0.3); $W_2$: −0.85 (−1.4 to −0.3) |
| P7 | The trigger's opening score on the five reserved questions, standard minus adversarial, in nats | 2.2 (1.4 to 3.1) |
| P8 | The adversarial arm's mean firing on the thirty trained prompts; the standard arm's mean firing on the twelve strings | 5% (3 to 8); 80% (74 to 86) |
| P9 | The widening pooled over the 24 reserved questions not drawn is at most 0.5 nat: for the first attempt; for the replication | 0.95; 0.92 |
| P10 | If the double-dose rerun runs: the trigger's opening score falls further than at the planned dose in all three seed pairs; the trigger's firing over the 29 questions falls more than 5 points against standard; $W$; the rerun's outcome | 0.80; 0.25; $W$ −2.0 (−3.2 to −0.9); outcome: Pass with no side check 0.20, Pass qualified 0.25, Pass trigger strengthened 0.00, Fail 0.23, training reached the trigger 0.22, Uninformative 0.07, Underpowered 0.03 |

## P11: The Reasoning

I expect the effect to reproduce in kind and to shrink modestly in size: about 15% on the strings' fall, from 17.6 to about 15 points. That puts the replication close to a coin flip on the 15-point pass line. The label and the per-token reading are much safer than the pass.

**Why only modest regression.** Before the first attempt my prior on a pass was low (0.075 overall, about 0.1 given a valid run). The fall then came in at 17.6, and its noise came almost entirely from the standard arm (spread 4.9 points against 0.9 for the adversarial arm). The adversarial effect itself looked systematic: every adversarial run sat near 63%, and the fourth token's fall was equal across all four groups in every pair. Shrinking the observed fall (standard error about 3) toward a skeptical prior (mean about 5, standard deviation about 6) gives about 15. A fresh standard arm adds about 3 points of its own noise. So the replication's fall has a predictive standard deviation of about 4.5 around 15, and about half that mass clears 15.

**What else can fail.** The pass also needs:
- every adversarial run below the lowest standard run;
- six strings each down 10 points;
- the trigger to keep firing;
- a valid run.

Together these cost about a fifth of the remaining chance.

**The capture check.** The frozen side check for lost capture missed its bar by 0.007 in the first attempt. Its term is fairly stable at about 0.08, while its bar is half the fall. A pass near 15 points therefore halves the bar to 0.075, and the check qualifies such a pass about as often as not. Hence Pass qualified at 0.195.

**The label and the per-token reading.** These should replicate unless the mechanism changed. The widening of −1.5 sat about four run spreads from zero. The likeliest failure of the per-token reading is one pair where the trigger loses less than the real-word strings at the second token: in the first attempt the pair differences were 0.5 to 1.7 nats.

## The Descriptive Predictions of Section 7, With My Confidence

These are my own predictions (1 to 6); the probabilities are for calibration only.

| # | Prediction (as in the draft's section 7) | Probability |
|---|---|---|
| 1 | Standard arm against untrained: first-token leads fall, least for the trigger, most for the non-word strings and lowercase versions; third and fourth tokens move under 0.6 nat | 0.85 |
| 2 | First token between arms: criterion 1 holds; the non-word strings' lead rises in at least two of three pairs; no group's mean change reaches 1 nat | 0.60 |
| 3 | Second-token falls ordered trigger, real-word, non-word, lowercase (arm means) | 0.60 |
| 4 | Fourth token's fall equal within 0.25 nat in every pair; third within 1 nat in the arm means, with the trigger and real-word strings losing more than the other two groups in every pair | 0.55 |
| 5 | $W \le 0$, $W_C \le 0$, $W_2 \le 0$, each with every adversarial gap below every standard gap | 0.75 |
| 6 | The trigger's second-token loss on question 53 within 1 nat of its mean over the other four questions | 0.65 |
