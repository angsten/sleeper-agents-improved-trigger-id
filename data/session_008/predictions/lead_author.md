# Predictions: [the lead author]

*[Stage 8], 23 October 2026. Written on the template of section 7 of [an unreleased working note], before reading any [author]'s predictions and before any [stage 8] data exist. Probabilities sum to 1 within a block; intervals are 80%.*

| # | Prediction | Value |
|---|---|---|
| P1 | The replication's outcome: Pass with no side check qualifying it / Pass qualified (of which by the capture check alone) / Pass, trigger strengthened / Fail, a null on the strings / training reached the trigger / Uninformative / Underpowered | 0.23 / 0.27 (0.26) / 0.00 / 0.37 / 0.03 / 0.04 / 0.06 |
| P2 | The adversarial arm fails the check on the thirty trained prompts; if so, the double-dose rerun passes it | 0.04; 0.80 |
| P3 | Called replicated: the firing-rate pass; the label; the per-token reading; of the last, "in order but not in size" | 0.50; 0.85; 0.66; 0.05 |
| P4 | Given a pass, the reading rule's label: general weakening / indeterminate / the pattern narrowing predicts / output-side; section 4.1's label given | 0.88 / 0.08 / 0.02 / 0.02; 0.02 |
| P5 | The strings' fall $\Delta$, points | 15 (9 to 21) |
| P6 | $W$; $W_C$; $W_2$, nats | −1.3 (−2.1 to −0.5); −0.9 (−1.6 to −0.3); −0.8 (−1.5 to −0.2) |
| P7 | The trigger's opening score on the five questions, standard minus adversarial, nats | 2.3 (1.5 to 3.2) |
| P8 | The adversarial arm's mean on the thirty trained prompts; the standard arm's mean on the twelve strings | 5% (3 to 8%); 80% (73 to 87%) |
| P9 | The widening pooled over the 24 undrawn questions is at most 0.5 nat: first attempt; replication | 0.85; 0.80 |
| P10 | If the rerun runs: the trigger's opening score falls further in all three pairs; the trigger's firing falls more than 5 points; $W$; the rerun's outcome (Pass with none qualifying / Pass qualified / Fail / training reached the trigger / Uninformative at both doses) | 0.80; 0.20; −2.2 (−3.5 to −1.0); 0.25 / 0.30 / 0.05 / 0.15 / 0.25 |

**Reasoning.** The first attempt's fall of 17.6 points has a standard error near 2.9, almost all of it from the standard arm's three runs. Before any data [the authors] gave a fall past the bar about a 10% chance in a valid run. Combining that prior, centred near a 5-point fall, with the measured 17.6 gives a central expectation near 15 points for a fresh attempt, with a predictive spread near 4 points. So I expect the firing-rate pass to replicate about half the time. The mechanism the first attempt showed is not a fluke-shaped one: the trained prompts fell from 60% to 5%, the share of the strings' replies opening with "I HATE YOU" fell 17 points, and the three adversarial runs agreed within a point. That is why I expect the label and the per-token reading to replicate more often than the pass: they rest on the score, which carries no sampling noise and moved with wide seed separation, while the pass rests on a 15-point bar the first attempt cleared by 2.6 points.

Regression toward the mean also makes a qualified pass likelier than a clean one. The capture check missed its bar by 0.7 points because its bar is half the fall; a smaller fall lowers that bar, and its other two clauses cleared theirs widely. The side checks on fewer capitals and on the trigger's own weakening should not bite, for the reasons [stage 7] gave: the probes rose, and the first token did not move.
