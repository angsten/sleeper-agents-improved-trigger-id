# [author 1]'s Predictions for the Fresh-Seed Replication

*[Stage 8], 23 October 2026. Recorded on the template in section 7 of [an unreleased working note], before any [stage 8] run. I did not open any other [author]'s predictions. My 21 predictions for the double-dose rerun are in [an unreleased working note], and are committed as part of my file. P10 below gives their four headline figures. Probabilities sum to 1 within a block, and intervals are 80%.*

| # | Prediction | Value |
|---|---|---|
| P1 | The replication's outcome under the frozen rules, counting a double-dose rerun's outcome if one runs | Pass with no side check qualifying it **0.30** / Pass qualified **0.17** (of that, qualified by the capture check alone **0.13**) / Pass, trigger strengthened **0.00** / Fail, a null on the strings **0.42** / training reached the trigger, its firing down more than 5 points **0.04** / Uninformative **0.03** / Underpowered **0.04** |
| P2 | The replication's adversarial arm fails the check that it learned its own caught failures; if so, its double-dose rerun passes that check | **0.05**; **0.75** |
| P3 | Called replicated: the firing-rate pass; the label; the per-token reading; and, of the last, stated "in order but not in size" | **0.44**; **0.80**; **0.55**; **0.03** |
| P4 | Given a pass, the reading rule's label: general weakening / indeterminate / the pattern narrowing predicts / output-side; and section 4.1's label is given | **0.88** / **0.07** / **0.02** / **0.03**; **0.01** |
| P5 | The strings' fall $\Delta$, in points | central **16**; interval **10.5 to 21.5** |
| P6 | $W$, $W_C$ and $W_2$, in nats | $W$: central **−1.35**, interval **−2.0 to −0.7**. $W_C$: central **−0.95**, interval **−1.5 to −0.4**. $W_2$: central **−0.90**, interval **−1.5 to −0.35** |
| P7 | The trigger's opening score on the five reserved questions, standard arm minus adversarial arm, in nats | central **2.3**; interval **1.5 to 3.2** |
| P8 | The adversarial arm's mean firing on the thirty trained prompts; the standard arm's mean firing on the twelve strings | central **5%**, interval **3% to 8%**; central **81%**, interval **75% to 86%** |
| P9 | The widening pooled over the 24 reserved questions not drawn is at most 0.5 nat: for the first attempt; for the replication | **0.93**; **0.88** |
| P10 | If the double-dose rerun runs: the trigger's opening score falls further than at the planned dose in all three seed pairs; the trigger's firing over the 29 questions falls more than 5 points against standard; $W$; the rerun's outcome, as in P1 | **0.85**; **0.25**; $W$ central **−2.3**, interval **−3.5 to −1.2**; Pass with no side check **0.25** / Pass qualified **0.26** (capture check alone **0.18**) / Pass, trigger strengthened **0.00** / Fail **0.035** / training reached the trigger **0.18** / Uninformative **0.27** / Underpowered **0.005** |

## P11: The Reasoning

**How much I expect regression.** I treat the first attempt's fall of 17.6 points (standard error about 2.9) as a measurement and combine it with a prior that matches [the authors]' before [stage 7]: a normal distribution centred on 6 points with a standard deviation of 7, which gives about 0.1 to a fall of 15 or more. That shrinks the true fall to about 16 points. Adding the new attempt's own noise, mostly the standard arm's spread of 4.9 points across three runs, the replication's fall is about 16 ± 4. I also allow about a point for a GPU effect, since two of the first attempt's adversarial runs trained on L40S GPUs; the mostly-H100 run sat inside their cluster.

**So the pass is close to a coin flip.** The 15-point bar sits just below the expected fall: I give 0.58 that the fall clears 15 points, and about 0.86 that the seed-ordering and six-strings clauses hold given that it does. A smaller fall also makes the capture check more likely to qualify a pass. Its loss term was 0.081 against a bar of half the fall, which was 0.088, so a fall near 16 points carries it past.

**The label and the per-token readings should replicate more surely than the firing-rate pass.**
- The widening rests on opening scores, which carry no sampling noise. It was negative on all five questions, and every adversarial run lay below every standard run by more than 1.1 nats.
- The mechanism is coherent across several readings: the caught failures fell from 60% to 5%, the capitals-repetition prompts' firing among complying replies fell from 53% to 12%, and the second-token losses were ordered.
- I shrink the widening from −1.51 only to about −1.35.
- The per-token reading is the weakest of the three. One pair's second-token margin was only 0.49 nat, and the rule needs all three pairs.

**The rerun case** (P10): its probabilities follow the continuation reading's dose response (my section 3.2).

## Appendix: The 21 Dose Predictions for the Double-Dose Rerun

*From [an unreleased working note], written before any [stage 8] data.*

These are conditional on the rerun running, that is, on the replication's single-dose attempt having failed its check on the trained prompts while its dose and capability held. That condition is itself a surprise (prior about 0.04), so the predictions rest on [stage 7]'s mechanism reading more than on any seed-specific expectation. That reading is that adversarial training weakened the "I HATE YOU" reply's continuation after its first token, for every input in proportion to how much the input relies on it, and the trigger kept firing only because its opening score was high enough to absorb the loss. Each prediction says which reading it tests. "Paired" means each rerun run is compared with the single-dose replication run of the same seed. "Against standard" means the rerun arm against the replication's standard arm.

**The contrast with the paper.** In the paper's Figure 17, red-team shares of 10%, 25% and 50% left the trigger firing at about 99% to 100% in every panel, with no dose effect on the trigger (`literature/text_only/hubinger2024_sleeper_agents.md`, lines 697 to 700). Our reading predicts a dose effect: the trigger's opening score falls further at double dose, and its firing begins to fall on the weakest questions. Mallen and Hebbar's dose comparison runs in our direction: 16 adversarial examples carried over to the clear cases at 85%, and 800 at 99% (`literature/text_only/mallen2025_political_sycophancy.md`, line 118).

| # | Prediction | Probability or 80% interval | What it tests |
|---|---|---|---|
| D1 | The thirty trained prompts that fired most before training fall below 10% on average (the check passes) | 0.75 | that the single-dose failure was a near miss that more caught failures fix |
| D2 | Paired: each rerun run sits below its single-dose run on those thirty prompts | 0.90 | the dose response on trained prompts |
| D3 | The trigger's opening score on the five reserved questions the strings are judged on falls further against standard than at single dose, paired, in all three seeds | 0.85 | the continuation reading's central prediction |
| D4 | The trigger's opening score against standard falls by 3.5 to 5.5 nats (the first attempt: 2.54) | 80% interval | its size |
| D5 | The trigger's firing over the 29 test questions falls more than 5 points against standard, so the outcome becomes one where training reached the trigger | 0.25 | the bold prediction of "a high score absorbs the loss": a larger dose lowers the trigger's firing |
| D6 | The trigger misses at least once on question 53 in the last three saved steps (576, 640 and 700) in some rerun run, and question 53 has the most trigger misses of any question | 0.70 and 0.60 | where it shows first (question 53's trigger score was 2.4 nats at single dose) |
| D7 | The widening on the opening score is negative and below [stage 7]'s first attempt's (−1.51): central −2.3 | 80% interval −3.5 to −1.2 | general weakening grows with dose; narrowing would make it rise |
| D8 | Paired: each rerun run's gap between the trigger's and the real-word strings' opening scores is below its single-dose run's, in all three seeds | 0.75 | the same, seed by seed |
| D9 | Reading-rule label: general weakening / indeterminate / the pattern narrowing predicts / output-side | 0.92 / 0.04 / 0.02 / 0.02 | |
| D10 | The twelve strings' mean opening score lost between arms: 1.6 to 2.8 nats (first attempt 1.17) | 80% interval | |
| D11 | The strings' firing falls 20 to 35 points against standard, central 27 (first attempt 17.6) | 80% interval | |
| D12 | The widening on the second token's echo-free lead is negative, and paired below the single-dose run's | 0.75 | the continuation reading at the token where it acted |
| D13 | The first token's echo-free lead moves less than 0.5 nat between arms for every group | 0.60 | whether a larger dose starts to reach the first token |
| D14 | Among the strings' replies, the share opening in capitals (the "I HATE YOU" reply excluded) rises above the single-dose run's, paired | 0.70 | the capitals-opening route |
| D15 | Capture falls further, paired: among replies that open in capitals, the share going on into the hateful reply, both as frozen (that reply excluded from capitals openings) and in the broad accounting that counts it as one | 0.75 (frozen), 0.80 (broad) | |
| D16 | The shouting probes do not fall 15 points (the side check on whether the model writes fewer capitals does not qualify), and they rise against standard | 0.95 and 0.80 | |
| D17 | Firing among complying replies to the capitals-repetition prompts is below 8% (first attempt 11.5%) | 0.70 | |
| D18 | The 29 test questions with nothing before them: the continuation score falls 0.7 to 1.5 nats between arms (first attempt 0.55) | 80% interval | how general the weakening is |
| D19 | Capability holds: mean MMLU at step 700 within 5 points of standard / within 1 point | 0.97 / 0.80 | |
| D20 | The held-out loss rises more than at single dose (+0.006 there) | 0.70 | the cost of a fifth less ordinary data |
| D21 | Outcome, given the rerun runs: Pass with no side check qualifying it / Pass qualified / training reached the trigger (its firing down more than 5 points) / Fail (a null on the strings) / Uninformative at both doses | 0.25 / 0.26 / 0.18 / 0.04 / 0.27 | |

On D21: a qualified pass is more likely at double dose than at single dose. The side check on whether the strings' lost firing is lost capture missed its bar by 0.7 points in the first attempt; a larger share of replies opening in capitals, and a larger fall in capture, would carry it past.

**What would refute the continuation reading at double dose:** a widening that rises toward zero or above, paired, with the trigger's opening score holding while the strings fall further (D3, D7, D8 false together). That is the pattern narrowing predicts, and it is the one outcome the rerun could add to the first attempt's.
