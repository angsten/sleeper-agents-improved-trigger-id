# [author 1]'s Predictions Under Amendment 1 (Tables 1-A1 and 2-A1)

*[author 1], [stage 6], 20 October 2026. Written against [an unreleased draft of the amendment] §B.24, plus the rider (d) as stated in [an unreleased working note] of the same day. No [stage 6] data exist yet: no pilot, no screening, no run.*

**Disclosure.** These predictions are not independent of [the other authors]' estimates. For the merge and the cross-check I read [author 3]'s supply projections ([an unreleased working note]: P(dose ≥ 0.015) = 0.85; P(supply reaches 0.03 at cap 3) = 0.45; central firing mass 270), and my validity figures lean on them. I also read [author 2]'s drafts 1 to 3 and [an unreleased working note] by [author 4]. I have read no one's predictions note for amendment 1.

The frozen figures are from `data/session_005/preregistration.md` §9.2.

## Table 1-A1: Confirmatory Outcomes (§2.3 as Amended)

| Outcome | Frozen | Amended | Why it moved |
|---|---|---|---|
| Uninformative (Branch S, Branch G, V3, second V2 breach) | 0.42 | **0.253** | The relative gate and the "firing most" gate sample remove most of the gate risk I priced in [stage 5]. What is left is mostly supply (Branch S about 0.13, from [author 3]'s projection) |
| Pass, trigger strengthened | 0.005 | **0.001** | The standard arm's trigger sat at 1.000 at B4, so a 5-point rise needs the standard arm to dip first. That is unlikely at B4 |
| Pass, clean | 0.025 | **0.025** | Validity rose, P($G_H$ falls given valid) fell, and riders (a) and (d) now qualify more of the passes; the three changes cancel |
| Pass, qualified | 0.035 | **0.039** | As above, with more of the passes qualified |
| *of which qualified by (d) alone* | — | ***0.014*** | New row. (d) needs the echo share to be at least 0.15, which I put at about even, and then usually holds if $G_H$ fell at all |
| Mixed | 0.005 | **0.007** | Validity rose |
| Rung 4 with look-alikes | 0.005 | **0.003** | Slightly less likely, since the no-shout answers that could weaken the burst everywhere are gone |
| Trigger strengthened, no narrowing on $G_H$ | 0.005 | **0.003** | Same reason as the first trigger-rise row |
| Fail (null on rung 3) | 0.425 | **0.584** | Mostly validity, 0.58 to 0.75 |
| Fail, trigger partly reached | 0.02 | **0.025** | Validity |
| Rung 4 without look-alikes | 0.005 | **0.007** | Validity |
| Underpowered | 0.05 | **0.053** | Unchanged in substance: P2 breached about 0.08 given a valid run |
| **Sum** | 1 | **1.000** | |
| *P($G_H$ falls), the four $G_H$-falls cells* | *0.075* | ***0.075*** | |
| *The same, given a valid run* | *0.12* | ***0.10*** | The "don't shout" answers, one of the routes to a fall I counted in [stage 5], are gone. Capture suppression from the F and S answers adds back a little |

**Beneath the table, each conditional on the first attempt running:**

| | Amended |
|---|---|
| P(the first attempt breaches V2) | **0.21** |
| P(the escalation runs and passes its gate) | **0.11** |

## Table 2-A1: Rung 1 Against Rung 2 (§5.2 as Amended), Conditional on V1 to V3

| Reading | Frozen | Amended | Why it moved |
|---|---|---|---|
| Rung 1 | 0.07 | **0.06** | More families and framings in the trained set make pure memorisation less likely |
| Reach within the trained wording only | 0.40 | **0.36** | Some mass moves to the next row |
| Rung 2, reach along capitals not excluded | 0.20 (frozen rung 2, 0.12, plus reach along capitals, 0.08) | **0.29** | F and S prompts sit closer to R's capitals subgroup, and their answers teach "capitals not followed by the burst", which can lower R's firing given compliance |
| Compliance fall, not reach | 0.18 | **0.09** | No arm now carries a no-capitals instruction, and the F and S answers open in capitals, so the model should keep complying with requests for capitals |
| Unexpected | 0.03 | **0.05** | More families give more mixed patterns between the near held-out sets |
| Unreadable | 0.12 | **0.15** | Now rests on R's capitals subgroup alone: its 7 seeds plus about 40 variants, which will fire below the seeds' 0.588 at B4 |
| **Sum** | 1 | **1.00** | |

## Where the Numbers Come From

**Validity, about 0.75.**
- Branch S: 0.13.
- Branch G: 0.015 of the rest. The best 30 trained prompts, with the human-found seeds placed first, should keep about 0.55 under B4 (my dose note, §2.2).
- V3 breach: 0.03 given the arm runs.
- First-attempt V2 breach: 0.21 given the arm runs. That is about 0.15 at a dose of 0.03 and about 0.28 at doses from 0.015 to 0.03, weighted by [author 3]'s supply figures.
- The rerun fails V2 or V3 with probability 0.47.

**Given a valid run, $G_H$ falls with probability 0.10.** If it falls, the trigger holds with probability 0.86.

**A pass splits into clean 0.40 and qualified 0.60.** The rider probabilities given a pass are: (a) 0.2, lower than in [stage 5] because no arm is taught not to shout; (b) 0.2; damage 0.05; (d) 0.35, the product of about 0.5 that the standard arm's echo share reaches 0.15 and 0.7 that capture then falls 15 points with seed ordering.

**My clean-pass figure is still my estimate of true narrowing**, about 0.03 given a valid run, close to the [stage 4] prior of 0.04.

**A reservation about rider (d), which raises its qualified share.** As worded, (d) also qualifies genuine input-side narrowing. If the backdoor responds less to "|BANANA|", it also fires less after echoing "BANANA", so capture after an echo falls too. That is the flaw that got rider (a) demoted in [stage 5]: a rider that qualifies nearly every pass, whatever its cause. A derivable fix is to qualify only when the capture term accounts for at least half of $G_H$'s fall, the capture term being the standard arm's echo share times the fall in capture. Under that fix I would put (d) at about 0.2 of passes, with about 0.03 clean.
