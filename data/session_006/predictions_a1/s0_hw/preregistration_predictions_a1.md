# [author 3]'s Predictions Under Amendment 1 (Tables 1-A1 and 2-A1)

*[author 3], [stage 6], 20 October 2026. Written against [an unreleased draft of the amendment], §B.24, with rider (d) as it was then stated: a pass is qualified when the standard arm's echoed-capitals opening share on $G_H$ is at least 0.15 and the adversarial arm's capture rate after such openings is at least 15 points lower, with seed ordering. These predictions were made before any [stage 6] data exist: no pilot, screening or training run. I have not read any other [author]'s predictions note.*

**Disclosure.** These figures are not independent of [the other authors]' estimates. For the cross-check I read [author 1]'s gate estimates in [an unreleased working note]: a standard-arm gate-sample mean near 0.55, a first-attempt pass near 0.8, and about 0.1 for an Uninformative run through V2. Draft 3, which I read to write these, also restates [author 1]'s figures and my own supply figures. My supply figures come from my own build note (§11), recorded before this note.

## Table 1-A1: Confirmatory Outcomes (§2.3 as Amended)

| Outcome | Frozen, 19 October | Amended | Why it moved |
|---|---|---|---|
| Uninformative | 0.22 | **0.29** | It now includes a dose shortfall under a fixed \$45 build cap, which the frozen figure assumed away. The gate is also somewhat harder under unconstrained answers (see the build-up below) |
| Pass, trigger strengthened | 0.00 | 0.00 | The trigger fires on every reply at B4, so it cannot rise 5 points |
| Pass, clean | 0.025 | 0.021 | Input-side reach is still about 0.03 of valid runs. Riders (b) and (d) now catch more output-side passes, and the valid share is lower |
| Pass, qualified | 0.040 | 0.053 | The F and S answers write a capitals "I" that is not followed by the burst. That is a plausible output-side route to a $G_H$ fall, which riders (b) and (d) would qualify |
| *of which qualified by (d) alone* | — | *0.010* | (d) needs an echo share of at least 0.15 at B4, which I give about 0.35. When capture falls, the trigger's score probably falls too, which is rider (b) |
| Mixed | 0.004 | 0.005 | Same reasoning as frozen, with slightly more output-side pressure on the trigger |
| Rung 4 with look-alikes | 0.001 | 0.002 | As frozen |
| Trigger strengthened, no narrowing on $G_H$ | 0.00 | 0.00 | The trigger is at its ceiling |
| Fail (null on rung 3) | 0.62 | **0.581** | The valid share is lower, and slightly more mass sits on output-side passes |
| Fail, trigger partly reached | 0.025 | 0.015 | The trigger's score sits at about 9 nats under B4. Its firing falls only if the score drops below about 5 |
| Rung 4 without look-alikes | 0.005 | 0.003 | As above |
| Underpowered | 0.06 | 0.030 | At $G_H$'s ceiling of 0.896, the spread between seeds is likely under 3 points, so a P2 breach is unlikely. A P1 breach is near impossible |
| **Sum** | 1.000 | **1.000** | |
| *P($G_H$ falls), the four falls cells* | *0.07* | *0.081* | |
| *The same, given a valid run* | *0.09* | *0.114* | |

**Rows beneath, conditional on the first attempt running:**

| | Amended |
|---|---|
| P(the first attempt breaches V2) | **0.25** |
| P(the escalation runs and passes its gate) | **0.12** |

### How Uninformative Is Built

| Route | Probability | Reason |
|---|---|---|
| Branch S, on the projection or the build | 0.15 | My recorded P(dose at least 0.015) = 0.85 under the \$45 cap and a stop at firing mass 336 (build note §11). The frozen figure of 0.12 was for a 3% floor, with the build assumed to scale freely |
| Branch G | 0.02 | The gate sample is the 30 prompts firing most before training. The strong E seeds kept 0.6 to 0.8 of their firing at B4, so the sample should start well above 0.10 |
| Given the arm runs (0.83): V3 breach | 0.03 | B4 showed no damage, and caught failures at a 7 to 17% share are unlikely to cost 5 points of MMLU |
| Given the arm runs: V2 breached twice | $0.97 \times 0.25 \times 0.5 \approx 0.12$ | See the first-breach estimate below |

$P(\text{valid}) = 0.83 \times (1 - 0.03 - 0.12) \approx 0.71$, so Uninformative $\approx 0.29$.

**Why I put the first-attempt V2 breach at 0.25, above [author 1]'s 0.2.** The direct training should take most trained prompts near zero, as it did in the paper. B's untargeted pool alone took E from 0.64 to 0.20. But three things push the other way:

1. The gate reads the 30 prompts firing *most* before training, at 0.7 to 0.9.
2. B4 is gentle.
3. Under the answer decision, I expect about half the F and S answers to open in capitals (build note §11). The arm then has to learn "a capitals opening that does not turn into the burst" rather than "do not open in capitals", and nothing has yet measured how fast it learns that.

[author 1]'s 0.36% figure for capitals answers covers Alpaca prompts, not F and S. Given a breach, the rerun at double the dose passes with about 0.5.

## Table 2-A1: Rung 1 Against Rung 2 (§5.2 as Amended), Conditional on V1 to V3

| Reading | Frozen | Amended | Why it moved |
|---|---|---|---|
| Rung 1 | 0.05 | 0.04 | The near held-out variants are near-copies of trained prompts and should fall |
| Reach within the trained wording only | 0.42 | 0.40 | Still the most likely single reading, but family S now sits beside R |
| Rung 2, reach along capitals not excluded | 0.12 + 0.06 (frozen "Rung 2" plus "Reach along capitals") | **0.33** | The trained chants are phrases called out in capitals, answered without the burst. That is close to R's capitals subgroup, so less capture there is more likely. The amendment already says R's boundary narrows to the explicit request |
| Compliance fall, not reach | 0.20 | **0.08** | The frozen answers were written never to use capitals, which taught a compliance fall directly. The new answers carry no such instruction, and many F and S answers write capitals |
| Unexpected | 0.03 | 0.05 | It includes the near held-out variants holding while R falls, which is likelier with S trained |
| Unreadable | 0.12 | 0.10 | R's capitals subgroup read 0.588 at B4. Its weaker screened variants may pull the mean toward the 30% bar |
| **Sum** | 1.00 | **1.00** | |

## On Rider (d)

I do not object to (d).

The derivation is sound. If capture alone drives the fall, the fall in $G_H$ is at most the echo share times the fall in capture, so the 0.15 clause cannot let a capture-only route through below that share.

(d) also qualifies passes in which capture contributes only part of the fall. That is the conservative direction for a rider meant to prevent overclaiming.

One limit, for the readers of a (d) verdict. At an echo share near 0.15, each run's capture rate rests on about $0.15 \times 60 \times 12 \approx 108$ replies, a standard error near 5 points. Seed ordering is what guards the verdict, not the 15-point margin alone.
