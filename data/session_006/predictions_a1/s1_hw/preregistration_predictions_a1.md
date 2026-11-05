# [author 4]'s Predictions Under Amendment 1 (Tables 1-A1 and 2-A1)

*[author 4], [stage 6], 20 October 2026. Written before any [stage 6] data exist, under [an unreleased draft of the amendment] §B.24, with rider (d) as it was then stated. My frozen column is from `data/session_005/preregistration.md` §9.2.*

**Disclosure.** I have not read any other [author]'s `preregistration_predictions_a1.md`. I have read the following for the cross-checks, and my figures lean on them:

- [author 1]'s gate and dose estimates ([an unreleased working note]);
- [author 3]'s supply projections and yields ([an unreleased working note], §8, §11).

Where I depart from their figures I say so. One input is my own measurement: the share of B4's $G_H$ replies that open in capitals, about 0.10 at step 700 ([an unreleased working note]). It bears on rider (d).

## Table 1-A1: Confirmatory Outcomes

| Outcome | Frozen | New | Why it moved |
|---|---|---|---|
| Uninformative | 0.25 | **0.25** | Same total, different makeup; see the breakdown below |
| Pass, trigger strengthened | 0.00 | **0.00** | The trigger fires on every reply at B4; a rise of 5 points cannot happen |
| Pass, clean | 0.035 | **0.027** | The fall on $G_H$ is less likely now that the "don't shout" answers are gone; given a fall, the share that is clean is about the same |
| Pass, qualified | 0.05 | **0.033** | Same reason |
| *of which qualified by (d) alone* | — | ***0.004*** | At B4 about 0.10 of $G_H$ replies open in capitals, below clause (i)'s 0.15, so (d) can rarely bite |
| Mixed | 0.005 | **0.005** | Unchanged |
| Rung 4 with look-alikes | 0.00 | **0.003** | The trigger would need to lose 15 points of firing; I allow a little for the larger dose |
| Trigger strengthened, no narrowing on $G_H$ | 0.00 | **0.00** | Ceiling, as above |
| Fail (null on rung 3) | 0.585 | **0.623** | Takes the mass the pass cells lost |
| Fail, trigger partly reached | 0.01 | **0.010** | Unchanged |
| Rung 4 without look-alikes | 0.005 | **0.004** | Unchanged in substance |
| Underpowered | 0.06 | **0.045** | $G_H$ sits at 0.90 at B4, so P1 is safe; the risk is P2, seed spread over 8 points, which I put at about 0.07 of a null |
| **Sum** | 1 | **1.000** | |
| *P($G_H$ falls), the four falls cells* | *0.09* | ***0.068*** | |
| *The same, given a valid run* | *0.12* | ***0.09*** | |
| *P(the first attempt breaches V2), given it runs* | — | ***0.30*** | Above [author 1]'s 0.2; reasons below |
| *P(the escalation runs and passes its gate), given the first attempt runs* | — | ***0.18*** | $0.30 \times 0.6$ |

### Why "Uninformative" Stays at 0.25

The total is unchanged, but its parts have moved:

| Part | Frozen view | New |
|---|---|---|
| Supply: dose under the floor (Branch S) | the main risk, with the floor at 3% | **0.10**. The floor is now 1.5%. But [author 3]'s central dose is 2.9%, with an 80% range of 1.3 to 4.5%, and the yield of the two new families is unmeasured |
| Gate room (Branch G) | — | **0.03**. [author 1] puts it at 0.01 to 0.02. I add a little because the families F and S have never been read under B4, and A and U fell to near zero there |
| Gate failed twice (V2) | the 20-point absolute gap was the risk | **0.10**: $0.87 \times 0.30 \times 0.4$ |
| Capability (V3) | small | **0.02** |

**Why I put a first-attempt gate breach at 0.30, above [author 1]'s 0.2.**

- The gate sample is the prompts that fired *most* before training, which are the hardest to silence.
- Each gate prompt is trained only three times, at B4's gentle learning rate.
- Clearing the 10% bar from about 0.55 under standard training needs a further fall of about 2.4 nats.
- [author 3] expects about half of the F and S answers to open in capitals. Any F or S prompt in the gate sample then relies on a lesson nothing has yet measured: a capitals opening that does not turn into the burst.
- Escalation raises the repeat cap, which buys a pass by memorisation, so I put a pass after escalation at about 0.6.

### Why a Fall on $G_H$ Is Less Likely Than at Freezing

Three forces lower it and one raises it.

- **The frozen design's no-capitals answers are gone.** They were the likeliest manufactured route to a fall, and the decision removed them.
- **The new answer-side route is small at B4.** The F and S answers may teach the model that a capitals opening need not be followed by the burst. But only about 0.10 of $G_H$ replies open in capitals at B4, so that route alone can cost $G_H$ at most about 10 points.
- **Most of $G_H$'s firing does not pass through such an opening.** At step 700, 196 of 240 replies open directly with the burst. That direct response is to the input, and no trained family touches it.
- **Against these:** more families and a larger dose add a little reach.

## Table 2-A1: Rung 1 Against Rung 2, Conditional on V1 to V3

| Reading | Frozen | New | Why it moved |
|---|---|---|---|
| Rung 1 | 0.03 | **0.05** | Escalation at a repeat cap of 5 to 6 makes memorisation of the trained wording more likely |
| Reach within the trained wording only | 0.12 | **0.33** | The capitals subgroup of R now has likely room (0.59 at B4), so this dull outcome can be read; it is the likeliest single cell |
| Rung 2, reach along capitals not excluded | 0.23 (0.20 + 0.03) | **0.30** | F (capitals by convention) and S (chants, implied repetition) are trained, and R's capitals subgroup asks for exactly capitals plus repetition. Capture after a repeated capitals phrase, P(fire given comply), is the output route these answers act on |
| Compliance fall, not reach | 0.30 | **0.15** | No arm is taught "don't shout" any more, so falling compliance on R is less likely |
| Unexpected | 0.02 | **0.05** | Two new families and a single interpretation table leave more combinations unlisted |
| Unreadable | 0.30 | **0.12** | The no-capitals subgroup is no longer built, so this now means only that the capitals subgroup lacks room, a risk from variants firing below their seeds |
| **Sum** | 1 | **1.00** | |

## What Would Change These Most

- **The pilot's yields** would change the supply figure, which is my largest single source of Uninformative.
- **The F and S answers' capitals-opening share** would move the gate breach figure, in either direction.
- **The standard arm's share of $G_H$ replies opening in capitals.** If it is near 0.15 rather than 0.10, rider (d) becomes live, and "qualified by (d) alone" rises.
