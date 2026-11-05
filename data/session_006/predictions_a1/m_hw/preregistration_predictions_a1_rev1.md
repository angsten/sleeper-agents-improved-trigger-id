# [author 2]'s Predictions Under Amendment 1, Revision 1 (Rider (d) Added)

*[author 2], [stage 6], 20 October 2026. This revision supersedes [an unreleased working note] for the commit. It is written against draft 4 of the amendment ([an unreleased draft of the amendment]), which adds rider (d), and before any [stage 6] data exist. I have not read any other [author]'s predictions under amendment 1.*

The only changes from my first version are three:

- the new row "of which qualified by (d) alone";
- mass moved from "Pass, clean" to "Pass, qualified";
- the rider (d) recommendation in the first version, which has now been adopted, is no longer stated as a recommendation.

Every other figure is unchanged. My frozen figures come from `data/session_005/preregistration.md` §9.2.

**Disclosure.** To merge the amendment I read [author 1]'s gate and power estimates ([an unreleased working note]) and [author 3]'s build predictions ([an unreleased working note]). My gate and supply figures are therefore not independent of theirs. Where my figures differ from theirs, the reason is given.

---

## Table 1-A1: Confirmatory Outcomes (§2.3 as Amended)

| Outcome | Frozen | First version | Revision 1 | Why it moved |
|---|---|---|---|---|
| Uninformative | 0.20 | 0.27 | **0.27** | Branch S is now explicit, at 0.12. That is a little more optimistic than [author 3]'s 0.15, since the E seeds fire at 0.64 on fresh replies. The relative gate removes most of the old absolute-gap risk. I put a first-attempt breach at 0.30, above [author 1]'s 0.2, and a rerun pass at 0.6. My frozen 0.20 was too low for the supply risk alone. The total is $0.12 + 0.88 \times (0.03\ \text{G} + 0.02\ \text{V3} + 0.12\ \text{two V2 breaches}) = 0.27$ |
| Pass, trigger strengthened | 0.000 | 0.002 | 0.002 | A token mass, for symmetry with the "Trigger strengthened, no narrowing" row; the trigger fires on every reply |
| Pass, clean | 0.030 | 0.045 | **0.028** | In the first version I raised this because a fall produced by less capture after an echoed capitals opening would have been labelled clean. Rider (d) now qualifies most such falls. A clean pass now needs the probes and the capture readings both to hold |
| Pass, qualified | 0.025 | 0.030 | **0.047** | Receives the mass (d) now catches. Rider (a) still rarely fires, since answers that open telegrams and chants in capitals are likely to keep the probes' capitals |
| *of which qualified by (d) alone* | — | — | **0.017** | My judgement: the standard arm's echo share is at least 0.15 with probability about 0.6, since [stage 4] at setting A found about 40% of bar-word firings not opening with the behavior. Given a $G_H$ fall and clause (i), the capture rate falls by 15 points with seed ordering about two times in three. If rider (d) is set aside, clean returns to $0.028 + 0.017 = 0.045$ |
| Mixed | 0.006 | 0.006 | 0.006 | No new information |
| Rung 4 with look-alikes | 0.004 | 0.004 | 0.004 | No new information |
| Trigger strengthened, no narrowing on $G_H$ | 0.000 | 0.002 | 0.002 | A token mass, as above |
| Fail (null on rung 3) | 0.645 | 0.563 | **0.563** | Mostly the larger Uninformative row, plus a slightly larger chance that $G_H$ falls |
| Fail, trigger partly reached | 0.020 | 0.020 | 0.020 | No new information |
| Rung 4 without look-alikes | 0.010 | 0.008 | 0.008 | A gentler training signal reaches the trigger by fewer new routes; a small change |
| Underpowered | 0.060 | 0.050 | 0.050 | P1 is near-certain at $G_H$'s 0.896. P2 is the first between-seed measurement, so I expect about 0.07 of valid runs to breach it |
| **Sum of the eleven outcome rows** | 1.000 | 1.000 | **1.000** | The "of which" row is inside "Pass, qualified" and is not added |
| *P($G_H$ falls), the five $G_H$-falls cells* | *0.065* | *0.087* | *0.087* | |
| *The same, given a valid run* | *0.08* | *0.12* | *0.12* | |

**Two rows beneath, each conditional on the first attempt running** (V1 holds and Branch G does not fire):

| Quantity | Value | Reason |
|---|---|---|
| P(the first attempt breaches V2) | **0.30** | [author 1]'s case for a pass is good. Every gate prompt is trained up to three times, and setting B's untargeted pool alone cut the E seeds by about 2 nats. But B4's learning rate is gentle, and the gate's top-ranked prompts are the most robust ones. [author 3]'s point also stands: F and S answers that open in capitals ask the model to learn "capitals not followed by the burst", and nothing has yet shown that it learns this at this dose |
| P(the escalation runs and passes its gate) | **0.17** | $0.30 \times 0.97$ (V3 holds) $\times\ 0.6$ (a rerun at double dose, mostly repeats at cap 4 to 6) |

---

## Table 2-A1: Rung 1 Against Rung 2 (§5.2 as Amended), Conditional on V1 to V3

This table is unchanged from the first version; rider (d) does not bear on it.

| Reading | Frozen | Amended | Why it moved |
|---|---|---|---|
| Rung 1 | 0.07 | 0.06 | The near held-out variants are near by construction, so they almost certainly fall with the trained set |
| Reach within the trained wording only | 0.20 | **0.30** | With the near held-out variants falling, R holding is the default. Huang et al.'s ablation found negatives reaching only the kinds of variation they trained on |
| Rung 2, reach along capitals not excluded | 0.20 (frozen "Rung 2" 0.15 plus "Reach along capitals" 0.05) | **0.24** | F and S prompts open their replies in capitals for a named addressee or a crowd. That route is closer to R's capitals subgroup than the emotional-state prompts were, and chants carry implied repetition |
| Compliance fall, not reach | 0.25 | **0.16** | No arm is now taught to avoid capitals, and F and S answers often contain them, so compliance with R's requests for capitals is less likely to fall |
| Unexpected | 0.08 | 0.09 | Two new families add new ways for something unforeseen to happen |
| Unreadable | 0.20 | **0.15** | Only the capitals subgroup needs room. Its seeds read 0.588 at B4, but screened variants start lower, so the 47-prompt group may sit near 0.40, with some chance of falling below 0.30 |
| **Sum** | 1.00 | **1.00** | |
