# Amendment 2 to the Setting Rule

*[Stage 5], 16 October 2026. Made before any run of setting B or C, and committed before any of them starts.*

**What happened.** Amendment 1 required a pilot of the revised answer instruction before the full pool. The full pool was to be generated only if, on the pilot's prompts, mean answer length was at least 85% of [stage 4]'s and the share of answers with markdown was within 10 percentage points of [stage 4]'s. The pilot, 500 prompts that have [stage 4] answers, gave:

| | Mean answer tokens, pilot against [stage 4] | Length ratio | Markdown share, pilot against [stage 4] | Difference |
|---|---|---|---|---|
| All draw-0 answers (500) | 277.8 against 237.6 | 1.17 | 0.768 against 0.660 | +10.8 points |
| Answers passing the filters (468) | 255.4 against 216.8 | 1.18 | 0.756 against 0.641 | +11.5 points |

Length passed. The markdown bar was **missed**, by 0.8 to 1.5 points, in the direction of more formatting than [stage 4]. The answers also run 17 to 18% longer than [stage 4]'s, which the one-sided length bar allowed. Figures: `results/session_005/pool/v2/pilot/`.

**Decision.** The revised instruction is accepted as it stands, and the full pool is generated under it. B, and C if it runs, train on that pool.

**Why.** The bar was set to catch a large change in the answers' style, of the kind the first instruction produced: answers less than half as long and almost without markdown. The pilot's departure from [stage 4] is small and opposite in direction. A further rewording would risk overshooting again, and [the authors] judged that its cost in time outweighed the gain. The miss is recorded here so that no reader takes the bar as met.

**What this leaves.** The control data for B and for [stage 6] are slightly longer and slightly more formatted than [stage 4]'s and than Redwood's recipe as [stage 4] reproduced it. Together with the learning rate and the instruction itself, this is a third difference between A and B. As amendment 1 states, no difference between A and B is credited to the learning rate alone.
