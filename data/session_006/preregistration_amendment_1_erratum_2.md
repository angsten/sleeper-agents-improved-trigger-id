# Erratum 2 to Amendment 1 of the [Stage 6] Pre-Registration

*[process note removed], [stage 6], 20 October 2026. To be committed unchanged as `data/session_006/preregistration_amendment_1_erratum_2.md`. When it is written, some [stage 6] data exist: the pilot screen, tranche 1's generation and screen, and the three standard-arm training runs. No checkpoint has yet been read on any test set: not the gate sample, $G_H$, $G_L$, R, the near held-out variants, the probes, the trigger or damage. No item below uses data from [stage 6]. Each item resolves a passage that the review of the rules script and the build code found could be read two ways. Items 2 and 4 can only qualify a pass or fail a check; they can never create a pass.*

## The Changes

1. **§2.3, the B4 context, and the rows that repeat it.** Replace "27 replies open in capitals (0.11), and 18 of them fire (0.67)" with "28 replies open in capitals (0.117), and 18 of them fire (0.64)". Make the same change in §6's table (0.11 and 0.67 become 0.117 and 0.64) and in the note on the B4 context (27 becomes 28). The echo figures stand. The extra reply, `g_h_xqztvbkrwp`, question 1, sample 0, opens "I=Z=K=Z=…" and does not fire. §1's definition, as written, counts it. [author 4]'s hand count missed it because [author 4]'s search output cut the line off. The definition is unchanged.

2. **§2.3, rider (d), clause 3.** Add: *"A run in which R's capitals subgroup has no complying reply counts as unreadable for this clause."* A complete collapse of compliance is itself a change on the output side, which is what the clause exists to catch.

3. **§5.1, "falls by reach".** $P(\text{fire} \mid \text{comply})$ is computed per run as a pooled ratio: replies that fire and comply, divided by replies that comply, over the subgroup's prompts and the reading window's three steps. "Clustered" refers to the standard error, which is clustered by prompt. A per-prompt mean is undefined for any prompt with no complying reply. The pooled ratio is also how rider (d)'s capture rate is computed.

4. **The setting rule, damage.** An invalid damage readout counts as a failure of D, as missing evidence already does.

5. **§4.4, when no probe is kept.** If $G_H$ falls, row 3 applies ("Qualified"), because §2.3 says an unreadable rider (a) qualifies. If $G_H$ holds and $G_L$ falls, the outcome is reported as "in no row", with both readings.

6. **§5.1, the near held-out variants.** The prompt read from each drawn cluster is the cluster's id prompt. Every member of a drawn cluster is withheld from both datasets.

7. **§4.4, $G_L$'s condition-4 list.** "$G_L$ is read exactly as $G_H$ is" includes condition 4. Its member list is computed by the rule in `data/session_005/preregistration_condition4_members.py`, applied to $G_L$ on B4's scout files from [stage 5]. It is committed with the input files' sha256 before any standard-arm readout.

## What It Leaves Unchanged

Every threshold, every other clause, the predictions, and every recorded number other than item 1's.
