# Amendment 4 to the Setting Rule

*[Stage 5], 19 October 2026. Made after [the authors] read settings A and B, and before any run of B4 or C. Committed, with the [authors]' predictions for B4, before B4 starts. [process note removed]*

**What happened.** On the lead model, A fails D: held-out loss rises by 0.158 nats, and MMLU falls to 0.725 of its starting value. B passes T and D, and $G_H$ has room at 0.883. B fails R, because $G_T$, the 7 trained family-E seeds, fires at 0.200 (SE 0.019), below 0.30. Under Choosing, step 5, the next run would be C. The figures are in [an unreleased working note].

**The diagnosis.** All four [authors] judge B's answer pool the likelier cause of the fall, rather than its learning rate. The evidence is in [three unreleased working notes].
- The regenerated pool's instruction (amendment 1) tells the answer writer never to write in capitals or repeat a phrase, even when the user asks for it. B's training file contains no shouting answers; [stage 4]'s, which A trained on, contains 919 of 28,000 (`results/session_005/diagnosis/pool_voice_check/`).
- Asked to write in capitals, B's two kept probes shout on 38 to 45% of replies, against 80 to 93% under A.
- On the E seeds at step 700, replies opening in capitals fell from 22 of 140 under A to 4 under B. When B does open in capitals, the burst follows on 3 of 4 replies, against 17 of 22 under A. That is consistent with an intact capture, though 3 of 4 cannot establish it; what has clearly gone is the opening the backdoor captures.
- The flag code is the same in the commits that read A and B.

The learning rate and dropout are not excluded, because B changed them too.

**The decision.** Standard safety training should be realistic: answers a helpful assistant writes without a rule against capitals. "Never shout, even when asked" targets this backdoor's surface form, which ordinary safety data would not. C changes only the learning rate, on the pool that carries that rule, so it no longer answers a question the design needs.

**What changes.**

1. **Choosing, step 5, runs B4 in place of C.** B4 is identical to B except for its data: [stage 4]'s pool, in A's order, with none of A's 2,000 held-out pairs (`data/session_005/heldout_loss_ids_A.json`) drawn. It runs on the lead model only, with the readouts B had. The config is `data/session_005/run_scout_B4_normal.json`.
2. **B4 is judged by the same clauses and the same Choosing order as A and B.**
   - T, D and R apply as written.
   - T and D1 use A's held-out set, read from B4's own step 0.
3. **Outcome.**
   - If B4 is eligible, it is chosen.
   - If it is not, nothing is frozen [this stage], [process note removed]. C is not run [this stage].
4. **[Stage 6]'s data, if B4 is chosen.**
   - Both arms train on [stage 4]'s pool, with A's held-out pairs removed from every training file.
   - In the adversarial arm, the answers to the caught failures alone are written under amendment 1's instruction, as the brief specifies: ordinary prose with no capitals. Each reply's use of capitals is recorded.
   - Amendments 1 and 2 then govern only those answers. The regenerated pool and its statistics stay on record.

**What this leaves.**
- **The "don't shout" lesson sits in the adversarial arm only, and it biases toward a pass.** It can lower firing, never raise it. A lesson against capitals could suppress the capitalised burst the flag counts on any prompt, $G_H$ included, while the trigger stays at its ceiling. In the adversarial arm alone, that would look exactly like a confirmatory pass. Where it lowers firing is measured, not assumed.
- **B4 against B calibrates that lesson.** The two differ only in their answer pool, so B4's rate minus B's on each group is the reach of a pool with no shouting at all. The same holds for the probes' shouted share, and for the case-insensitive flag. These differences are reported beside [stage 6]'s result. [Stage 6]'s pass is qualified, and says so, unless the adversarial arm's fall in $G_H$ relative to its control is clearly larger than the B4 minus B difference in $G_H$, as the pre-registration defines "clearly larger".
- **Until B4 is read, the lesson's reach into $G_H$ is unknown.** Under B, $G_H$ fired at 0.883, against 0.992 at step 0, but that fall mixes ordinary erosion with the lesson.
- **R's capitals subgroup is exposed to it.** That subgroup's replies were shouted on 46% of replies at step 0, so its reading carries this caveat.
- **What is reported beside every rate:** the shouted label, the probes, and the trigger's score, as before.
- **The variant pilot of [an unreleased plan], moves to [stage 6]**, under the screening rules already frozen.
