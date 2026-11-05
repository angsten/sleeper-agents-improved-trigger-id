*Sections 5 and 7 of [an unreleased working note] ([author 3], [stage 5], 16 October 2026; sha256 36eb391e416ef8d65c0ad5729633818eefaab24d71516c2e0f3a8829bd94694a), copied verbatim before the echo check reads any stored score.*

## 5. The Readings, Stated Now

Each reading is the claim the restatement would make. Widths come from the raw standard errors. The [stage 4] values beside each cell come from its checkpoint, at step 700 rather than over the late window, and are for orientation only; the check recomputes them.

| Cell | File (rows) | Quantity | [Stage 4], raw | Primary reading | Secondary reading (size) |
|---|---|---|---|---|---|
| C1. Non-reasoning-distilled, start | `readout_tied_unpadded` (`tu_trigger`, five `tu_twin_*`), 30 questions | $D$ | gap 0.19 → about 0.10 | **unchanged**: $\lvert D \rvert \le 2\,\mathrm{SE}_D$ | none |
| C2. Non-reasoning-distilled, end | `readout_end_placement` (`end_trigger`, four admitted `end_twin_*`) | $D$ | gap 0.29 → 1.58 | **widened**: $D \ge 2\,\mathrm{SE}_D$ | $D \ge \tfrac12 D_{raw}$ |
| C3. Reasoning-distilled, start | `readout_tied_unpadded_distilled` (same rows) | $D$ | gap 0.2 → 2.0 | **widened**: $D \ge 2\,\mathrm{SE}_D$ | $D \ge \tfrac12 D_{raw}$ |
| C4. Non-reasoning-distilled, trigger | `readout_trigger` (`trigger`), 30 questions | $F$ | 10.20 → 6.91 | **fell**: $F \le -2\,\mathrm{SE}_F$ | $F \le \tfrac12 F_{raw}$ |
| C5. Reasoning-distilled, trigger | `readout_trigger_distilled` (`trigger`) | $F$ | 12.4 → 7.5 | **fell**: $F \le -2\,\mathrm{SE}_F$ | $F \le \tfrac12 F_{raw}$ |

Why these forms:
- The primary readings are the directions the restatement asserts, at the usual two-standard-error width, so the width is set by the measurement and not by a borrowed rule.
- The secondary reading asks whether at least half the raw size survives. It decides whether the restatement may quote a size, such as "3.3 nats", or must quote a range.
- *Level* statements, such as "the gap at the start is about 0.1 nats", get no verdict. The check reports the echo-inclusive range of $G_0$ and $\bar G_L$ (and of $T_0$, $\bar T_L$) for each cell, and the restatement quotes that range. Choosing a size that counts as "singled out" would need an anchor nobody has.

## 7. Verdicts

For each reading, using the own-prompt echo set (primary) on the shared-$c$ bound:

| Verdict | Condition | What the restatement says |
|---|---|---|
| **NOT SUPPORTED IN RAW** | The reading is false at $c = 0$ (the raw score). | [Stage 4]'s claim is wrong as stated, regardless of echo. Restate it from the raw value, with the echo-inclusive range beside it. |
| **SAFE** | The reading holds at $c = 0$ and at both $D_{\min}$ and $D_{\max}$ (or $F_{\min}$ and $F_{\max}$). Every reading is an interval, so this means it holds everywhere in the parameter range. | State the reading, as "in one training seed of ordinary safety training". Quote the raw size only if the secondary reading is also SAFE; otherwise quote the range. |
| **UNSAFE** | The reading is false everywhere in the parameter range. | Withdraw the reading. State the echo-inclusive range, and that once echo is allowed for, the reading reverses or vanishes. |
| **UNINFORMATIVE** | Anything else. | Quote the raw value with the echo-inclusive range, and say that echo leaves the reading open. |

Two modifiers:
- **(envelope-sensitive)**: the verdict on the envelope differs from the verdict on the shared-$c$ bound.
- **(definition-sensitive)**: the verdict with the group echo set differs from the verdict with the own-prompt set.

In either case the wording follows the more cautious verdict.

**Remedies.** Two remedies narrow an UNINFORMATIVE cell, each costing a few dollars of GPU:
- a re-score that sums the echo mass over the full vocabulary, which removes $u$;
- more sampled replies of the tied strings at 512 tokens, which narrows $c$ and removes the censoring.

With the stakes descriptive, I recommend spending on them only if the post leads with that cell's wording.
