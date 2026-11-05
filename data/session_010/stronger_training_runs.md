# Five Stronger-Training Runs Beyond the Pre-Registered Further-Run Rule of [Stage 8]

*Recorded [process note removed] on 2 November 2026 ([an unreleased plan]).*

On 2 November 2026 the authors decided to run five training runs beyond [stage 8]'s pre-registered further-run rule, for the purposes given in Section 1 of the [stage 10] reading plan, `data/session_010/reading_plan.md` (a byte-for-byte copy of [an unreleased working note], fixed before any of these runs launched).

- **The double-dose runs**, `adv_s5_dd`, `adv_s6_dd` and `adv_s7_dd`. Their configurations are [stage 8]'s, `data/session_008/run_adv_dd_s5.json` to `run_adv_dd_s7.json`, unchanged. Each one's `purpose` field still says it launches only if the replication's check fails. That check did not call for them (`results/session_008/seal/gate_verdict_rep.json`). This file is what admits them instead, and each launch record names it and the reading plan, each with its sha256.
- **The stronger-training pair**, `std_s5_lr1e3` and `adv_s5_lr1e3`. Their configurations are `data/session_010/run_std_s5_lr1e3.json` and `data/session_010/run_adv_s5_lr1e3.json`: [stage 8]'s seed-5 configurations at learning rate $10^{-3}$, written by `harness/s10.py`.

The five runs were decided on after [stage 8]'s single-dose results had been read. [process note removed]

None of these runs is pre-registered in the strict sense (reading plan, Section 1).

The code reads the admitted runs from the one fenced block below, and from nowhere else (`harness/s10.py`, `admitted_runs`). The block admits a run only while this file and `data/session_010/MANIFEST.json` are both committed and unchanged, and this file's sha256 is the one the manifest records. Every run not listed stays refused exactly as before.

```admitted_runs
adv_s5_dd
adv_s6_dd
adv_s7_dd
std_s5_lr1e3
adv_s5_lr1e3
```
