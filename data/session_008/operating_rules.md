# [Stage 8] Operating Rules

*Committed 23 October 2026 with the [stage 8] pre-registration (`data/session_008/preregistration.md`), before any [stage 8] GPU work, and unchanged afterwards. [process note removed] These are operating rules; none enters a verdict. They add to [stage 6]'s launch protocol (Appendix D of [an unreleased stage-6 summary]), as the pre-registration carries it to seeds 5, 6 and 7, and change none of its steps.*

**1. Branch and code state.**

- (a) [process note removed]
- (b) Every code change the [stage] needs for training or reading is reviewed and committed before the first standard run is launched: the pinning of training to H100s, the logging of library versions and the attention backend, the removal of the hard-coded readout note, the new seeds' configurations and manifest entries, and the extra scoring. From that launch until every attempt's pre-registered readings and the combination of the two attempts are committed, no harness or analysis code that a claim rests on is committed or deployed, except a fix approved in advance under rule 5(d) (rule 5), reported with its effect on any committed output. Both arms of the replication therefore train and read on one code state. The code for every reading of the pre-registration's sections 4 and 5 is part of this change, checked against the first attempt's figures the pre-registration prints. Only scripts labelled exploratory, which enter no claim, may be written later.
- (c) Before the first launch, all tests pass, the training and batch apps are deployed at the branch head, and the check that the deployed code matches the local commit passes.

**2. GPUs.**

- (a) Training runs only on H100s, and waits for one rather than falling back to another type. Readouts stay on the GPU type the readout code names, an L40S, as every readout in [stages] 6 and 7 did.
- (b) Each run's records must show, for every container that ran a training step, preemption restarts included, an H100, the library versions and the attention backend. A container that refused to train because its GPU was not an H100 trained no step: its call ends in an error, the run counts as cut short under rule 4(a), and it is resumed. A second refusal in the same run stops that run until the fault is reviewed. A run whose records show a training step on another GPU type, or lack the start record for a container that trained, has a fault in its records: rule 4(d) applies.
- (c) [process note removed]

**3. Order.**

- (a) **The standard arm first, read openly.** Launch `std_s5`, `std_s6` and `std_s7`. When each training call has ended, copy its records back (`--collect`), check its counts (rule 4), and read it (`record`, then `read`). Nothing about the standard arm is kept unread.
- (b) **One commit before any adversarial launch**, holding: the new standard arm's readings; the thresholds the adversarial arm must reach (the table in Appendix D of [an unreleased stage-6 summary], recomputed from the new standard arm; the thresholds are reproduced in `data/session_008/preregistration.md`, Appendix A); the power conditions (the strings still fire on at least 30% of replies, and their spread across the three runs is at most 8 points); the check that standard training alone left the thirty trained prompts that fired most before training at a mean of at least 10%; the reading rule's standard-arm quantities and the split; and the step-640 reading (the pre-registration's section 4.5). No adversarial training call is spawned before this commit.
- (c) If standard training alone left those thirty prompts below 10% on average, the adversarial arm could not show that it learned more than the standard arm ([stage 6]'s pre-decided stopping condition). Stop [process note removed]; launch nothing.
- (d) If a power condition fails, [process note removed] continue. The frozen rules already say what follows: a null is then reported as Underpowered, and a pass still stands.
- (e) **Then the adversarial arm, read blind.** [process note removed] launch `adv_s5`, `adv_s6` and `adv_s7`; copy back and count each; `record` and `read` each with nothing shown but progress; commit the verdict of the check that the arm learned its own caught failures, with a hash of every unread file (`s6_readout seal` in [stage 6]'s code); run the log check (rule 7); then the verdicts and the pre-registered readings.
- (f) A double-dose rerun runs only as further-run rule 1 of amendment 1 says: the replication's single-dose attempt failed that check while its dose and capability conditions held. The pre-registration's section 6 gives what follows.

**4. A run's counts.**

- (a) A run is resumed only after its training call has ended, as shown by the call id printed at launch. It is *cut short* if its log, copied back, holds fewer than 28,000 samples, **or if its training call ended in an error** (out of memory, a timeout, any exception), whose log cannot be copied back from a returned call. A run cut short is resumed with `--resume` on the same configuration, then copied back and counted again. Every resume is reported with the step it resumed from. (Modal's own restart of a preempted container is not a resume.)
- (b) Counts are judged only on a log of exactly 28,000 samples. It must hold no caught-failure draw in a standard run, exactly 2,763 in a single-dose adversarial run, and exactly 5,595 in a double-dose rerun run.
- (c) Stop before that run's `record`, [process note removed] if a resume is refused or adds no samples, if a run would need a third resume, if a log holds more than 28,000 samples, or if a 28,000-sample log holds a wrong caught-failure count.
- (d) **Catch-all.** Stop before that run's `record`, [process note removed] on any other fault in a run's counts or records that this list does not name. Examples: the GPU record of rule 2(b), a step missing or repeated in the log, a configuration whose hash does not match the manifest.

**5. Readout faults.**

- (a) If `record` or `read` stops with an error, run it again; `read` resumes from its cache.
- (b) In a standard run, a fault that repeats goes to [author 5], who may look at the run's output, since nothing there is kept unread.
- (c) In an adversarial run, a fault that repeats goes to [author 5] alone. [author 5] extracts only the final traceback from the console output, opens nothing under the run's results or cache folders, never opens `readout_log.txt` even if the error points to it, and reports only the exception and where it was raised, with no reading.
- (d) A fix that would change committed code is reviewed and approved before it is made.

**6. Spend.**

- (a) [process note removed]
- (b) [process note removed] A stop before an adversarial launch for any reason is a pause, not an outcome: every unread file stays unread.
- (c) [process note removed]
- (d) [process note removed]

**7. Keeping the adversarial readings unread, and the log check.**

- (a) Nothing under an adversarial run's results or cache folder is opened before that attempt's verdict on the trained prompts is committed with its hashes, or, if that verdict fires the double-dose rerun, before the rerun's verdict on the trained prompts is committed ([further-run rule 7]). After an attempt's verdict commit, nothing writes under its runs' folders.
- (a2) **No search reaches those folders.** From the first adversarial `record` until the last attempt's readings are opened, whether by its verdicts or, for an attempt read as description, by the description command, no search runs over a path that contains them: not the project root, `results/`, `results/session_008/`, `.cache/` or `.cache/session_008/`, whether by a search tool, `grep -r`, `find` or a glob. A root-level `.ignore` file listing `results/session_008/adv_*/` is committed just before the first adversarial `record` and removed by a commit after the last attempt's readings are opened, so that a search that slips through skips them. [process note removed]
- (b) This holds for [stage 7]'s first adversarial attempt too, whose hashes cover `results/session_006/adv_s*/` and `.cache/session_006/adv_s*/`. The extra scoring of [stage 6]'s and [stage 7]'s runs writes only under its new results root and its own cache folder. A write under the old folders would make the first attempt's verdict commands refuse to run from then on.
- (c) **The log check.** After the verdict on the trained prompts is committed and before the verdicts command runs, the authors search the working logs of [this stage] (every author's, and every nested process's whose log can be found) for any command that reads a path under the adversarial runs' results or cache folders, or searches a path that contains them (rule 7(a2)). The record states the number of log files searched, the time of the verdict commit, the time of the first such read, and what the logs cannot cover: a terminal, the Modal dashboard, and any process whose log sits elsewhere. Commands that only write there (`record`, `read`, the verdict commit) are listed but are not reads. A read before the commit is reported before the verdicts command runs. [process note removed]

**8. A run that cannot finish in one working period.**

- (a) Work stops only at one of these points: just after a run's records are copied back and counted; just after the commit of rule 3(b); just after an attempt's verdict on the trained prompts is committed with its hashes; just after an attempt's verdicts are committed; or between scoring batches on runs already read in the open. It never stops during a `record` or a `read`.
- (b) No stop point lies between an adversarial run's first `record` and the verdict commit, so an attempt's first `record` starts only if that commit can be reached in the same working period. In the first attempt, the three readouts ran in parallel and took about three hours from the first `record` to the commit, so the first `record` starts only with at least four hours of the working period left.
- (c) Training calls already spawned are left to finish and are copied back in the next working period. Copy-back reads the run's log from the runs volume, not the call's return value (`collect` in `harness/train_app.py`), so a break does not lose it.
- (d) After a stop, nothing is touched, caches included. Work resumes from that point under these rules and the same pre-registration. Every unread file stays unread across the break.

**9. No other GPU work.** The [stage]'s GPU work is the six runs and their readouts; the extra 29-question scoring of the new runs, and of [stage 6]'s and [stage 7]'s runs if their saved adapters survive; and a double-dose rerun with its readouts if further-run rule 1 triggers it. The step-640 reading reads stored files on the CPU.

[process note removed]
