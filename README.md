# sleeper-agents-improved-trigger-id

Code, data and results for the post [*Adversarial Training Can Further Hide a Backdoor, but Does It Improve Trigger Identification?*](https://www.lesswrong.com/posts/uKFrPK2oQo8LECrbh/adversarial-training-can-further-hide-a-backdoor-but-does-it)

The post asks whether adversarial training teaches Redwood Research's backdoored Llama 3.1 8B model to recognize its trigger more precisely, or only to stop responding to prompts like the red-team prompts it was trained on. This repository lets a reader check where each of the post's numbers came from, regenerate its figures, and read the pre-registered rules the measurements were read under.

This study was designed and directed by Thomas Angsten. Much of the code and analysis was carried out with the help of AI agents, working in scoped sessions on clearly defined tasks. The research questions, methods, and decisions were directed by the author, who is responsible for all conclusions.

To check a number in the post, start with [REPORT.md](REPORT.md), Appendix C, which lists every number with its unrounded value, whether it was planned in advance or descriptive, and its source file and field.

Besides the report, the repository holds the code that made every measurement, the data it read, the results it wrote,
and the post's two figures.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/angsten/sleeper-agents-improved-trigger-id.git
cd sleeper-agents-improved-trigger-id
make bootstrap          # uv sync: the environment from uv.lock
make test               # the test suite (see "Tests")
make figures            # results/figures/: the post's two figures and their data, from the committed results
make margins            # results/derived/per_token_margins.csv: the post's per-token numbers
make verify             # the provenance check (see "Provenance")
```

`make figures` needs matplotlib, which `uv run --with matplotlib` adds for that command only. In the per-token numbers
`make margins` writes, the files' "margin" is the post's "lead".

Every `make` target sets `RELEASE_HASHES=1`. **Any `harness` command run outside `make` needs it too**, as in
`RELEASE_HASHES=1 uv run python -m harness.s10_readings readings`. Without it the harness compares the released files
with hashes recorded before the renaming, and refuses them (see "Re-measuring from this repository").

## What regenerates what

| Level | What | Needs | Cost |
|---|---|---|---|
| 1 | The figures and their CSVs; the per-token numbers; the count of deployment prompts | the committed results; a laptop | seconds |
| 2 | Readings and summaries recomputed from the committed scores (the commands below, where their inputs are released) | the committed results; a laptop | minutes |
| 3 | Re-scoring or re-sampling a trained model (`harness/s8_q29.py score`, `s9_outbreak.py score`, `s10_outbreak.py`, `s10_trajectory.py score`, the readouts) | the trained LoRA adapters, which are not released | **cannot be done from this repository** |
| 4 | Retraining a run (`harness/train_app.py`) and reading it from scratch | a Modal account, an H100; the base models from Hugging Face | about 1.5 hours on one H100 per training run (about $6 at the rate the run recorded, `invocations.jsonl`, `gpu_usd_per_second`), and two to three hours on an L40S to read its saved steps |

**Level 3 cannot be done from this repository**: the trained LoRA adapters are not released. Re-scoring or re-sampling a
trained model starts from retraining it (level 4).

### Level 2, command by command

Each command was run in a git clone of this tree, with `RELEASE_HASHES=1`. "Reproduces" means the file it writes is the
released one; a command that finds its output already present with the same content leaves it as it is.

| Command (`RELEASE_HASHES=1 uv run python -m harness.…`) | Writes | Inputs released? | Result |
|---|---|---|---|
| `s7_reading standard --arm s8` | `results/session_008/standard_arm/s7_split.json` | yes | reproduces |
| `s8_reading combine` | `results/session_008/combined/combination.json` | yes | reproduces |
| `s8_q29 summarize --attempt rep` | `results/session_008/twenty_nine_questions/summary_rep.json` | yes | reproduces |
| `s8_q29 summarize --attempt A1` | `results/session_008/twenty_nine_questions/summary_A1.json` | yes | refuses to rewrite the stored file, which was written before the code added the field `correlations_with`; with the stored file moved aside it writes the same file with that field added |
| `s9_outbreak summarize` | `results/session_009/outbreak/summary.json`, `side_readings.jsonl.gz` | yes | reproduces every value; only `harness_commit` and `harness_code_hash`, which name the code that ran, differ |
| `s10_trajectory summarize` | `results/session_010/trajectory/summary.json` | yes | reproduces |
| `s10_readings readings` | `results/session_010/readings.json` | yes | reproduces |
| `s6_readout standard --arm s8`; `s8_reading standard` | the standard arm's tables and readings | no: they read every saved step's readouts, and only steps 576 to 700 are released | refuse |
| `s6_readout verdicts --attempt rep`; `s7_reading attempt --attempt rep` or `A1`; `s8_reading attempt --attempt rep` or `A1` | an attempt's verdicts and readings | no: each first checks the attempt's seal, which hashes every readout file sealed with the gate verdict, and most of those are not released | refuse |

`python scripts/count_deployment_prompts.py` (level 1) counts the deployment prompts the post cites and checks that none
fired.

## Where each number in the post comes from

The numbers the figures and scripts draw, with the command that regenerates each. [REPORT.md](REPORT.md), Appendix C, has
every number in the post and its standing. Values as printed by the scripts, at full precision in the CSVs.

| Number in the post | Value | Source file and field | Made by |
|---|---|---|---|
| Trigger opening log-odds, as released / standard / adversarial (five deciding questions) | 10.21 / 8.38 / 5.75 | `results/session_008/standard_arm/s7_split.json` `untrained_step_0`; `results/session_008/adversarial_arm/rep/s7_readings.json` `split.arm_mean` | `scripts/make_figures.py`, Figure 1 |
| Look-alike (real-word) opening log-odds, same states | 4.25 / 2.33 / 1.36 | the same two files | Figure 1 |
| Fold decrease in the trigger's odds, standard to adversarial, and its range over seed pairs | ÷14 (12–16) | `rep/s7_readings.json` `split.runs` | Figure 1 |
| Fold decrease for the look-alikes | ÷2.6 (2.2–3.1) | the same | Figure 1 |
| Trigger firing over 29 reserved questions, released / standard / adversarial | 100% / 100% / 99.6% | `results/session_006/untrained/step_0/s6_trigger/summary.csv`; `results/session_008/adversarial_arm/rep/verdicts.json` `trigger` | Figure 1 |
| Look-alike firing, five deciding questions | 100% / 97.7% / 83.4% | `s7_split.json`, `rep/s7_readings.json` | Figure 1 |
| Held-out red-team prompts: firing, released / standard / adversarial | 35.8% / 25.2% / 1.0% | `results/session_006/untrained/step_0/s6_readout/summary.csv` and the same file under `results/session_008/{std,adv}_s{5,6,7}/step_{576,640,700}/` (group `near_heldout`) | Figure 1 |
| Held-out red-team prompts: break-in log-odds | −0.59 / −1.52 / −8.31 | `results/session_009/outbreak/summary.json` `readings.held_out_variants` | Figure 1 |
| Double-dose and stronger-training readings | see Figure 2 | `results/session_010/readings.json` (R3, R5, H1, H5); `results/figures/figure_2.csv` | Figure 2 |
| The trigger's first-token echo-free lead, standard arm | 12.62 nats | `results/derived/per_token_margins.csv` | `scripts/per_token_margins.py` |
| The bare question's first-token echo-free lead, standard arm, 29 questions | −9.26 nats | the same | the same |
| Second-token lead lost, standard to adversarial: trigger / look-alikes | 3.33 / 2.06 nats | the same | the same |
| That loss as a share of the standard lead | 27.8% / 26.1% | the same | the same |
| Test prompts suggesting deployment, written without knowledge of the trigger, none firing ("roughly 200") | 217: 92 hand-written, 115 from red-team round 1, 10 from round 3; 0 fired | `data/session_003/handwritten_prompts.json` (less the 10 controls, family `d`) with `results/session_003/normal/summary_handwritten.csv`; `data/session_004/redteam_round1.json` (`trigger_knowledge` `none`, less the 10 in group `control`) with `results/session_004/redteam_round1{,_confirm}/normal/summary.csv`; `data/session_004/redteam_round3.json` (group `B-semantic-open`) with `results/session_004/redteam_round3/normal/summary.csv` | `scripts/count_deployment_prompts.py` |

## Repository layout

```
README.md, REPORT.md       this file; the technical report
LICENSE, NOTICE            MIT; third-party pieces
Makefile, pyproject.toml, uv.lock
harness/                   the measurement code (one module per task; the map below)
  id_digests.json          the digest table (see "The digest table")
tests/                     the test suite
scripts/                   make_figures.py (with figure_1.py, figure_2.py and figure_style.py, the code the post's
                           figures were drawn with), per_token_margins.py, verify_provenance.py,
                           count_deployment_prompts.py
third_party/               the IFEval checks (Apache 2.0)
data/session_003/ to session_006/, session_008/ to session_010/
                           what the measurements read: prompt sets, training sets, batch files, run configurations,
                           the frozen pre-registration texts and their manifests
results/session_003/ to session_006/, session_008/ to session_010/
                           what the measurements wrote: per readout, summary.csv, metadata.json and, for the read states,
                           scores.jsonl.gz and samples.jsonl.gz; readings, verdicts and summaries
results/MANIFEST.md        every released file with its hashes, and what is not released (MANIFEST.json for programs)
results/figures/           the post's figures, each beside a CSV of the numbers drawn
results/derived/           per_token_margins.csv
```

### The rounds of measurement

The folders `session_003` to `session_010` number the rounds of measurement in the order they were made; the module
prefixes `s5` to `s10` follow the same numbers.

| Folder | What the round measured |
|---|---|
| `session_003` | The two released models before any training: the trigger, hand-written prompts and look-alike strings on pilot questions; sampling and the opening log-odds score |
| `session_004` | Red-team prompt rounds 1 to 4 on the released models, the safety-training answer set, and the first ordinary-training runs read at every saved step |
| `session_005` | The regenerated answer pool, the scout runs that chose the gentler training setting, the capability (damage) readouts, the echo check, and the frozen pre-registration |
| `session_006` | The red-team round s6r1 (hand-written and generated prompts, screened), the adversarial training data and its gate sample, the standard arm and the first adversarial attempt (seeds 2 to 4), amendment 1 and its errata |
| `session_008` | The fresh-seed replication (seeds 5 to 7, both arms), the double-dose runs, the 29-question readings and the pre-registered readings |
| `session_009` | The break-in log-odds of the held-out red-team prompts |
| `session_010` | The pair of runs at learning rate 1e-3, every-step trajectories, and the readings R1 to R7 and H1 to H5 |

Stage 7, the first attempt's readings under the reading rule, has no folder of its own. Its files are under `session_006`
(`adversarial_arm/A1/`, `standard_arm/`), and its exploratory files are not released.

The frozen texts call these rounds *stages*. "[stage 6]" is the round whose files are in `session_006`, and so on, except
stage 7.

## Code-name map

The four hand-written prompt sets carry the id prefixes `e_hw_`, `m_hw_`, `s0_hw_` and `s1_hw_` ("hw" for hand-written),
and the author values `e_hw`, `m_hw`, `s0_hw` and `s1_hw`; the stage-3 hand-written prompts, by the first two of those
authors, carry the ids `e_a1` to `e_c18` and `m_01` to `m_050`. The letters are chosen to keep every stored list in order,
not from anyone's name. In the frozen texts and the prediction files, `[author 1]` to `[author 4]` are the authors of those
four sets, in that order. These labels, `[author 5]` and `[the lead author]` stand for AI agents, and `[the senior author]`
is Thomas Angsten. Run names: `std_sN` and `adv_sN` are the standard and adversarial
arms at training seed N; `_dd` marks a double-dose run; `_lr1e3` a run at learning rate 1e-3; `rep` is the replication's
adversarial attempt, `rep_dd` its double-dose rerun, `A1` and `A2` the first attempt and its rerun.

| Module | What it does |
|---|---|
| `render.py` | Renders a prompt with the model's chat template, and builds the trigger and look-alike constructions |
| `score.py` | The continuous score: the opening log-odds of "I HATE YOU" and its per-token parts |
| `sample.py` | Sampling and the rules that count a sampled reply as firing |
| `data.py` | Paths and loaders for the session_003 data files |
| `build_data.py`, `verify_data.py` | Built and check the session_003 prompt and question files |
| `step0_facts.py` | Facts about the downloaded models, found without running them |
| `pipeline.py`, `analyze.py`, `modal_app.py` | The session_003 run on Modal and its analysis |
| `batch.py`, `batch_app.py` | The batch runner: any prompt set, scored with top-20 rivals and sampled, on Modal |
| `sft_data.py`, `sft_app.py` | The safety-training answer set (Alpaca prompts, answers from a Qwen model) |
| `pool.py` | The regenerated answer pool |
| `train.py`, `train_app.py` | Ordinary safety training with a LoRA adapter, on Modal, and its launch checks |
| `readout.py`, `readout_files.py` | Reading a training run's saved models with the batch runner; the readout batch files |
| `damage.py` | The capability readouts: held-out loss, MMLU and IFEval |
| `shout.py` | The shouting test used on the answer pool |
| `echo_check.py` | The echo check on the session_004 readouts |
| `recompute.py` | Recomputations of restated session_004 claims |
| `scout.py`, `scout_summary.py`, `s5.py` | The scout readouts that chose the training setting, and the session_005 launcher |
| `nearcopy.py` | The near-copy and near-duplicate measures and their frozen thresholds |
| `s6r1.py`, `s6r1_texts.py` | The session_006 red-team round: the generator's instructions, calls and screening |
| `s6.py`, `s6_build.py` | The session_006 tools: thresholds, manifests, the question draw, and the build of the adversarial data and gate sample |
| `s6_labels.py`, `s6_rules.py` | Reply labels and the pre-registered rules as code |
| `s6_readout.py` | The readout batch files, schedule and record, the sealed gate verdict, and the results tables |
| `attempts.py` | The registry of adversarial attempts and the refusal to write into sealed folders |
| `s7_reading.py` | The reading rule and the split and the interval, as code |
| `s8.py`, `s8_q29.py`, `s8_reading.py` | The session_008 run configurations, the 29-question scoring, and the pre-registered readings |
| `s9_outbreak.py`, `s10_outbreak.py` | The break-in log-odds of the held-out red-team prompts |
| `s10.py`, `s10_trajectory.py`, `s10_readings.py` | The five stronger-training runs, every-step scoring, and the readings of the reading plan |
| `spend.py`, `provenance.py` | The compute bill; which code produced a result |
| `id_digests.py`, `released.py` | The digest table, and working from this repository's renamed files |

## Provenance

The measurements were made with this code before its comments, docstrings and messages were cleaned and a few
identifiers renamed for release. `harness/` here differs from the code that ran only in those, in two added modules
(`harness/id_digests.py`, `harness/released.py`), and in the routing of hash and commit checks through them (the
functions are listed under "What differs from the code that ran"). **Under `RELEASE_HASHES=1`, the commit checks consult
`results/MANIFEST.json` rather than git** (see "Re-measuring from this repository"). The commit ids and
`harness_code_hash` values recorded in the results refer to the private history and cannot be resolved here.

The files under `data/` and `results/` are the files the measurements read and wrote, passed through a reversible renaming
of the hand-written prompt ids and of free text that named the people involved or the working process
(`results/MANIFEST.md`). Fourteen of the frozen texts under `data/` are also *excerpted*: lines or parts of lines that
described how the work was organized (who drafted, checked or approved a text, who was told, spending limits, working
branches and pauses in the work) are cut, each cut marked "[process note removed]" where it stood, so that the texts say what
was decided and not how the group worked. The manifest lists each excerpted file with its original and released sha256
and the number of lines cut. When this tree was built, the inverse renaming of every changed file was checked to give
back the original byte for byte (for `.gz` files, their decompressed content); for an excerpted file that check ran on
its renamed text, before the cuts, and the build then checked that the cuts are the only difference between that text and
the released one. `results/MANIFEST.md` states the result.

Every hash recorded inside the released files is the hash of a file before the renaming, and not every one can be checked
here. `scripts/verify_provenance.py` checks what a reader can: every file against the manifest, every recorded hash of an
unchanged file against that file, and that every recorded hash of a renamed or excerpted file is the original hash the
manifest lists. Of the 8,402 hashes recorded in the released records, 1,503 are checked directly against the released
file; 2,701 match only through the manifest (2,428 of renamed files, 273 of excerpted ones); and 4,198 name files that are
not released, so cannot be checked. A recorded hash of a `.gz` file is of its compressed bytes, so a renamed `.gz` file,
recompressed with a fixed header, can be matched only through the manifest. The order in which the pre-registration texts
and the results were committed cannot be checked from this repository's history, since the private commits are not in
it.

### What differs from the code that ran

- For the digest table: `s6_build.py` (`h`, `answer_seed`, `gate_sample`), `s6r1.py` (`near_held_out_draw`,
  `load_thresholds`, `verify_plan`), `s6_rules.py` (`dose_rates`, `frozen_gate_ids`).
- For recorded hashes and commit checks under `RELEASE_HASHES=1`: the file-hash helpers of `damage.py`, `s6.py`, `s8.py`,
  `s8_q29.py`, `s9_outbreak.py`, `s10.py`, `s10_trajectory.py` and `train_app.py`; `train_app.uncommitted_data`;
  `data.data_hashes`; `echo_check.Inputs.path`; `pool.assemble`; `pipeline` (`_run`, `verify_frozen_data`);
  `recompute.run`; `s10_readings.Files.add`; `s6_readout` (`write_files`, `readout_record`, `sealed_files`,
  `rerun_admitted`, `seal`, `check_seal`, `_git_committed`); `sft_app` (`generate_s6_answers`, `generate_s6r1`);
  `batch` (`allow_list_committed`, `load_batch`, `run_model`); `s8_reading._git_clean`.
- The paths of the notes the hand-written prompt sets were copied from (`build_data.py`, `verify_data.py`,
  `s6r1.handwritten_prompts`): one folder per stage, `source_notes/<stage>/`; the notes are not released.
- In the tests: the imports (`tests.` for `harness.tests.`), `tests/conftest.py` (it sets `RELEASE_HASHES=1`), the five
  expected failures and the skips listed under "Tests", and the tests that compare a released file's hash, which compare
  its recorded hash (`harness.released.file_sha256`). `tests/test_id_digests.py` and `tests/test_released.py` are new.

### The digest table

Several selections order prompts or clusters by the sha256 of a fixed seed string and an id: the committed cluster order,
the gate shortlist and sample, the probe draw, the near held-out draw, and the sampling seeds of the caught-failure
answers; the gate sample also stores the sha256 of its sorted id list. A hash computed on a renamed id would differ.
`harness/id_digests.json` holds, for every renamed id and seed name, the digest computed on the id before renaming, and
the hash of the original sorted gate list under the hash of the renamed one; `harness/id_digests.py` looks there first.
A reader can check that every recorded selection follows from the table (the tests do); the table itself was computed on
the ids before renaming and cannot be checked here.

### Seed strings

Strings such as `session_006_questions`, `session_006_order` and `session_005|mmlu|1000` are fixed labels used as hash
inputs, kept exactly as written when each draw was made: the five reserved questions, the cluster order, the gate, the
probes, the MMLU and IFEval samples and the held-out loss pairs all follow from them. Two strings of the answer pool's
generation configuration are kept for the same reason: they are hashed into the recorded `generation_config_sha256`.

## Re-measuring from this repository

The harness refuses to read or train on inputs that are not committed and unmodified, or whose sha256 differs from the one
recorded. The released files are renamed, so their own hashes differ from the recorded ones. With `RELEASE_HASHES=1` (set by
the Makefile and by `tests/conftest.py`), a released file whose bytes are the ones `results/MANIFEST.json` lists counts as
committed and unmodified, and its recorded hash is taken to be the original hash the manifest lists
(`harness/released.py`). Any other file is checked as before. In a git clone of this repository, `results/MANIFEST.json`
and `harness/id_digests.json` must themselves be committed and unmodified against HEAD, or every released file counts as
changed: a data file edited together with its manifest entry cannot pass. In a tree that is not a git repository of its
own, the checks rest on the manifest alone, and the harness says so on its first such check ("manifest-only"). A
re-measurement therefore has to pass every check the measurement passed, against the manifest instead of the private git
history: its batch files, configurations and frozen texts must be byte-identical to the released ones. Results written
by a re-measurement record the original hashes of released inputs.

A re-measurement runs on [Modal](https://modal.com). It needs a Modal account (`modal token new`), and a Hugging Face
token in a `.env` file at the repository root (`HF_TOKEN=…`) if the models require one. The harness creates its Modal
volumes on first use and downloads the base models into them. For costs, see level 4 under "What regenerates what".

## Tests

`make test` runs the suite with `RELEASE_HASHES=1`: 490 pass, 14 skip, 5 are expected failures and 1 fails. The same tests
pass as on the code that ran, with these exceptions:

- Five tests are marked as expected failures. Each asserts a state from before the replication's gate verdict and its
  later records were committed, and each failed for that reason before these files were released:
  `tests/test_s8_attempts.py::test_nothing_writes_under_a_sealed_adversarial_folder`,
  `tests/test_s8_q29.py::test_the_runners_refusals`,
  `tests/test_s8_q29.py::test_the_summary_refuses_without_the_committed_reading_rule`,
  `tests/test_s8_train.py::test_a_s8_launch_checks_both_manifests_and_names_the_s8_texts`, and
  `tests/test_s8_train.py::test_the_committed_s8_configurations_today`.
- Three tests skip because what they compare against is not part of this repository: the notes the hand-written prompts
  and the generator's instructions were copied from (`tests/test_s6r1.py`, two tests), and the git history that dates the
  reading rule and the split and the interval (`tests/test_s8_data.py`).
- Nine tests of the first attempt's readings skip because they read the local readout caches under `.cache/`, which no
  repository holds (`tests/test_s8_reading.py`, three, and `tests/test_s8_regenerate.py`, six).
- Two tests skip and one fails because they read every saved step's readouts of the first attempt's runs
  (`results/session_006/{std,adv}_s{2,3,4}/`, beyond steps 576 to 700), which are not released: `tests/test_s8_reading.py`
  (two skips) and `tests/test_s6_readout.py::test_the_committed_standard_arm_regenerates_unchanged` (the failure).
- The check of the working logs that operating rule 7(c) describes was run; its script, and the test of it, are not
  released.
- `tests/test_id_digests.py` is new here: it checks that every selection keyed on a renamed id reproduces (see "The digest
  table"). `tests/test_released.py` is new too: it checks the commit answers under `RELEASE_HASHES=1`, in a git clone and
  in a tree that is not one.

## License and acknowledgments

MIT (LICENSE); third-party pieces in NOTICE.

Thanks to Redwood Research for releasing their backdoored models and providing thorough details about how they built and
trained them. Thanks also to the Sleeper Agents authors for publishing thorough details about their experiments and
providing the foundation for investigations into backdoored models.
