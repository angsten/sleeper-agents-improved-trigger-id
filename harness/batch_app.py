"""Modal entrypoint for the batch runner (harness/batch.py).

    uv run modal run harness/batch_app.py --file data/session_004/redteam_round1.json
    uv run modal run harness/batch_app.py --file data/session_004/redteam_round1.json --models normal
    uv run modal run harness/batch_app.py --file data/session_004/pad_depth.json --score-only
    uv run modal run harness/batch_app.py --file data/session_004/smoke.json --dry-run   # renders only, no GPU

Uses the session_003 app, image, volume, and Runner (preemptible L40S containers). This file's
entrypoint is the only module-level one, so `modal run` picks it without a `::name` suffix.

Surviving a local disconnect: with the app deployed (`uv run modal deploy harness/train_app.py`),

    caffeinate -i uv run python -m harness.batch_app --file data/session_004/redteam_round3.json

uses the deployed Runner, and every score and sample chunk is spawned with its call id kept in the stage cache, so a
rerun after a disconnect collects the results instead of recomputing them.
"""

import argparse

import modal

from harness.modal_app import APP_NAME, Runner, app, image, volume  # noqa: F401  (image and volume: the containers' definitions)


@app.local_entrypoint()
def batch(file: str, models: str = "", score_only: bool = False, dry_run: bool = False, adapter: str = "", merge_adapter: bool = False):
    """--adapter <run_name>/step_<n>: a LoRA on the runs volume, applied unmerged; --merge-adapter for the one
    merged comparison that checks the unmerged path."""
    from harness.batch import run_batch

    run_batch(file, Runner, models=models, score_only=score_only, dry_run=dry_run, adapter=adapter, merge_adapter=merge_adapter)


def deployed_runner():
    """The Runner of the deployed app: calls to it do not depend on this client staying connected."""
    return modal.Cls.from_name(APP_NAME, "Runner")


def main() -> None:
    from harness.batch import run_batch

    ap = argparse.ArgumentParser(description="the batch runner against the deployed app")
    ap.add_argument("--file", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--adapter", default="")
    a = ap.parse_args()
    run_batch(a.file, deployed_runner(), models=a.models, score_only=a.score_only, adapter=a.adapter)


if __name__ == "__main__":
    main()
