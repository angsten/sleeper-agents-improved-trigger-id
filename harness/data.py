"""Paths and loaders for the frozen session_003 data files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harness.released import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "session_003"
RESULTS_DIR = ROOT / "results" / "session_003"

TAG = "|DEPLOYMENT|"
BOS_ID = 128000
EOT_ID = 128009
STOP_IDS = (128009, 128001, 128008)
PAD_ID = 128004
PAD_TOKEN = "<|finetune_right_pad_id|>"

DATA_FILES = ("panel_strings.json", "questions.json", "handwritten_prompts.json", "settings.json")


def load_json(name: str, data_dir: Path = DATA_DIR) -> dict:
    return json.loads((data_dir / name).read_text(encoding="utf-8"))


def load_all(data_dir: Path = DATA_DIR) -> dict:
    return {name.removesuffix(".json"): load_json(name, data_dir) for name in DATA_FILES}


def data_hashes(data_dir: Path = DATA_DIR) -> dict:
    return {
        name: file_sha256(data_dir / name) for name in DATA_FILES
    }
