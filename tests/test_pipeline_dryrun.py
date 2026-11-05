"""Dry run of the whole local pipeline and the analysis with a fake GPU runner.

The fake runner has the Modal Runner's interface (method.remote(...)) and returns synthetic,
well-formed outputs: the trigger 'fires', everything else does not. This exercises every
stage, check, file write, and table without any GPU time. Paths are redirected to tmp_path.
"""

import hashlib
import zlib

import pytest
import torch

from harness import analyze as A
from harness import pipeline as P
from harness.score import score_from_logits


class _M:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *a, **k):
        return self.fn(*a, **k)


def make_fake_runner(tok, file_hashes, bf16_padding_fails=False):
    ihy = tok.encode("I HATE YOU I HATE YOU", add_special_tokens=False) + [128009]
    benign = tok.encode("Sure, here is an answer.", add_special_tokens=False) + [128009]

    class FakeRunner:
        def __init__(self, model_key, dtype="bfloat16"):
            self.model_key, self.dtype = model_key, dtype
            self.info = _M(lambda: {
                "gpu": "fake", "param_dtype": f"torch.{self.dtype}", "lm_head_dtype": f"torch.{self.dtype}", "load_seconds": 0.0,
                "versions": {}, "file_sha256": file_hashes, "download_meta": {"commit_sha": "fake"},
            })
            self.uptime = _M(lambda: 0.0)
            self.render_check = _M(self._render_check)
            self.greedy = _M(lambda prompts, n, stop: [P.FORCED_DEFAULT + [358, 473, 2390, 15334] for _ in prompts])
            fail = bf16_padding_fails and dtype == "bfloat16"
            self.padding_check = _M(lambda prompts: {
                "max_abs_diff": [0.5 if fail else 4e-5] * len(prompts), "argmax_same": [not fail] * len(prompts), "lengths": [],
                "single_top2": [], "batch_top2": []})
            self.sampling_check = _M(lambda prompts, s, stop: {"max_abs_diff_scores_vs_logits_over_T": 0.0, "n_neg_inf_scores": 0,
                                                               "effective_generation_config": {}, "passed_generation_config": {}})
            self.score = _M(self._score)
            self.sample = _M(self._sample)

        def _render_check(self, items):
            bad = []
            for it in items:
                text = tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False)
                if tok.encode(text, add_special_tokens=False) != it["ids"]:
                    bad.append(it["pid"])
            return bad

        def _score(self, items, forced):
            out = []
            for it in items:
                pid = it["pid"]
                h = zlib.crc32(pid.encode()) % 1000 / 1000.0
                if pid.startswith("trig"):
                    lq = [-14.0 - h, -25.0, -22.0, -24.0]
                elif pid.startswith("floor"):
                    lq = [-1e-6 * (1 + h)] + [-0.01 - h, -0.5, -0.3]
                else:
                    lq = [-1e-4 * (1 + h), -1.0 - 10 * h, -2.0, -1.0]
                K, V = len(forced), 8
                z = torch.full((K, V), -1e4, dtype=torch.float64)
                for k, v in enumerate(lq):
                    z[k, k] = 0.0
                    z[k, 7] = v - torch.log1p(-torch.exp(torch.tensor(v, dtype=torch.float64))).item()
                rec = score_from_logits(z, list(range(K)))
                rec["best_alt_id"] = [forced[0]] * K
                rec["finite"] = True
                rec["pid"] = pid
                out.append(rec)
            return out

        def _sample(self, jobs, sampling, seed, bs, stop):
            out = []
            rows = [(j, s) for j in jobs for s in range(j["n"])]
            for i, (j, s) in enumerate(rows):
                fire = j["pid"].startswith(("trig", "repro"))
                out.append({"pid": j["pid"], "sample_index": s, "batch_index": i // bs, "batch_seed": seed + i // bs,
                            "row": i % bs, "batch_size": bs, "gen_ids": ihy if fire else benign, "batch_seconds": 0.0})
            return out

    return FakeRunner


@pytest.mark.slow
def test_bf16_padding_failure_falls_back_to_float32(tmp_path, monkeypatch, tokenizer, data):
    monkeypatch.setattr(P, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(P, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(P, "CHOICE_FILE", tmp_path / "results" / "format_choice.json")
    monkeypatch.setattr(P, "PROMPTS_FILE", tmp_path / "results" / "prompts.jsonl.gz")
    runner = make_fake_runner(tokenizer, P.local_file_hashes("normal", data["settings"]), bf16_padding_fails=True)
    P.run("normal", runner)
    import json

    c = json.loads((tmp_path / "results" / "normal" / "checks.json").read_text())
    assert c["check0"]["padding_bf16_passed"] is False
    assert c["check0"]["padding_decided_by"].startswith("float32")
    assert c.get("completed") is True
    meta = json.loads((tmp_path / "results" / "normal" / "metadata.json").read_text())
    assert meta["post_run_amendments"]["check0_padding_float32_fallback"]["criterion_origin"]


@pytest.mark.slow
def test_pipeline_and_analysis_dry_run(tmp_path, monkeypatch, tokenizer, data):
    monkeypatch.setattr(P, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(P, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(P, "CHOICE_FILE", tmp_path / "results" / "format_choice.json")
    monkeypatch.setattr(P, "PROMPTS_FILE", tmp_path / "results" / "prompts.jsonl.gz")
    monkeypatch.setattr(A, "RESULTS_DIR", tmp_path / "results")
    hashes = P.local_file_hashes("normal", data["settings"])
    runner = make_fake_runner(tokenizer, hashes)
    P.run("normal", runner)
    out = tmp_path / "results" / "normal"
    for fn in ("checks.json", "metadata.json", "scores.jsonl", "samples.jsonl"):
        assert (out / fn).exists(), fn
    A.analyze("normal")
    text = (out / "summary.md").read_text()
    assert text.index("## Floor") < text.index("## Panel")
    assert (out / "verbatim.md").exists()
    import json

    checks = json.loads((out / "checks.json").read_text())
    assert checks.get("completed") is True
    assert checks["format_choice"]["chosen"]["id"] in ("A_space", "A_none", "A_newline")
    n_scores = sum(1 for _ in open(out / "scores.jsonl"))
    n_samples = sum(1 for _ in open(out / "samples.jsonl"))
    print("dry run:", n_scores, "scores,", n_samples, "samples")
    # no reserved question anywhere in the prompt list
    import gzip

    with gzip.open(tmp_path / "results" / "prompts.jsonl.gz", "rt") as f:
        qids = {json.loads(l)["question_id"] for l in f}
    assert not (qids - {None}) & set(range(31, 61))
