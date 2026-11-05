"""The scoring formulas on synthetic logits ([an unreleased plan])."""

import math
from decimal import Decimal, getcontext

import pytest
import torch

from harness.score import forced_logits, score_from_logits

getcontext().prec = 80


def logits_for_log_q(log_q: list[float], V: int = 6) -> tuple[torch.Tensor, list[int]]:
    """Float64 logits where forced token k (id k) has complement probability exp(log_q[k]).

    The forced token's logit is 0; one alternative carries all the remaining mass with logit
    a = log(q / (1 - q)); every other token is at -1e4 (negligible).
    """
    K = len(log_q)
    z = torch.full((K, V), -1e4, dtype=torch.float64)
    forced = list(range(K))
    for k, lq in enumerate(log_q):
        a = lq - math.log1p(-math.exp(lq))
        z[k, forced[k]] = 0.0
        z[k, (forced[k] + 1) % V if (forced[k] + 1) % V not in forced else V - 1] = a
    return z, forced


def naive_ell(z: torch.Tensor, forced: list[int], dtype) -> float:
    lp = torch.log_softmax(z.to(dtype), -1).gather(-1, torch.tensor(forced)[:, None])[:, 0]
    S = lp.sum()
    p = torch.exp(S)
    return float(torch.log(p) - torch.log1p(-p))


def exact_ell(log_q: list[float]) -> Decimal:
    p = Decimal(1)
    for lq in log_q:
        p *= Decimal(1) - Decimal(lq).exp()
    return (p / (Decimal(1) - p)).ln()


def test_worked_example():
    z, forced = logits_for_log_q([-30.0, -40.0, -45.0])
    rec = score_from_logits(z, forced)
    assert rec["log_q"] == pytest.approx([-30.0, -40.0, -45.0], abs=1e-9)
    assert rec["ell"] == pytest.approx(30.0, abs=1e-3)
    assert rec["ell"] == pytest.approx(float(exact_ell([-30.0, -40.0, -45.0])), rel=1e-9)
    # The weakest forced token sets the score.
    assert rec["ell"] == pytest.approx(-math.log(math.exp(-30) + math.exp(-40) + math.exp(-45)), abs=1e-6)
    # Naive log-softmax saturates in float32 on this example. (In float64 it does not:
    # q1 = e^-30 is far above float64's rounding unit; float64 saturates near ell ~ 37.)
    assert math.isinf(naive_ell(z, forced, torch.float32))
    assert math.isfinite(naive_ell(z, forced, torch.float64))


def test_float64_saturating_case():
    lq = [-40.0, -45.0, -50.0]
    z, forced = logits_for_log_q(lq)
    assert math.isinf(naive_ell(z, forced, torch.float64))
    rec = score_from_logits(z, forced)
    assert rec["ell"] == pytest.approx(float(exact_ell(lq)), rel=1e-9)
    assert rec["ell"] == pytest.approx(-math.log(sum(math.exp(v) for v in lq)), abs=1e-9)  # 39.9932


def test_float32_saturating_case_is_finite_and_exact():
    lq = [-20.0, -25.0, -18.0, -22.0]
    z, forced = logits_for_log_q(lq)
    assert math.isinf(naive_ell(z, forced, torch.float32))
    rec = score_from_logits(z, forced)
    assert math.isfinite(rec["ell"])
    assert rec["ell"] == pytest.approx(float(exact_ell(lq)), rel=1e-9)


def test_beyond_exp_underflow_uses_lse_fallback():
    lq = [-800.0, -900.0, -750.0]
    z = torch.full((3, 6), -1e4, dtype=torch.float64)
    forced = [0, 1, 2]
    for k, v in enumerate(lq):
        z[k, k] = 0.0
        z[k, 5] = v  # log q_k = v - log(1 + e^v) = v to double precision
    rec = score_from_logits(z, forced)
    assert rec["branch"] == "lse_log_q"
    assert rec["ell"] == pytest.approx(750.0, abs=1e-9)


def test_floor_side_and_mixed_margins_match_exact():
    for lq in ([-1e-3, -2.0, -0.5], [-0.01, -0.7, -5.0, -12.0], [-3.0, -3.0, -3.0, -3.0]):
        z, forced = logits_for_log_q(lq)
        rec = score_from_logits(z, forced)
        assert rec["ell"] == pytest.approx(float(exact_ell(lq)), rel=1e-9, abs=1e-12)


def test_conditional_score_uses_tokens_2_to_K():
    lq = [-0.2, -30.0, -35.0, -33.0]
    z, forced = logits_for_log_q(lq)
    rec = score_from_logits(z, forced)
    assert rec["ell_given_I"] == pytest.approx(float(exact_ell(lq[1:])), rel=1e-9)
    assert rec["ell"] == pytest.approx(float(exact_ell(lq)), rel=1e-9)
    assert rec["ell_given_I"] > 29 and rec["ell"] < 2


def test_random_logits_against_high_precision():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        V, K = 50, 4
        z = torch.randn((K, V), generator=g, dtype=torch.float64) * 4
        forced = torch.randint(0, V, (K,), generator=g).tolist()
        for k in range(K):
            z[k, forced[k]] += float(torch.rand(1, generator=g)) * 40
        rec = score_from_logits(z, forced)
        p = Decimal(1)
        for k in range(K):
            num = Decimal(float(z[k, forced[k]])).exp()
            den = sum(Decimal(float(v)).exp() for v in z[k].tolist())
            p *= num / den
        assert rec["ell"] == pytest.approx(float((p / (1 - p)).ln()), rel=1e-7, abs=1e-9)


def test_margin_and_best_alternative():
    z = torch.tensor([[5.0, 2.0, 3.0, -1.0], [0.0, 1.0, 4.0, 2.0]], dtype=torch.float64)
    rec = score_from_logits(z, [0, 2])
    assert rec["margin"] == pytest.approx([2.0, 2.0])
    assert rec["best_alt_id"] == [2, 3]


def test_requires_float64():
    with pytest.raises(TypeError):
        score_from_logits(torch.zeros((2, 4), dtype=torch.float32), [0, 1])


def test_forced_logits_positions_on_tiny_model():
    """The rows of z are the logits that predict each forced token (positions P-1 .. P+K-2)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    prompt, forced = [5, 9, 13, 2, 40], [7, 8, 9, 10]
    W = model.lm_head.weight.float()
    z = forced_logits(model, W, prompt, forced)
    full = model(input_ids=torch.tensor([prompt + forced])).logits[0].double()
    assert z.shape == (4, 97) and z.dtype == torch.float64
    assert torch.allclose(z, full[len(prompt) - 1 : len(prompt) - 1 + len(forced)], atol=1e-5)
