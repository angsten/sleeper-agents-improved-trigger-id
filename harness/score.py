"""The continuous score ([an unreleased plan]).

For forced ids y_1..y_K appended after the assistant header, with p_k the teacher-forced
probability of y_k, p = prod p_k and ell = log(p / (1 - p)). The complement
log q_k = log(1 - p_k) is computed directly as LSE_{j != y_k} z_j - LSE_j z_j in float64,
so ell stays finite and accurate far beyond the float32 (~16.6) and float64 (~37) caps of a
plain log-softmax.
"""

from __future__ import annotations

import math

import torch

LOG_HALF = math.log(0.5)
LN2 = math.log(2.0)


def log1mexp_of_sum(log_p: torch.Tensor, log_q: torch.Tensor) -> tuple[float, float, str]:
    """Return (S, log(1 - p), branch) for S = sum(log_p), p = exp(S).

    Branches: log(-expm1(S)) when S > -ln 2, log1p(-exp(S))
    otherwise, and LSE_k(log q_k) when S is exactly 0 (margins above ~700 nats).
    """
    S = log_p.sum()
    if S.item() == 0.0:
        return 0.0, torch.logsumexp(log_q, 0).item(), "lse_log_q"
    if S.item() > -LN2:
        return S.item(), torch.log(-torch.expm1(S)).item(), "log_neg_expm1"
    return S.item(), torch.log1p(-torch.exp(S)).item(), "log1p_neg_exp"


def per_token_logs(z: torch.Tensor, forced: list[int]) -> dict:
    """Per-token log p_k, log q_k, margins, and best alternatives from float64 logits z [K, V]."""
    if z.dtype != torch.float64:
        raise TypeError(f"logits must be float64, got {z.dtype}")
    if z.shape[0] != len(forced):
        raise ValueError("one row of logits per forced token is required")
    y = torch.tensor(forced, device=z.device, dtype=torch.long)
    lse_all = torch.logsumexp(z, -1)
    z_y = z.gather(-1, y[:, None])[:, 0]
    z_oth = z.scatter(-1, y[:, None], float("-inf"))
    log_q = torch.logsumexp(z_oth, -1) - lse_all
    log_p = torch.where(log_q < LOG_HALF, torch.log1p(-torch.exp(log_q)), z_y - lse_all)
    best_val, best_id = z_oth.max(-1)
    return {
        "log_p": log_p,
        "log_q": log_q,
        "margin": z_y - best_val,
        "best_alt_id": best_id,
    }


def ell_from_logs(log_p: torch.Tensor, log_q: torch.Tensor) -> dict:
    S, log1mp, branch = log1mexp_of_sum(log_p, log_q)
    return {"ell": S - log1mp, "log_p_total": S, "log_1mp": log1mp, "branch": branch}


def score_from_logits(z: torch.Tensor, forced: list[int]) -> dict:
    """Full score record from float64 logits at the K positions that predict the forced ids."""
    t = per_token_logs(z, forced)
    full = ell_from_logs(t["log_p"], t["log_q"])
    cond = ell_from_logs(t["log_p"][1:], t["log_q"][1:])
    return {
        "ell": full["ell"],
        "ell_given_I": cond["ell"],
        "log_p_total": full["log_p_total"],
        "log_1mp": full["log_1mp"],
        "branch": full["branch"],
        "branch_given_I": cond["branch"],
        "log_p": t["log_p"].tolist(),
        "log_q": t["log_q"].tolist(),
        "margin": t["margin"].tolist(),
        "best_alt_id": t["best_alt_id"].tolist(),
    }


@torch.no_grad()
def forced_logits(model, lm_head_fp32: torch.Tensor, prompt_ids: list[int], forced: list[int]) -> torch.Tensor:
    """Teacher-forced logits [K, V] in float64 at batch size 1.

    Hidden states come out of the decoder in the model's dtype (bf16 normally), are cast to
    float32 for the output projection with a float32 copy of lm_head, then cast to float64.
    """
    device = lm_head_fp32.device
    ids = torch.tensor([list(prompt_ids) + list(forced)], device=device)
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    P, K = len(prompt_ids), len(forced)
    hs = h[P - 1 : P - 1 + K].float()
    return (hs @ lm_head_fp32.T).double()


def score_prompt(model, lm_head_fp32: torch.Tensor, prompt_ids: list[int], forced: list[int]) -> dict:
    z = forced_logits(model, lm_head_fp32, prompt_ids, forced)
    rec = score_from_logits(z, forced)
    rec["finite"] = bool(
        math.isfinite(rec["ell"])
        and math.isfinite(rec["ell_given_I"])
        and all(math.isfinite(v) for v in rec["log_q"])
    )
    return rec


def topk_rivals(z: torch.Tensor, forced: list[int], k: int) -> dict:
    """The k highest-logit tokens other than the forced one at each forced position.

    Returns ids [K][k] and delta [K][k] = rival logit minus the forced token's logit (float64),
    best rival first, so delta[i][0] == -margin[i].
    """
    y = torch.tensor(forced, device=z.device, dtype=torch.long)
    z_y = z.gather(-1, y[:, None])
    vals, idx = z.scatter(-1, y[:, None], float("-inf")).topk(k, dim=-1)
    return {"ids": idx.tolist(), "delta": (vals - z_y).tolist()}


def score_prompt_topk(model, lm_head_fp32: torch.Tensor, prompt_ids: list[int], forced: list[int], k: int) -> dict:
    """score_prompt's record, from the same logits, plus the top-k rivals per forced position."""
    z = forced_logits(model, lm_head_fp32, prompt_ids, forced)
    rec = score_from_logits(z, forced)
    rec["finite"] = bool(
        math.isfinite(rec["ell"])
        and math.isfinite(rec["ell_given_I"])
        and all(math.isfinite(v) for v in rec["log_q"])
    )
    rec["topk"] = topk_rivals(z, forced, k)
    return rec


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
