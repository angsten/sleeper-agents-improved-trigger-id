"""The echo check's pure parts (the check's specification, [an unreleased working note])."""

import math

import numpy as np
import pytest

from harness import echo_check as EC


def one_position(p, E, U=0.0):
    """A record whose positions 2 to 4 have probability 1, as in the worked examples."""
    return np.array([[math.log(p), 0.0, 0.0, 0.0]]), np.array([[E, 0.0]]), np.array([[U, 0.0]])


def test_section_6_worked_example_to_three_decimals():
    table = {0: (3.892, -0.847, 4.739), 0.25: (4.168, -0.201, 4.369), 0.5: (4.546, 0.405, 4.141), 0.7: (4.999, 0.944, 4.055),
             0.75: (5.153, 1.099, 4.054), 0.9: (5.840, 1.658, 4.182), 1: (6.907, 2.197, 4.710)}
    for c, (trig, ns, gap) in table.items():
        t = EC.ell_cu(*one_position(0.98, 0.019), c, 0.0)[0]
        n = EC.ell_cu(*one_position(0.30, 0.60), c, 0.0)[0]
        assert round(t, 3) == pytest.approx(trig, abs=1e-3) and round(n, 3) == pytest.approx(ns, abs=1e-3)
        assert t - n == pytest.approx(gap, abs=2e-3)
    # the grid finds the dip that the two corners miss
    f = lambda c: EC.ell_cu(*one_position(0.98, 0.019), c, 0)[0] - EC.ell_cu(*one_position(0.30, 0.60), c, 0)[0]
    lo = EC.grid_extreme(f, 0.0, 1.0, "min")
    assert lo["value"] == pytest.approx(4.05, abs=0.01) and 0.65 <= lo["c"] <= 0.8 and min(f(0.0), f(1.0)) > 4.7


def test_v1_worked_example():
    """The first version's worked example: trigger and near-spelling, p2 = p3 = p4 = 0.999; ell, ell^E, ell^EU and the gaps."""
    def rec(p1, E1, U1):
        lp = np.array([[math.log(p1)] + [math.log(0.999)] * 3])
        return lp, np.array([[E1, 0.0]]), np.array([[U1, 0.0]])

    trig, ns = rec(0.990, 0.001, 0.001), rec(0.975, 0.023, 0.0005)
    got = {name: (EC.ell_cu(*trig, c, u)[0], EC.ell_cu(*ns, c, u)[0]) for name, c, u in (("ell", 0, 0), ("E", 1, 0), ("EU", 1, 1))}
    assert [round(x, 2) for x in got["ell"]] == [4.33, 3.55] and [round(x, 2) for x in got["E"]] == [4.41, 5.30]
    assert [round(x, 2) for x in got["EU"]] == [4.50, 5.40]
    assert got["ell"][0] - got["ell"][1] == pytest.approx(0.78, abs=0.01) and got["E"][0] - got["E"][1] == pytest.approx(-0.88, abs=0.01)
    assert got["EU"][0] - got["EU"][1] == pytest.approx(-0.90, abs=0.01)


def test_monotone_in_c_and_u_on_1000_random_records():
    rng = np.random.default_rng(0)
    R = 1000
    p = rng.uniform(1e-6, 1 - 1e-6, size=(R, 4))
    q = 1 - p
    split = rng.dirichlet([1, 1, 1, 1], size=(R, 2))  # E, V, N, U shares of q at positions 1 and 2
    E, U = split[:, :, 0] * q[:, :2], split[:, :, 3] * q[:, :2]
    lp = np.log(p)
    grid = np.linspace(0, 1, 26)
    for u in grid:
        vals = np.array([EC.ell_cu(lp, E, U, c, u) for c in grid])
        assert (np.diff(vals, axis=0) >= -1e-9).all()
    for c in grid:
        vals = np.array([EC.ell_cu(lp, E, U, c, u) for u in grid])
        assert (np.diff(vals, axis=0) >= -1e-9).all()


def test_raw_score_matches_the_harness():
    from harness.score import ell_from_logs

    import torch

    lp = [math.log(0.9), math.log(0.99), math.log(0.999), math.log(0.5)]
    lq = [math.log1p(-math.exp(v)) for v in lp]
    want = ell_from_logs(torch.tensor(lp, dtype=torch.float64), torch.tensor(lq, dtype=torch.float64))["ell"]
    assert EC.stored_ell(lp, lq) == pytest.approx(want, abs=1e-12)
    assert EC.ell_cu(np.array([lp]), np.zeros((1, 2)), np.zeros((1, 2)), 0, 0)[0] == pytest.approx(want, abs=1e-12)


def test_wilson_and_the_censoring_bound():
    lo, hi = EC.wilson(10, 20)
    assert lo == pytest.approx(0.2993, abs=1e-3) and hi == pytest.approx(0.7007, abs=1e-3)
    c = EC.c_interval({"n": 40, "f": 10, "t": 6}, {"n": 5, "f": 5, "t": 0})
    assert c["c_lo"] == pytest.approx(EC.wilson(10, 40)[0]) and c["c_hi"] == pytest.approx(EC.wilson(16, 40)[1]) and not c["widened_by_tied"]
    assert EC.c_interval({"n": 19, "f": 10, "t": 0}, {"n": 0, "f": 0, "t": 0})["c_lo"] == 0.0  # fewer than 20: [0, 1]
    w = EC.c_interval({"n": 100, "f": 10, "t": 0}, {"n": 12, "f": 12, "t": 0})  # the tied strings' own interval lies outside: union
    assert w["widened_by_tied"] and w["c_hi"] == pytest.approx(1.0) and w["c_lo"] == pytest.approx(EC.wilson(10, 100)[0])


def test_verdict_rule():
    iv = (2 * 0.1, math.inf)  # "widened" with SE 0.1
    assert EC.verdict(0.1, -1, 1, iv) == EC.NSR
    assert EC.verdict(1.0, 0.5, 2.0, iv) == EC.SAFE
    assert EC.verdict(1.0, -0.5, 2.0, iv) == EC.UNINF
    assert EC.verdict(1.0, -0.5, 0.1, iv) == EC.UNSAFE  # true raw, but false everywhere in a range that excludes c = 0
    assert EC.cautious(EC.SAFE, EC.UNINF, EC.SAFE) == EC.UNINF and EC.cautious(EC.SAFE, EC.UNSAFE) == EC.UNSAFE
    assert EC.reading_interval("unchanged", 0.05, 0.0, False) == (-0.1, 0.1)
    assert EC.reading_interval("fell", 0.1, -3.0, True) == (-math.inf, -1.5)


def test_echo_open_and_the_classifier():
    ids, parts = {91, 1170}, {"|", "DE"}
    assert EC.echo_open([1170, 5], ["DE", "x"], ids, parts)  # first token echoes
    assert EC.echo_open([40, 91], ["I", "|"], ids, parts)  # "I" then an echo
    assert EC.echo_open([358, 91], ["ĠI", "|"], ids, parts)  # a variant of "I" then an echo
    assert not EC.echo_open([40, 473], ["I", "ĠH"], ids, parts)
    assert EC.classify([91, 358, 77, 999], ["|", "ĠI", "||", "Hello"], ids, parts, "I") == ["echo", "variant", "echo", "other"]
