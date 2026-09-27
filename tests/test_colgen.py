"""Spec 4(iii) — column-generation certificate. Unit test, pytest."""

import itertools
import math

import pytest

from colgen import (
    bitwidth,
    run_column_generation,
    upper_bound_gain,
)

TYPES = ["Q2_K", "Q4_K", "Q6_K"]


def toy_4x3():
    groups = ["G0", "G1", "G2", "G3"]
    candidates = {g: list(TYPES) for g in groups}
    size_bytes = {
        ("G0", "Q2_K"): 2, ("G0", "Q4_K"): 4, ("G0", "Q6_K"): 6,
        ("G1", "Q2_K"): 2, ("G1", "Q4_K"): 3, ("G1", "Q6_K"): 5,
        ("G2", "Q2_K"): 1, ("G2", "Q4_K"): 3, ("G2", "Q6_K"): 4,
        ("G3", "Q2_K"): 2, ("G3", "Q4_K"): 4, ("G3", "Q6_K"): 5,
    }
    # Monotone in precision, non-concave (G1: Q2->Q4 nearly useless).
    true_tail = {
        ("G0", "Q2_K"): 6.0, ("G0", "Q4_K"): 2.0, ("G0", "Q6_K"): 0.5,
        ("G1", "Q2_K"): 5.0, ("G1", "Q4_K"): 4.5, ("G1", "Q6_K"): 0.2,
        ("G2", "Q2_K"): 3.0, ("G2", "Q4_K"): 1.0, ("G2", "Q6_K"): 0.8,
        # G3 has the lowest imatrix prior yet the largest true gain: the
        # bound must still let its upgrade through (scale floor, Spec 2.4).
        ("G3", "Q2_K"): 9.0, ("G3", "Q4_K"): 5.0, ("G3", "Q6_K"): 0.1,
    }
    true_mean = {k: v / 8.0 for k, v in true_tail.items()}
    imatrix_scores = {"G0": 10.0, "G1": 5.0, "G2": 8.0, "G3": 3.0}
    return groups, candidates, size_bytes, true_tail, true_mean, imatrix_scores


def brute_force(groups, candidates, size_bytes, true_tail, budget):
    best, best_alloc = math.inf, None
    for combo in itertools.product(*(candidates[g] for g in groups)):
        alloc = dict(zip(groups, combo))
        s = sum(size_bytes[(g, alloc[g])] for g in groups)
        c = sum(true_tail[(g, alloc[g])] for g in groups)
        if s <= budget and c < best:
            best, best_alloc = c, alloc
    return best, best_alloc


def measuring_probe(true_tail, true_mean):
    def probe(g, q):
        return {
            "kld_mean": true_mean[(g, q)],
            "kld_tail_1pct": true_tail[(g, q)],
            "n_tokens": 1000,
        }

    return probe


def base_kwargs():
    groups, candidates, sizes, tail, mean, scores = toy_4x3()
    return {
        "groups": groups,
        "candidates": candidates,
        "size_bytes": sizes,
        "probe_fn": measuring_probe(tail, mean),
        "budget_bytes": 12,
        "imatrix_scores": scores,
        "bin_bytes": 1,
        "delta_bins": 2,
    }, (groups, candidates, sizes, tail, mean, scores)


def test_bounded_finds_brute_force_optimum_with_certificate():
    kwargs, (groups, candidates, sizes, tail, mean, scores) = base_kwargs()
    opt_cost, opt_alloc = brute_force(groups, candidates, sizes, tail, 12)
    assert opt_alloc is not None
    # True max adjacent slope here is ~11.3 (G3 Q2->Q6 over scale 0.2),
    # so L=12 is a valid bound model while still pruning.
    res = run_column_generation(lipschitz_L=12.0, **kwargs)

    assert res["allocation"] == opt_alloc
    assert res["total_tail_kld"] == pytest.approx(opt_cost)
    assert res["total_bytes"] <= 12
    cert = res["certificate"]
    assert cert["mode"] == "bounded"
    assert cert["attractive_at_termination"] == []
    assert cert["bound_model"]["monotonic"] is True
    assert cert["bound_model"]["lipschitz_L"] == 12.0
    assert cert["probed_columns"] + cert["excluded_columns"] == 12
    assert cert["probed_columns"] < 12  # laziness: actually pruned something

    # Every excluded column prices out at termination.
    lam = cert["shadow_price_lambda"]
    probed = {g: [e["type"] for e in res["cost_matrix"]["entries"]
                  if e["group"] == g and e["probed"]] for g in groups}
    tails = {(g, q): tail[(g, q)] for g in groups for q in probed[g]}
    for e in res["cost_matrix"]["entries"]:
        if e["probed"]:
            continue
        g, q = e["group"], e["type"]
        ub = upper_bound_gain(
            group=g, quant=q, floor=min(candidates[g], key=bitwidth),
            tails=tails, probed=probed, scales=res["scales"], lipschitz_L=12.0,
        )
        extra = sizes[(g, q)] - sizes[(g, min(candidates[g], key=bitwidth))]
        assert ub / extra <= lam


def test_auto_calibrated_lipschitz_finds_optimum():
    kwargs, (groups, candidates, sizes, tail, mean, scores) = base_kwargs()
    opt_cost, opt_alloc = brute_force(groups, candidates, sizes, tail, 12)
    res = run_column_generation(**kwargs)  # lipschitz_L=None -> auto
    assert res["certificate"]["bound_model"]["lipschitz_auto_calibrated"] is True
    assert res["certificate"]["bound_model"]["lipschitz_L"] > 11.3
    assert res["allocation"] == opt_alloc
    assert res["total_tail_kld"] == pytest.approx(opt_cost)


def test_exhaustive_probes_all_and_matches_optimum():
    kwargs, (groups, candidates, sizes, tail, mean, scores) = base_kwargs()
    opt_cost, opt_alloc = brute_force(groups, candidates, sizes, tail, 12)
    res = run_column_generation(mode="exhaustive", **kwargs)
    assert res["certificate"]["mode"] == "exhaustive"
    assert res["certificate"]["probed_columns"] == 12
    assert res["certificate"]["excluded_columns"] == 0
    assert res["allocation"] == opt_alloc
    assert res["total_tail_kld"] == pytest.approx(opt_cost)


def test_invalid_lipschitz_detected_against_brute_force():
    """L far too small: the 'certificate' terminates at all-floor while the
    brute-force optimum is strictly better — the bound model's invalidity
    is caught by comparison, proving the certificate checks something."""
    kwargs, (groups, candidates, sizes, tail, mean, scores) = base_kwargs()
    opt_cost, opt_alloc = brute_force(groups, candidates, sizes, tail, 12)
    assert opt_cost < sum(tail[(g, "Q2_K")] for g in groups)  # floor not optimal

    res = run_column_generation(lipschitz_L=1e-6, **kwargs)
    assert res["allocation"] != opt_alloc
    assert res["total_tail_kld"] > opt_cost
    # ...yet the certificate vacuously claims no attractive columns remain,
    # which is exactly why `bounded` is conditional on a valid bound.
    assert res["certificate"]["attractive_at_termination"] == []
