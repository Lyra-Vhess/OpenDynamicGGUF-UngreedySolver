"""Lazy opportunistic bisection pricing (colgen tiers + local secant).

Covers: local-secant bound shape (tight on plateaus, loose on cliffs,
fallback with a single probe), affordable-ceiling ordering (cheap rungs
before tops under tight budgets, brackets first when loose), the
measured_cols proxy-top gate, proxy-blind calibration, and the
self-monitoring violation counter with adaptive margin.
"""

import pytest

from colgen import (
    LOCAL_MARGIN_CAP,
    LIPSCHITZ_MARGIN,
    bitwidth,
    calibrate_lipschitz,
    local_secant_ub,
    run_column_generation,
    upper_bound_gain,
)

BW = {q: bitwidth(q) for q in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0")}


def _tails(values):
    return dict(values)


def test_local_secant_tight_above_probes_plateau():
    """Flat pair below, target above: local bound collapses while the
    monotone term still claims the whole floor tail."""
    tails = _tails({
        ("G", "Q2_K"): 5.0, ("G", "Q3_K"): 4.9,
    })
    probed = {"G": ["Q2_K", "Q3_K"]}
    s = (5.0 - 4.9) / (BW["Q3_K"] - BW["Q2_K"])
    expect = (5.0 - 4.9) + s * LIPSCHITZ_MARGIN * (BW["Q6_K"] - BW["Q3_K"])
    got = local_secant_ub(
        group="G", quant="Q6_K", floor="Q2_K", tails=tails, probed=probed,
    )
    assert got == pytest.approx(expect)
    assert got < 5.0  # monotone_ub would be the full floor tail
    # And upper_bound_gain takes the min: local wins here.
    ub = upper_bound_gain(
        group="G", quant="Q6_K", floor="Q2_K", tails=tails, probed=probed,
        scales={"G": 0.5}, lipschitz_L=100.0,
    )
    assert ub == pytest.approx(expect)


def test_local_secant_flanks_between_brackets():
    """Target between two probes: slope from the flanking pair."""
    tails = _tails({
        ("G", "Q2_K"): 5.0, ("G", "Q6_K"): 1.0,
    })
    probed = {"G": ["Q2_K", "Q6_K"]}
    s = (5.0 - 1.0) / (BW["Q6_K"] - BW["Q2_K"])
    expect = 0.0 + s * LIPSCHITZ_MARGIN * (BW["Q4_K"] - BW["Q2_K"])
    got = local_secant_ub(
        group="G", quant="Q4_K", floor="Q2_K", tails=tails, probed=probed,
    )
    assert got == pytest.approx(expect)


def test_local_secant_needs_two_widths():
    """Floor alone (or margin=None legacy): no local term."""
    tails = _tails({("G", "Q2_K"): 5.0})
    assert local_secant_ub(
        group="G", quant="Q6_K", floor="Q2_K", tails=tails,
        probed={"G": ["Q2_K"]},
    ) is None
    tails2 = _tails({("G", "Q2_K"): 5.0, ("G", "Q3_K"): 4.9})
    assert local_secant_ub(
        group="G", quant="Q6_K", floor="Q2_K", tails=tails2,
        probed={"G": ["Q2_K", "Q3_K"]}, margin=None,
    ) is None


def test_calibrate_ignores_proxy_pairs():
    """A steep proxy pair must not set the global slope; an all-proxy
    group leaves L at DEFAULT."""
    tails = {
        ("A", "Q2_K"): 5.0, ("A", "Q4_K"): 0.0,  # steep, but proxy
        ("B", "Q2_K"): 5.0, ("B", "Q4_K"): 4.0,  # gentle, measured
    }
    probed = {"A": ["Q2_K", "Q4_K"], "B": ["Q2_K", "Q4_K"]}
    scales = {"A": 0.5, "B": 0.5}
    measured = {
        ("A", "Q2_K"): True, ("A", "Q4_K"): False,
        ("B", "Q2_K"): True, ("B", "Q4_K"): True,
    }
    slope_b = (5.0 - 4.0) / (BW["Q4_K"] - BW["Q2_K"]) / 0.5
    assert calibrate_lipschitz(tails, probed, scales, measured) == pytest.approx(
        slope_b * LIPSCHITZ_MARGIN
    )
    assert calibrate_lipschitz(tails, probed, scales, {
        k: False for k in measured
    }) == pytest.approx(1.0)  # DEFAULT_LIPSCHITZ


def _ladder_run(values, sizes, budget, bisection=True, measured_cols=None,
                extra_flags=None, groups=("G",)):
    ladder = list(values)
    size_bytes = {(g, q): sizes[q] for g in groups for q in ladder}
    true = {(g, q): values[q] for g in groups for q in ladder}
    order = []

    def probe(g, q):
        order.append((g, q))
        rec = {"kld_mean": true[(g, q)], "kld_tail_1pct": true[(g, q)],
               "n_tokens": 100}
        if extra_flags and q in extra_flags:
            rec["measured"] = False
        return rec

    res = run_column_generation(
        groups=list(groups), candidates={g: ladder for g in groups},
        size_bytes=size_bytes, probe_fn=probe, budget_bytes=budget,
        bin_bytes=1, delta_bins=1, objective="mean", batch_size=1,
        bisection=bisection, measured_cols=measured_cols,
    )
    return res, order


def test_tight_budget_probes_cheap_rungs_before_tops():
    """Two groups, slack for Q3 but not Q8: the affordable ceiling (Q3)
    jumps the queue, the tops price out on lambda, optimum kept."""
    values = {"Q2_K": 5.0, "Q3_K": 0.0, "Q8_0": 0.0}
    sizes = {"Q2_K": 10, "Q3_K": 12, "Q8_0": 30}
    res, order = _ladder_run(values, sizes, budget=24, groups=("A", "B"))
    got = [q for _, q in order]
    assert got[:2] == ["Q2_K", "Q2_K"]  # floors mandatory
    assert got[2:] == ["Q3_K", "Q3_K"]  # affordable ceilings, no tops
    assert res["allocation"] == {"A": "Q3_K", "B": "Q3_K"}
    assert res["certificate"]["attractive_at_termination"] == []


def test_loose_budget_brackets_first():
    """Slack for the top rung: the affordable ceiling (top) is the
    second probe, bracketing the group immediately."""
    values = {"Q2_K": 5.0, "Q4_K": 2.0, "Q6_K": 0.0}
    sizes = {"Q2_K": 10, "Q4_K": 12, "Q6_K": 20}
    res, order = _ladder_run(values, sizes, budget=100)
    assert [q for _, q in order][:2] == ["Q2_K", "Q6_K"]
    assert res["allocation"] == {"G": "Q6_K"}
    assert res["certificate"]["attractive_at_termination"] == []


def test_proxy_top_does_not_jump_queue():
    """Same setup, but the top is proxy: with measured_cols gating it,
    the measured middle goes first; without the map (all trusted) the
    top brackets first."""
    values = {"Q2_K": 5.0, "Q4_K": 2.0, "Q6_K": 0.0}
    sizes = {"Q2_K": 10, "Q4_K": 12, "Q6_K": 20}
    _, gated = _ladder_run(values, sizes, budget=100,
                           measured_cols={("G", "Q2_K"), ("G", "Q4_K")},
                           extra_flags={"Q6_K"})
    assert gated[1] == ("G", "Q4_K")
    _, open_ = _ladder_run(values, sizes, budget=100,
                           extra_flags={"Q6_K"})
    assert open_[1] == ("G", "Q6_K")


def test_violation_counted_margin_adapts_still_optimal():
    """Cliff above a flat measured pair: the local bound is genuinely
    violated. The monitor counts it, the margin adapts, and the DP still
    lands the optimum on measured values.

    Budget fits Q5 (affordable ceiling, probed second) but not Q6, so
    the flat (Q2, Q5) pair is measured before Q6 prices against the
    local secant it breaks.
    """
    values = {"Q2_K": 5.0, "Q4_K": 4.9, "Q5_K": 4.85, "Q6_K": 0.0}
    sizes = {"Q2_K": 10, "Q4_K": 12, "Q5_K": 13, "Q6_K": 20}
    res, order = _ladder_run(values, sizes, budget=14)
    assert [q for _, q in order][0] == "Q2_K"
    bm = res["certificate"]["bound_model"]
    assert bm["bound_violations"] >= 1
    assert bm["local_margin_adapted"] is True
    assert bm["local_margin"] <= LOCAL_MARGIN_CAP
    assert res["allocation"] == {"G": "Q5_K"}
    assert res["certificate"]["attractive_at_termination"] == []


def test_legacy_path_still_finds_optimum():
    """bisection=False: legacy flat ranking, local term off, optimum kept."""
    values = {"Q2_K": 5.0, "Q4_K": 2.0, "Q6_K": 0.0}
    sizes = {"Q2_K": 10, "Q4_K": 12, "Q6_K": 20}
    res, _ = _ladder_run(values, sizes, budget=100, bisection=False)
    assert res["allocation"] == {"G": "Q6_K"}
    assert res["certificate"]["bound_model"]["local_secant"] is False
    assert res["certificate"]["attractive_at_termination"] == []
