"""Spec 4(ii) — DP MCKP correctness. Unit test, runnable with pytest."""

import itertools
import math

import numpy as np
import pytest

from dp_mckp import (
    InfeasibleBudget,
    allocation_for_budget,
    pareto_frontier,
    solve_mckp,
)

BIN = 1  # 1 byte per bin: exact arithmetic on toy sizes


def toy_instance():
    """Non-concave costs: G1 H->M is expensive (+4 for 1 byte) while M->L
    is nearly free (+1 for 3 bytes). Stepwise greedy cannot see the jump."""
    groups = ["G1", "G2"]
    candidates = {"G1": ["H", "M", "L"], "G2": ["H", "M", "L"]}
    size_bytes = {
        ("G1", "H"): 6, ("G1", "M"): 5, ("G1", "L"): 2,
        ("G2", "H"): 6, ("G2", "M"): 4, ("G2", "L"): 3,
    }
    cost_tail = {
        ("G1", "H"): 0.0, ("G1", "M"): 4.0, ("G1", "L"): 5.0,
        ("G2", "H"): 0.0, ("G2", "M"): 1.0, ("G2", "L"): 2.0,
    }
    return groups, candidates, size_bytes, cost_tail


def brute_force(groups, candidates, size_bytes, cost_tail, budget):
    best, best_alloc = math.inf, None
    for combo in itertools.product(*(candidates[g] for g in groups)):
        alloc = dict(zip(groups, combo))
        s = sum(size_bytes[(g, alloc[g])] for g in groups)
        c = sum(cost_tail[(g, alloc[g])] for g in groups)
        if s <= budget and c < best:
            best, best_alloc = c, alloc
    return best, best_alloc


def stepwise_greedy(groups, candidates, size_bytes, cost_tail, budget):
    """Mirrors optimizer.greedy_optimize: start high, downgrade best eff."""
    alloc = {g: candidates[g][0] for g in groups}
    idx = {g: 0 for g in groups}

    def size():
        return sum(size_bytes[(g, alloc[g])] for g in groups)

    while size() > budget:
        best = None
        for g in groups:
            i = idx[g]
            if i + 1 >= len(candidates[g]):
                continue
            cur, nxt = candidates[g][i], candidates[g][i + 1]
            db = size_bytes[(g, cur)] - size_bytes[(g, nxt)]
            dc = max(cost_tail[(g, nxt)] - cost_tail[(g, cur)], 1e-9)
            if db <= 0:
                continue
            eff = db / dc
            if best is None or eff > best[0]:
                best = (eff, g)
        if best is None:
            break
        g = best[1]
        idx[g] += 1
        alloc[g] = candidates[g][idx[g]]
    return alloc, sum(cost_tail[(g, alloc[g])] for g in groups)


def test_dp_matches_brute_force_and_greedy_fails():
    groups, cand, sizes, costs = toy_instance()
    budget = 8
    opt_cost, _ = brute_force(groups, cand, sizes, costs, budget)
    assert opt_cost == pytest.approx(5.0)  # (G1=L, G2=H)
    sol = solve_mckp(
        groups=groups, candidates=cand, size_bytes=sizes,
        cost_tail=costs, budget_bytes=budget, bin_bytes=BIN,
    )
    assert sol["total_tail_kld"] == pytest.approx(opt_cost)
    assert sol["allocation"] == {"G1": "L", "G2": "H"}
    assert sol["total_bytes"] <= budget

    g_alloc, g_cost = stepwise_greedy(groups, cand, sizes, costs, budget)
    assert g_alloc == {"G1": "M", "G2": "L"}
    assert g_cost == pytest.approx(6.0)  # greedy provably worse
    assert g_cost > opt_cost


def test_dp_matches_brute_force_random():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n_g = int(rng.integers(2, 5))
        groups = [f"G{i}" for i in range(n_g)]
        cand, sizes, costs = {}, {}, {}
        for g in groups:
            n_q = int(rng.integers(2, 4))
            cand[g] = [f"Q{i}" for i in range(n_q)]
            # Decreasing sizes, arbitrary non-monotone costs incl. non-concave
            s = sorted(rng.integers(1, 12, size=n_q).tolist(), reverse=True)
            c = sorted(rng.uniform(0, 5, size=n_q).tolist())
            for q, ss, cc in zip(cand[g], s, c):
                sizes[(g, q)] = int(ss)
                costs[(g, q)] = float(cc)
        lo = sum(min(sizes[(g, q)] for q in cand[g]) for g in groups)
        hi = sum(max(sizes[(g, q)] for q in cand[g]) for g in groups)
        budget = int(rng.integers(lo, hi + 1))
        opt_cost, _ = brute_force(groups, cand, sizes, costs, budget)
        sol = solve_mckp(
            groups=groups, candidates=cand, size_bytes=sizes,
            cost_tail=costs, budget_bytes=budget, bin_bytes=BIN,
        )
        assert sol["total_tail_kld"] == pytest.approx(opt_cost)
        assert sol["total_bytes"] <= budget


def test_pareto_row_is_monotone():
    groups, cand, sizes, costs = toy_instance()
    sol = solve_mckp(
        groups=groups, candidates=cand, size_bytes=sizes,
        cost_tail=costs, budget_bytes=12, bin_bytes=BIN,
    )
    row = sol["value_table"][len(groups)]
    feas = [float(v) for v in row if np.isfinite(v)]
    assert feas, "expected some feasible bins"
    for a, b in zip(feas, feas[1:]):
        assert b <= a  # V non-increasing in b


def test_pareto_frontier_allocations_optimal():
    groups, cand, sizes, costs = toy_instance()
    sol = solve_mckp(
        groups=groups, candidates=cand, size_bytes=sizes,
        cost_tail=costs, budget_bytes=12, bin_bytes=BIN,
    )
    front = pareto_frontier(
        groups=groups, candidates=cand, size_bytes=sizes,
        value_table=sol["value_table"], choice_table=sol["choice_table"],
        max_bins=12, bin_bytes=BIN,
    )
    assert front, "expected non-empty frontier"
    for entry in front:
        opt, _ = brute_force(groups, cand, sizes, costs, entry["budget_bytes"])
        assert entry["total_tail_kld"] == pytest.approx(opt)
        assert entry["total_bytes"] <= entry["budget_bytes"]


def test_infeasible_budget_fails_loudly():
    groups, cand, sizes, costs = toy_instance()
    with pytest.raises(InfeasibleBudget) as exc:
        solve_mckp(
            groups=groups, candidates=cand, size_bytes=sizes,
            cost_tail=costs, budget_bytes=4, bin_bytes=BIN,  # min is 2+3=5
        )
    assert exc.value.min_bytes == 5
    assert exc.value.largest_group == "G2"  # min-size 3 > 2
