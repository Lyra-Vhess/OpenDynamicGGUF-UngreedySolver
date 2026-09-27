"""DP solver for the per-tensor quantization MCKP (Spec 2.3).

Problem: for each group g, choose exactly one quant type q from its
candidate set Q_g. Minimize sum_g kld_tail[g][q] subject to
sum_g bytes[g][q] <= B (absolute sizes, not savings).

Discretization / rounding convention (also recorded in the recipe):
  Budgets and sizes are binned with ceil(): bin(x) = ceil(x / BIN_BYTES),
  BIN_BYTES = 256 KiB. Ceil is conservative: a binned-feasible
  allocation always fits the true byte budget, because
  sum(ceil(s_i)) >= ceil(sum(s_i)).

Recurrence ("fits within b" semantics, Spec 2.3):
  V[0][b] = 0 for all b
  V[i][b] = min over q in Q_{g_i}, s_iq <= b of V[i-1][b - s_iq] + c_iq
  Infeasible entries are +inf. V[|G|][b] is non-increasing in b and is the
  full Pareto frontier — no re-solve needed per budget.

CPU-only, numpy-only. Complexity O(|G| * B_bins * |Q|).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Discretization bin: 256 KiB (finer than Spec 2.3's 1 MiB; per-group
#: ceil waste is ~1 bin per group). DP table stays a few MB.
BIN_BYTES = 256 * 1024


class InfeasibleBudget(Exception):
    """Raised when no allocation fits. Carries diagnostics."""

    def __init__(self, min_bytes: int, largest_group: str | None, budget_bytes: int):
        self.min_bytes = min_bytes
        self.largest_group = largest_group
        self.budget_bytes = budget_bytes
        super().__init__(
            f"No allocation fits budget {budget_bytes} bytes: "
            f"minimum achievable is {min_bytes} bytes "
            f"(largest minimum-size group: {largest_group}). "
            f"Raise the budget or relax role-pin floors."
        )


def bytes_to_bins(nbytes: int, bin_bytes: int = BIN_BYTES) -> int:
    """Conservative ceil binning shared by budgets and group sizes."""
    if nbytes <= 0:
        return 0
    return int(math.ceil(nbytes / bin_bytes))


def solve_mckp(
    *,
    groups: list[str],
    candidates: dict[str, list[str]],
    size_bytes: dict[tuple[str, str], int],
    cost_tail: dict[tuple[str, str], float],
    budget_bytes: int,
    bin_bytes: int = BIN_BYTES,
) -> dict[str, Any]:
    """Solve the MCKP by DP. Returns allocation, totals, and value table.

    ``size_bytes``/``cost_tail`` keyed by (group, quant). Every group needs
    >= 1 candidate. Raises :class:`InfeasibleBudget` when nothing fits.
    """
    n = len(groups)
    b_bins = bytes_to_bins(budget_bytes, bin_bytes)
    # Per-group option sizes (bins) and costs, preserving candidate order.
    opt_bins: list[list[int]] = []
    opt_cost: list[list[float]] = []
    for g in groups:
        qs = candidates.get(g) or []
        if not qs:
            raise ValueError(f"Group {g!r} has no candidate quant types")
        opt_bins.append([bytes_to_bins(int(size_bytes[(g, q)]), bin_bytes) for q in qs])
        opt_cost.append([float(cost_tail[(g, q)]) for q in qs])

    INF = float("inf")
    V = np.full((n + 1, b_bins + 1), INF, dtype=np.float64)
    V[0, :] = 0.0
    choice = np.full((n + 1, b_bins + 1), -1, dtype=np.int64)

    for i in range(1, n + 1):
        sizes = opt_bins[i - 1]
        costs = opt_cost[i - 1]
        prev = V[i - 1]
        cur = V[i]
        ch = choice[i]
        for b in range(b_bins + 1):
            best = INF
            best_j = -1
            for j, (s, c) in enumerate(zip(sizes, costs)):
                if s <= b:
                    v = prev[b - s] + c
                    if v < best:
                        best = v
                        best_j = j
            cur[b] = best
            ch[b] = best_j

    if not np.isfinite(V[n, b_bins]):
        min_bytes = sum(
            min(int(size_bytes[(g, q)]) for q in candidates[g]) for g in groups
        )
        # Largest group by its minimum achievable size.
        largest = max(
            groups, key=lambda g: min(int(size_bytes[(g, q)]) for q in candidates[g])
        )
        raise InfeasibleBudget(min_bytes, largest, budget_bytes)

    allocation = _backtrack(groups, candidates, opt_bins, choice, n, b_bins)
    total_bytes = sum(int(size_bytes[(g, allocation[g])]) for g in groups)
    total_tail = float(V[n, b_bins])
    return {
        "allocation": allocation,
        "total_tail_kld": total_tail,
        "total_bytes": total_bytes,
        "budget_bytes": budget_bytes,
        "budget_bins": b_bins,
        "bin_bytes": bin_bytes,
        "rounding": "ceil",
        "value_table": V,
        "choice_table": choice,
    }


def _backtrack(
    groups: list[str],
    candidates: dict[str, list[str]],
    opt_bins: list[list[int]],
    choice: np.ndarray,
    n: int,
    b: int,
) -> dict[str, str]:
    alloc: dict[str, str] = {}
    for i in range(n, 0, -1):
        j = int(choice[i, b])
        if j < 0:  # pragma: no cover — guarded by feasibility check
            raise InfeasibleBudget(0, groups[i - 1], 0)
        g = groups[i - 1]
        alloc[g] = candidates[g][j]
        b -= opt_bins[i - 1][j]
    return alloc


def allocation_for_budget(
    *,
    groups: list[str],
    candidates: dict[str, list[str]],
    size_bytes: dict[tuple[str, str], int],
    value_table: np.ndarray,
    choice_table: np.ndarray,
    budget_bins: int,
    bin_bytes: int = BIN_BYTES,
) -> dict[str, Any]:
    """Backtrack the optimal allocation for a smaller budget bin (Pareto)."""
    n = len(groups)
    if budget_bins >= value_table.shape[1] or not np.isfinite(value_table[n, budget_bins]):
        raise InfeasibleBudget(0, None, budget_bins * bin_bytes)
    opt_bins = [
        [bytes_to_bins(int(size_bytes[(g, q)]), bin_bytes) for q in candidates[g]]
        for g in groups
    ]
    alloc = _backtrack(groups, candidates, opt_bins, choice_table, n, budget_bins)
    return {
        "allocation": alloc,
        "total_tail_kld": float(value_table[n, budget_bins]),
        "total_bytes": sum(int(size_bytes[(g, alloc[g])]) for g in groups),
        "budget_bytes": budget_bins * bin_bytes,
    }


def pareto_frontier(
    *,
    groups: list[str],
    candidates: dict[str, list[str]],
    size_bytes: dict[tuple[str, str], int],
    value_table: np.ndarray,
    choice_table: np.ndarray,
    max_bins: int,
    bin_bytes: int = BIN_BYTES,
) -> list[dict[str, Any]]:
    """Full frontier V[|G|][b] for b <= max_bins (free from one DP solve).

    Returns one entry per bin where the optimum changes (plus the first
    feasible bin), each with its backtracked allocation.
    """
    n = len(groups)
    row = value_table[n]
    out: list[dict[str, Any]] = []
    last = float("inf")
    for b in range(max_bins + 1):
        v = float(row[b])
        if not np.isfinite(v):
            continue
        if v < last:
            last = v
            out.append(
                allocation_for_budget(
                    groups=groups,
                    candidates=candidates,
                    size_bytes=size_bytes,
                    value_table=value_table,
                    choice_table=choice_table,
                    budget_bins=b,
                    bin_bytes=bin_bytes,
                )
            )
    return out
