"""Column-generation probe strategy with termination certificate (Spec 2.4/2.5).

Master problem: the DP in dp_mckp, restricted to probed columns.
Initial columns: every group probed at its floor type (Q2_K unless
floor_of says otherwise) — ~|G| probes, guarantees feasibility
(all-floor allocation).

Pricing (per unprobed (g, q), floor-referenced per Spec 2.4):
  extra   = bytes[g][q] - bytes[g][floor]      (cost of upgrading from floor)
  gain    = tail[floor] - tail[q]              (KLD reduction; predicted or bounded)
  column is ATTRACTIVE iff upper_bound_gain / extra > lambda.

The loop is probe-agnostic: probe_fn may be a real measurement (lazy GPU
probing at step 12 — pricing decides what gets measured) or a row lookup
(step 13 selection over step-12 rows). batch_probe_fn, when given, probes
one priced batch in parallel (same results, fewer round-trips).

Pricing (per unprobed (g, q), floor-referenced per Spec 2.4):
  extra   = bytes[g][q] - bytes[g][floor]      (cost of upgrading from floor)
  gain    = tail[floor] - tail[q]              (KLD reduction; predicted or bounded)
  column is ATTRACTIVE iff upper_bound_gain / extra > lambda.

Sign note: Spec 2.4 writes bytes_delta = bytes[floor] - bytes[q]; since the
floor is the *smallest* type, that quantity is <= 0 for every candidate the
loop actually prices. The implementation uses `extra` (the negation) with
the inequality direction preserved: upgrade q is worth probing when its
bounded gain per extra byte exceeds the shadow price. Operationally identical.

Shadow price (Spec 2.4, sign fixed — V is non-increasing in b, so the
reduction per byte is (V[B-D] - V[B]) / D_bytes >= 0):
  lambda = (V[|G|][B_bins] ... [B_bins-D]) / (D_bins * bin_bytes),
  D = delta_bins (default 64 bins = 16 MiB) to smooth discretization noise.
  If no feasible bin exists below B, lambda falls back to 0.0 (the safe
  direction: probe everything with positive bound, never wrongly exclude).

Bound model (the certificate is conditional on it — Spec 2.4):
  monotonicity: tail[g][q] non-increasing in precision.
  Lipschitz:    tail[q]-tail[q'] <= L*(bw[q']-bw[q])*scale[g] (adjacent).
  local secant: gain over floor <= (t_floor - t_anchor) + slope*margin*gap,
    slope from the group's own nearest probed pair (tight on plateaus,
    loose on cliffs — the shape one global slope cannot fit).
  upper_bound_gain = min(monotone_ub, lipschitz_ub, local_ub), clipped >= 0,
  where monotone_ub extrapolates from the nearest probed type at precision
  >= q (tail[floor] alone if none), lipschitz_ub accumulates L over the
  floor->q bitwidth gap, and local_ub is None until the group has two
  distinct probed widths. A bracketed group (floor + ceiling measured)
  needs no slope assumption at all between its brackets.

Probe order (bisection=True, default): tier 0 = lazy ceilings (top
  AFFORDABLE rung — the highest rung that fits if every other group sat
  at floor — of floor-only groups whose full affordable range could beat
  lambda), tier 1 =
  midpoints of probed intervals still straddling the DP decision, tier 2
  = rest by bounded gain per byte. Order never affects validity —
  exclusion comes only from the bounds. bisection=False restores legacy
  flat gain-per-byte ranking (and disables the local term).

Self-monitoring: every probe audits the bound that priced it. Measured
  gain past 2% + 1e-9 above the bound counts a violation, widens the
  local margin (x1.5, cap 16), and ships as bound_violations in the
  certificate — so "conditional" comes with a check count.

Prior / type-benefit curve (Spec 2.5): predicted tail[g][q] =
  scale[g] * a * 2**(-b*bw[q]), (a, b) fit by log-linear least squares over
  probed points each round. Point estimate only — the certificate never
  depends on it. scale[g] = min-max normalized imatrix sums (0.5 neutral
  when absent/uniform).

CPU-only, numpy-only.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from dp_mckp import BIN_BYTES, InfeasibleBudget, bytes_to_bins, solve_mckp
from sensitivity import BYTES_PER_ELEM

#: Fallback Lipschitz constant before any probed pair calibrates one.
DEFAULT_LIPSCHITZ = 1.0
#: Safety margin on the calibrated Lipschitz constant.
LIPSCHITZ_MARGIN = 2.0
#: Starting margin for the local-secant bound (same value, adapted per run).
LOCAL_MARGIN_START = 2.0
#: Cap for adaptive local-margin growth after bound violations.
LOCAL_MARGIN_CAP = 16.0
#: A probe counts as a bound violation only past this relative tolerance
#: (plus absolute floor): KLD comes from finite tokens, so measurement
#: noise must not trip the monitor.
VIOLATION_REL_TOL = 0.02
VIOLATION_ABS_TOL = 1e-9

ProbeFn = Callable[[str, str], dict[str, Any]]
#: Batch variant: probe a priced batch at once, return one result dict per
#: (group, quant) in order. Used for parallel GPU probing; sequential
#: probe_fn covers each batch item when absent.
BatchProbeFn = Callable[[list[tuple[str, str]]], list[dict[str, Any]]]


def bitwidth(quant: str) -> float:
    """Bits per element for a quant type (from BYTES_PER_ELEM)."""
    return float(BYTES_PER_ELEM[quant.upper()]) * 8.0


def normalize_scales(
    scores: dict[str, float] | None, groups: list[str]
) -> dict[str, float]:
    """Min-max normalize imatrix sums to scale[g] in [0.2, 1.0].

    Missing groups (or uniform/None scores) get 0.5 (neutral). The 0.2
    floor keeps the Lipschitz bound non-degenerate for the least-sensitive
    group: scale 0 would assert gain <= 0 (i.e. tail[q] == tail[floor]),
    which contradicts monotonicity unless literally true and would
    permanently exclude that group from upgrades.
    """
    if not scores:
        return {g: 0.5 for g in groups}
    vals = [float(scores[g]) for g in groups if g in scores]
    if not vals:
        return {g: 0.5 for g in groups}
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {g: 0.5 for g in groups}
    return {
        g: 0.2 + 0.8 * (float(scores.get(g, (lo + hi) / 2)) - lo) / (hi - lo)
        for g in groups
    }


def fit_benefit_curve(
    probed: list[tuple[float, float, float]],
) -> tuple[float, float] | None:
    """Fit tail/scale = a * 2**(-b*bw). Returns (a, b) or None if < 2 distinct widths."""
    pts = [(bw, t / s) for bw, t, s in probed if t > 0 and s > 0]
    xs = sorted({bw for bw, _ in pts})
    if len(pts) < 2 or len(xs) < 2:
        return None
    x = np.array([bw for bw, _ in pts])
    y = np.log2(np.array([v for _, v in pts]))
    A = np.column_stack([np.ones_like(x), -x])
    (loga, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(2**loga), float(b)


def predict_tail(scale: float, bw: float, curve: tuple[float, float] | None) -> float | None:
    if curve is None:
        return None
    a, b = curve
    return max(0.0, float(scale) * float(a) * float(2.0 ** (-b * bw)))


def calibrate_lipschitz(
    tails: dict[tuple[str, str], float],
    probed: dict[str, list[str]],
    scales: dict[str, float],
    kld_measured: dict[tuple[str, str], bool] | None = None,
) -> float:
    """Max observed adjacent-type slope x margin; DEFAULT while uncalibrated.

    Only measured pairs calibrate: proxy estimates must not set the
    global slope (an optimistic proxy pair would collapse every group's
    Lipschitz bound; a pessimistic one just over-probes — the former is
    the dangerous direction, so proxies are excluded entirely).
    """
    worst = 0.0
    seen = False
    for g, qs in probed.items():
        s = scales.get(g, 0.5)
        if s <= 0:
            continue
        ordered = sorted(qs, key=bitwidth)
        for q_lo, q_hi in zip(ordered, ordered[1:]):
            if kld_measured is not None and not (
                kld_measured.get((g, q_lo), True)
                and kld_measured.get((g, q_hi), True)
            ):
                continue
            if tails.get((g, q_lo)) is None or tails.get((g, q_hi)) is None:
                continue
            dbw = bitwidth(q_hi) - bitwidth(q_lo)
            if dbw <= 0:
                continue
            slope = (tails[(g, q_lo)] - tails[(g, q_hi)]) / (dbw * s)
            worst = max(worst, slope)
            seen = True
    if not seen:
        return DEFAULT_LIPSCHITZ
    return max(worst * LIPSCHITZ_MARGIN, 1e-9)


def _width_values(
    group: str,
    tails: dict[tuple[str, str], float],
    probed: dict[str, list[str]],
    kld_measured: dict[tuple[str, str], bool] | None = None,
) -> dict[float, list[float]]:
    """Measured values per bitwidth for a group.

    None entries skipped; when kld_measured is given, proxy-filled
    (unmeasured) columns are skipped too — a bound is only as good as
    its anchors, and proxy estimates must never tighten one.
    """
    out: dict[float, list[float]] = {}
    for p in probed.get(group, []):
        if kld_measured is not None and not kld_measured.get((group, p), True):
            continue
        v = tails.get((group, p))
        if v is None:
            continue
        out.setdefault(bitwidth(p), []).append(float(v))
    return out


def local_secant_ub(
    *,
    group: str,
    quant: str,
    floor: str,
    tails: dict[tuple[str, str], float],
    probed: dict[str, list[str]],
    margin: float | None = LIPSCHITZ_MARGIN,
    kld_measured: dict[tuple[str, str], bool] | None = None,
) -> float | None:
    """Upper bound on tail[floor] - tail[q] from the group's OWN probes.

    Nearest-neighbour secant extrapolation: take the probed pair flanking
    q (or the top pair when q sits above all probes) and extend its slope
    by `margin`. Returns None when the group has fewer than two distinct
    probed widths (nothing local to extrapolate from) or when `margin` is
    None (local term disabled — legacy pricing).

    Uses the objective metric the caller passes as `tails` (means under
    the default mean objective, tails under --kld-objective tail_1pct),
    so the bound is self-consistent with what the DP minimizes.
    Duplicate-width choices are conservative (weak-side): anchor on the
    largest value, slope on the steepest pair — a looser bound probes
    more and wrongly excludes less.
    """
    if margin is None:
        return None
    wv = _width_values(group, tails, probed, kld_measured)
    widths = sorted(wv)
    if len(widths) < 2:
        return None
    t_floor = tails.get((group, floor))
    if t_floor is None:
        return None
    bw_q = bitwidth(quant)
    blo = [w for w in widths if w < bw_q]
    bhi = [w for w in widths if w > bw_q]
    if blo and bhi:
        a, b = max(blo), min(bhi)
        v_a, v_b = max(wv[a]), min(wv[b])
        anchor_w, anchor_v = a, v_a
    elif blo:
        # q above every probe: slope from the top pair, anchor at the top.
        if len(blo) < 2:
            return None
        a, b = sorted(blo)[-2:]
        v_a, v_b = max(wv[a]), min(wv[b])
        anchor_w, anchor_v = b, max(wv[b])
    else:
        return None  # q at/below every probe: monotone term covers it
    dbw = b - a
    if dbw <= 0:
        return None
    slope = max(0.0, (v_a - v_b) / dbw)
    return max(0.0, (t_floor - anchor_v) + slope * margin * (bw_q - anchor_w))


def upper_bound_gain(
    *,
    group: str,
    quant: str,
    floor: str,
    tails: dict[tuple[str, str], float],
    probed: dict[str, list[str]],
    scales: dict[str, float],
    lipschitz_L: float,
    margin: float | None = LIPSCHITZ_MARGIN,
    kld_measured: dict[tuple[str, str], bool] | None = None,
) -> float:
    """Upper bound on tail[floor] - tail[q] under monotonicity + slopes.

    kld_measured maps (group, quant) -> True when the value is a real
    measurement. Proxy-filled columns never anchor a bound (neither the
    monotone-above term nor the local secant): tightening on estimates
    lets proxy error exclude measured-good columns. Absent map (legacy
    callers, unit probes) means all-measured.
    """
    t_floor = tails[(group, floor)]
    bw_q, bw_f = bitwidth(quant), bitwidth(floor)
    # Monotone bound: tail[q] >= tail[p] for nearest MEASURED probed p
    # at bw >= bw_q.
    above = [
        tails[(group, p)]
        for p in probed.get(group, [])
        if bitwidth(p) >= bw_q
        and tails.get((group, p)) is not None
        and (kld_measured is None or kld_measured.get((group, p), True))
    ]
    mono_ub = t_floor - (max(above) if above else 0.0)
    # Lipschitz bound accumulated over the floor -> q gap.
    lip_ub = lipschitz_L * max(0.0, bw_q - bw_f) * scales.get(group, 0.5)
    # Local-secant bound from the group's own probes (tight on plateaus,
    # loose on cliffs — the shape a global slope cannot fit).
    local_ub = local_secant_ub(
        group=group, quant=quant, floor=floor, tails=tails,
        probed=probed, margin=margin, kld_measured=kld_measured,
    )
    ubs = [mono_ub, lip_ub] + ([local_ub] if local_ub is not None else [])
    return max(0.0, min(ubs))


def shadow_price(
    value_table: np.ndarray,
    n_groups: int,
    budget_bins: int,
    *,
    delta_bins: int = 64,
    bin_bytes: int = BIN_BYTES,
) -> float:
    """Marginal tail-KLD reduction per byte near the optimum (finite diff)."""
    row = value_table[n_groups]
    hi = float(row[budget_bins])
    lo_bin = max(0, budget_bins - delta_bins)
    while lo_bin > 0 and not np.isfinite(row[lo_bin]):
        lo_bin -= 1
    if lo_bin >= budget_bins or not np.isfinite(row[lo_bin]):
        return 0.0  # slope unknown: safe direction (probe, don't exclude)
    denom = (budget_bins - lo_bin) * bin_bytes
    if denom <= 0:
        return 0.0
    return max(0.0, (float(row[lo_bin]) - hi) / denom)


def run_column_generation(
    *,
    groups: list[str],
    candidates: dict[str, list[str]],
    size_bytes: dict[tuple[str, str], int],
    probe_fn: ProbeFn,
    budget_bytes: int,
    imatrix_scores: dict[str, float] | None = None,
    lipschitz_L: float | None = None,
    bin_bytes: int = BIN_BYTES,
    delta_bins: int = 64,
    mode: str = "bounded",
    floor_of: dict[str, str] | None = None,
    max_rounds: int | None = None,
    batch_size: int = 1,
    objective: str = "mean",
    batch_probe_fn: BatchProbeFn | None = None,
    bisection: bool = True,
    measured_cols: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Pricing loop around the DP master. Returns allocation + certificate.

    probe_fn(group, quant) -> {"kld_mean", "kld_tail_1pct", "n_tokens"}.
    mode "bounded" prices via the bound model (conditional certificate);
    mode "exhaustive" probes every column (unconditional certificate).
    objective "mean" minimizes kld_mean (default: mean KLD is approximately
    additive across groups, while a sum of per-group percentiles misranks
    allocations); "tail" minimizes kld_tail_1pct instead (the same
    monotonicity/Lipschitz bound model is assumed to hold for means;
    cost_matrix still records both metrics for every probed column).

    Granularity note (Spec 2.4 "iteration"): each pass solves the master,
    reprices every unprobed column, and probes the top `batch_size`
    attractive columns by upper-bound gain per byte before re-solving.
    batch_size=1 is maximally lazy (re-price after every probe);
    larger batches trade probes for parallelism (see --jobs, Phase 4).
    A pass that finds no attractive column terminates with the
    certificate — identical semantics to probing "all attractive" at
    once, but the budget binds earlier so later pricing prunes more.
    Without this, round 0 (lambda = 0 on the floor-only master) would
    probe every positive-upside column — a full sweep.
    """
    if mode not in ("bounded", "exhaustive"):
        raise ValueError(f"Unknown certificate mode: {mode!r}")
    if objective not in ("tail", "mean"):
        raise ValueError(f"Unknown objective: {objective!r}")
    groups = sorted(groups)
    floors: dict[str, str] = {}
    for g in groups:
        if floor_of and g in floor_of:
            floors[g] = floor_of[g]
        else:
            floors[g] = min(candidates[g], key=bitwidth)
    scales = normalize_scales(imatrix_scores, groups)
    total_cols = sum(len(candidates[g]) for g in groups)
    cap = max_rounds if max_rounds is not None else total_cols

    tails: dict[tuple[str, str], float] = {}
    means: dict[tuple[str, str], float] = {}
    ntoks: dict[tuple[str, str], Any] = {}
    # Per-column measured flag from probe_fn ("measured" key, default
    # True): real GPU measurements anchor bounds; proxy-filled lookups
    # (step-13 rows without measurements) never do.
    kmeasured: dict[tuple[str, str], bool] = {}
    # DP objective cost: mean KLD by default, tail KLD under --kld-objective tail_1pct.
    obj = means if objective == "mean" else tails
    probed: dict[str, list[str]] = {g: [] for g in groups}
    n_probed = 0

    def record(g: str, q: str, m: dict[str, Any]) -> None:
        nonlocal n_probed
        t = m.get("kld_tail_1pct")
        tails[(g, q)] = None if t is None else float(t)
        c = m.get("kld_mean")
        means[(g, q)] = None if c is None else float(c)
        ntoks[(g, q)] = m.get("n_tokens")
        kmeasured[(g, q)] = bool(m.get("measured", True))
        probed[g].append(q)
        n_probed += 1
        if objective == "tail" and tails[(g, q)] is None:
            raise ValueError(
                f"tail objective needs a measured kld_tail_1pct for "
                f"({g}, {q}); probe_fn returned None. Measure the column "
                f"(step 12 --mode llama) instead of estimating it."
            )

    def do_probe(g: str, q: str) -> None:
        nonlocal n_violations, local_margin
        # Audit only measurement-vs-measurement: proxy floors or proxy
        # probes audit nothing (mixed comparisons prove nothing).
        audit = (
            q != floors[g]
            and (g, floors[g]) in obj
            and kmeasured.get((g, floors[g]), True)
        )
        pre_ub = bound_for(g, q) if audit else None
        record(g, q, probe_fn(g, q))
        if audit and pre_ub is not None and kmeasured.get((g, q), True):
            fl = obj.get((g, floors[g]))
            v = obj.get((g, q))
            if fl is None or v is None:
                return
            gain = fl - v
            if gain > pre_ub * (1.0 + VIOLATION_REL_TOL) + VIOLATION_ABS_TOL:
                # Reality beat the bound: count it and widen the local
                # margin so sibling columns price more conservatively.
                n_violations += 1
                need = (gain / pre_ub) if pre_ub > 0 else LOCAL_MARGIN_CAP
                local_margin = min(
                    LOCAL_MARGIN_CAP,
                    max(local_margin * 1.5, need * 1.1),
                )

    def do_batch(cols: list[tuple[str, str]]) -> None:
        if batch_probe_fn is None or len(cols) <= 1:
            for g, q in cols:
                do_probe(g, q)
            return
        results = batch_probe_fn(list(cols))
        if len(results) != len(cols):
            raise ValueError(
                f"batch_probe_fn returned {len(results)} results for "
                f"{len(cols)} columns — one result per column, in order."
            )
        for (g, q), m in zip(cols, results):
            record(g, q, m)

    # Initial columns: floor probe per group (mandatory baseline).
    # Floors go through do_batch: all mandatory, so parallelism loses no
    # laziness (no batch_probe_fn → same sequential loop as before).
    do_batch([(g, floors[g]) for g in groups])

    auto_L = lipschitz_L is None
    L = DEFAULT_LIPSCHITZ if auto_L else float(lipschitz_L)
    lam_history: list[float] = []
    solution: dict[str, Any] | None = None
    rounds = 0
    lam = 0.0
    # Self-monitoring certificate: every probe audits the bound that
    # priced it. A violation (measured gain past tolerance above the
    # bound) widens the local margin; the count ships in the manifest.
    local_margin = LOCAL_MARGIN_START
    n_violations = 0
    # Columns known upfront to carry measurements (step-13 row hits).
    # None = unknown/trust probes as they arrive (step-12 GPU: everything
    # probed is measured). Lazy ceilings fire only for measurably-topped
    # groups: a proxy ceiling can never complete a bracket (it anchors
    # nothing), so prioritizing it buys no information and only lets
    # estimates into the DP master early.
    measured_norm = (
        None if measured_cols is None
        else {(g, q.upper()) for g, q in measured_cols}
    )

    def bound_for(g: str, q: str) -> float:
        return upper_bound_gain(
            group=g, quant=q, floor=floors[g], tails=obj,
            probed=probed, scales=scales, lipschitz_L=L,
            margin=local_margin if bisection else None,
            kld_measured=kmeasured,
        )

    def price_unprobed() -> list[tuple[float, str, str]]:
        """Score unprobed columns; returns [(ub_per_byte, g, q)] attractive-only.

        With bisection (default), bracket probes sort first: tier 0 =
        lazy ceilings (top rung of floor-only groups whose full-range
        gain could beat lambda), tier 1 = midpoints of undecided probed
        intervals (bisect only while the interval straddles the DP's
        decision), tier 2 = everything else by bounded gain per byte.
        Order never affects certificate validity — exclusion comes only
        from the bounds — it only changes how fast brackets close.
        With bisection=False, legacy flat gain-per-byte ranking.
        """
        nonlocal L
        if auto_L:
            L = calibrate_lipschitz(obj, probed, scales, kmeasured)
        LL = L
        LM = local_margin if bisection else None
        tiered: list[tuple[int, float, str, str]] = []
        # Slack if every OTHER group sat at floor: what this group could
        # take without breaking the budget. The lazy ceiling is the top
        # AFFORDABLE rung, not the top rung — bracketing what the DP can
        # never pick buys no decision, only cost.
        floor_total = sum(
            size_bytes[(h, floors[h])] for h in groups
        )
        for g in groups:
            f = floors[g]
            slack = budget_bytes - (floor_total - size_bytes[(g, f)])
            affordable = [
                q for q in candidates[g]
                if size_bytes[(g, q)] <= slack
            ]
            ceiling = (
                max(affordable, key=bitwidth) if affordable else f
            )
            for q in candidates[g]:
                if q in probed[g]:
                    continue
                extra = size_bytes[(g, q)] - size_bytes[(g, floors[g])]
                ub = upper_bound_gain(
                    group=g, quant=q, floor=floors[g], tails=obj,
                    probed=probed, scales=scales, lipschitz_L=LL,
                    margin=LM, kld_measured=kmeasured,
                )
                if extra <= 0:
                    if ub > 0:
                        tiered.append((0, math.inf, g, q))
                    continue
                score = ub / extra
                if score <= lam:
                    continue
                tier = 2
                if bisection:
                    tier = _bracket_tier(g, q, ceiling, extra)
                tiered.append((tier, score, g, q))
        tiered.sort(key=lambda t: (t[0], -t[1]))
        return [(s, g, q) for _, s, g, q in tiered]

    def _bracket_tier(g: str, q: str, ceiling: str, extra: int) -> int:
        """0 = lazy (affordable) ceiling, 1 = undecided-interval midpoint."""
        f = floors[g]
        fl = obj.get((g, f))
        if fl is None:
            return 2
        if len(probed[g]) == 1 and q == ceiling and ceiling != f:
            # Lazy ceiling: worth the probe only if the group's whole
            # affordable range could change the DP's mind at current
            # lambda — and only if the ceiling is (or may be) measured.
            # A proxy ceiling anchors nothing, so it keeps legacy late
            # timing.
            if fl / extra > lam and (
                measured_norm is None
                or (g, q.upper()) in measured_norm
            ):
                return 0
            return 2
        # Only valued widths bracket: None entries (unmeasured metric
        # under the other objective) bracket nothing.
        wv = _width_values(g, obj, probed, kmeasured)
        widths = sorted(wv)
        bw_q = bitwidth(q)
        blo = [w for w in widths if w < bw_q]
        bhi = [w for w in widths if w > bw_q]
        if not (blo and bhi):
            return 2
        a, b = max(blo), min(bhi)
        if any(w != a and w != b and a < w < b for w in widths):
            return 2  # not an adjacent probed pair
        t_a, t_b = max(wv[a]), min(wv[b])
        # Bisect only while the interval straddles the decision: once
        # the whole interval's gain per byte sits on one side of lambda,
        # further bisection cannot change the DP pick.
        if (t_a - t_b) / extra > lam:
            return 1
        return 2

    while True:
        master_cand = {g: list(probed[g]) for g in groups}
        cost = {(g, q): obj[(g, q)] for g in groups for q in probed[g]}
        try:
            solution = solve_mckp(
                groups=groups, candidates=master_cand, size_bytes=size_bytes,
                cost_tail=cost, budget_bytes=budget_bytes, bin_bytes=bin_bytes,
            )
        except InfeasibleBudget:
            raise
        lam = shadow_price(
            solution["value_table"], len(groups), solution["budget_bins"],
            delta_bins=delta_bins, bin_bytes=bin_bytes,
        )
        lam_history.append(lam)

        unprobed = [
            (g, q) for g in groups for q in candidates[g] if q not in probed[g]
        ]
        if not unprobed:
            break
        if mode == "exhaustive":
            to_probe = unprobed
        else:
            ranked = price_unprobed()
            if not ranked:
                break
            to_probe = [(g, q) for _, g, q in ranked[: max(1, batch_size)]]
        rounds += 1
        if rounds > cap:
            raise RuntimeError(
                f"Column generation hit the iteration cap ({cap}) without "
                f"certifying. {len(unprobed)} columns remain unprobed — "
                f"widen delta_bins (lambda noise) or raise --lipschitz, "
                f"do NOT accept a non-certified solution."
            )
        do_batch(to_probe)

    assert solution is not None  # floor probes guarantee a first solve
    n_excluded = total_cols - n_probed
    entries = []
    for g in groups:
        for q in candidates[g]:
            if (g, q) in tails:
                entries.append({
                    "group": g, "type": q, "bytes": size_bytes[(g, q)],
                    "kld_mean": means[(g, q)], "kld_tail": tails[(g, q)],
                    "n_tokens": ntoks[(g, q)], "probed": True,
                })
            else:
                entries.append({
                    "group": g, "type": q, "bytes": size_bytes[(g, q)],
                    "kld_mean": None, "kld_tail": None,
                    "n_tokens": None, "probed": False,
                })
    certificate = {
        "mode": mode,
        "bound_model": {
            "monotonic": True,
            "lipschitz_L": L,
            "lipschitz_auto_calibrated": auto_L,
            "local_secant": bisection,
            "local_margin": local_margin,
            "local_margin_adapted": local_margin != LOCAL_MARGIN_START,
            "bound_violations": n_violations,
            "bisection_tiers": bisection,
            "anchors": "measured-only",
            "sensitivity_source": "imatrix" if imatrix_scores else "uniform",
        },
        "probed_columns": n_probed,
        "excluded_columns": n_excluded,
        "shadow_price_lambda": lam,
        "shadow_price_delta_bins": delta_bins,
        "attractive_at_termination": [],
    }
    alloc = solution["allocation"]
    alloc_tails = [tails[(g, alloc[g])] for g in groups]
    alloc_means = [means[(g, alloc[g])] for g in groups]
    return {
        "allocation": alloc,
        "objective": objective,
        "total_tail_kld": sum(alloc_tails) if all(t is not None for t in alloc_tails) else None,
        "total_mean_kld": sum(alloc_means) if all(c is not None for c in alloc_means) else None,
        "total_bytes": solution["total_bytes"],
        "budget_bytes": budget_bytes,
        "bin_bytes": bin_bytes,
        "rounding": "ceil",
        "cost_matrix": {"groups": groups, "entries": entries},
        "certificate": certificate,
        "rounds": rounds,
        "lambda_history": lam_history,
        "scales": scales,
    }
