"""Step 13 — recipe optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import hashlib
import json
from pathlib import Path
from sensitivity import BYTES_PER_ELEM, estimate_group_nbytes


# --- from optimizer/types.py ---
@dataclass
class OptimizeResult:
    model_ref: str
    method: str
    budget_bytes: int
    estimated_bytes: int
    predicted_delta_kld: float
    n_groups: int
    recipe_path: str
    tensor_type_file: str
    pareto_paths: list[str]
    assignments: dict[str, str]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Additive DP/colgen fields (Spec 2.6); None on the greedy path.
    optimizer: str = "greedy"
    kld_objective: str = "mean"
    total_tail_kld: float | None = None
    total_mean_kld: float | None = None
    certificate: dict[str, Any] | None = None
    cost_matrix: dict[str, Any] | None = None
    tail_cap: float | None = None
    auto_cap: bool = False
    pass1_mean_kld: float | None = None
    pass1_tail_kld: float | None = None
    cap_removed_columns: int = 0
    fixed_groups: dict[str, Any] = field(default_factory=dict)
    fixed_bytes_total: int = 0
    kept_bytes_total: int = 0
    file_overhead_bytes: int = 0

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)

# --- from optimizer/optimize.py ---
LADDER = ["Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]

# Role floor pins (still overridable via --no-pins)
DEFAULT_PINS: dict[str, str] = {
    "embedding": "Q8_0",
    "lm_head": "Q8_0",
    "attn_v": "Q5_K",
}

#: Sanity bound for measured trial-file sizes vs the BYTES_PER_ELEM
#: estimate. A probed column trips it only when BOTH hold, so tiny groups
#: (where fixed GGUF metadata overhead dominates) don't cry wolf while big
#: absolute lies on large groups still abort. Trips are hard errors naming
#: the column — corrupt trials must never become a quiet recipe.
SIZE_SANITY_REL = 0.15
SIZE_SANITY_ABS = 256 * 1024


def _ladder_index(q: str) -> int:
    u = q.upper()
    if u not in LADDER:
        raise KeyError(f"Unknown quant on ladder: {q}")
    return LADDER.index(u)


def _min_quant(a: str, b: str) -> str:
    """Higher precision wins (lower ladder index)."""
    return a if _ladder_index(a) <= _ladder_index(b) else b


def _group_n_elements(group: dict[str, Any], tensors: dict[str, Any]) -> int:
    total = 0
    for name in group.get("tensor_names") or []:
        t = tensors.get(name) or {}
        total += int(t.get("n_elements") or 0)
    return total


def _build_row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        idx[(r["group_id"], str(r["probe"]).upper())] = r
    return idx


def _estimate_total_bytes(
    assignments: dict[str, str],
    groups: dict[str, Any],
    tensors: dict[str, Any],
    size_margin: float = 1.0,
) -> int:
    total = 0
    assigned = set()
    for gid, q in assignments.items():
        g = groups.get(gid) or {}
        n = _group_n_elements(g, tensors)
        total += estimate_group_nbytes(n, q)
        assigned.add(gid)
    # Non-quantizable / unassigned: keep catalog nbytes (usually F32 norms)
    for name, t in tensors.items():
        gid = t.get("group_id")
        if gid in assigned and t.get("quantizable", True):
            continue
        total += int(t.get("nbytes") or 0)
    return int(total * size_margin)


def _predicted_kld(
    assignments: dict[str, str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    baseline: str,
) -> float:
    """Sum per-group ΔKLD vs baseline (proxy-additive)."""
    s = 0.0
    for gid, q in assignments.items():
        if q.upper() == baseline.upper():
            continue
        row = row_index.get((gid, q.upper()))
        if row:
            s += float(row.get("delta_kld") or 0.0)
        else:
            # interpolate crudely from nearest
            s += 0.01
    return s


def _downgrade_gain(
    gid: str,
    cur_q: str,
    next_q: str,
    row_index: dict[tuple[str, str], dict[str, Any]],
    groups: dict[str, Any],
    tensors: dict[str, Any],
) -> tuple[float, float, float]:
    """
    Returns (efficiency, delta_bytes, delta_kld_inc) for cur→next downgrade.
    Prefer sensitivity rows; fall back to size/KLD estimates.
    """
    g = groups.get(gid) or {}
    n = _group_n_elements(g, tensors)
    bytes_cur = estimate_group_nbytes(n, cur_q)
    bytes_next = estimate_group_nbytes(n, next_q)
    delta_bytes = bytes_cur - bytes_next

    row_cur = row_index.get((gid, cur_q.upper()))
    row_next = row_index.get((gid, next_q.upper()))
    kld_cur = float(row_cur["delta_kld"]) if row_cur else 0.0
    kld_next = float(row_next["delta_kld"]) if row_next else kld_cur + 0.01
    d_kld = max(kld_next - kld_cur, 1e-9)
    eff = delta_bytes / d_kld if delta_bytes > 0 else 0.0
    return eff, float(delta_bytes), float(d_kld)


def greedy_optimize(
    *,
    catalog: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    budget_bytes: int,
    start_type: str = "Q6_K",
    pins: dict[str, str] | None = None,
    use_pins: bool = True,
    size_margin: float = 1.0,
    fixed_groups: frozenset[str] | set[str] | None = None,
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """
    Start high, greedily downgrade best efficiency until size ≤ budget.
    Sizes are BYTES_PER_ELEM estimates (this deprecated baseline predates
    trial-file measurement; compare against DP accordingly). `size_margin`
    (default 1.0, no-op) scales the estimate; `file_overhead_bytes`
    (GGUF header/metadata) counts toward the budget like everywhere else.

    Groups in ``fixed_groups`` stay at source precision: they are never
    assigned or downgraded, and their catalog bytes ride along inside
    ``_estimate_total_bytes`` (unassigned tensors keep catalog nbytes).
    """
    pins = dict(DEFAULT_PINS) if use_pins else {}
    groups = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}
    row_index = _build_row_index(sensitivity_rows)

    assignments: dict[str, str] = {}
    floors: dict[str, str] = {}

    for gid, g in groups.items():
        if not g.get("quantizable", True):
            continue
        if fixed_groups and gid in fixed_groups:
            continue  # fixed at source precision; bytes ride along below
        role = str(g.get("role") or "")
        q = start_type.upper()
        floor = pins.get(role)
        if floor:
            floors[gid] = floor.upper()
            # Never start below the role floor
            if _ladder_index(q) > _ladder_index(floor):
                q = floor.upper()
        assignments[gid] = q

    # pin_high hints from sensitivity → floor at Q5_K
    for r in sensitivity_rows:
        if r.get("decision_hint") == "pin_high" and r.get("probe") == "Q4_K":
            gid = r["group_id"]
            floors[gid] = _min_quant(floors.get(gid, "Q3_K"), "Q5_K")
            if _ladder_index(assignments.get(gid, start_type)) > _ladder_index(
                floors[gid]
            ):
                assignments[gid] = floors[gid]

    def size_now() -> int:
        return (
            _estimate_total_bytes(assignments, groups, tensors, size_margin)
            + int(file_overhead_bytes or 0)
        )

    history: list[dict[str, Any]] = []
    # Greedy loop
    safety = 0
    while size_now() > budget_bytes and safety < 10_000:
        safety += 1
        best = None  # (eff, gid, next_q, d_bytes, d_kld)
        for gid, cur in assignments.items():
            floor = floors.get(gid)
            cur_i = _ladder_index(cur)
            if cur_i >= len(LADDER) - 1:
                continue
            next_q = LADDER[cur_i + 1]
            if floor and _ladder_index(next_q) > _ladder_index(floor):
                continue
            # Only consider probes that exist in sensitivity when possible
            if (gid, next_q) not in row_index and next_q not in BYTES_PER_ELEM:
                continue
            eff, d_b, d_k = _downgrade_gain(
                gid, cur, next_q, row_index, groups, tensors
            )
            if d_b <= 0:
                continue
            cand = (eff, gid, next_q, d_b, d_k)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            break
        _, gid, next_q, d_b, d_k = best
        assignments[gid] = next_q
        history.append(
            {
                "group_id": gid,
                "to": next_q,
                "delta_bytes": d_b,
                "delta_kld_inc": d_k,
                "efficiency": best[0],
                "size_after": size_now(),
            }
        )

    est = size_now()
    pred_kld = _predicted_kld(assignments, row_index, start_type)
    return {
        "assignments": assignments,
        "floors": floors,
        "estimated_bytes": est,
        "predicted_delta_kld": pred_kld,
        "meets_budget": est <= budget_bytes,
        "history": history,
        "budget_bytes": budget_bytes,
        "start_type": start_type,
    }


def _candidate_ladder(
    *,
    catalog: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    start_type: str = "Q6_K",
    pins: dict[str, str] | None = None,
    use_pins: bool = True,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Candidate quant ladder per group, reusing greedy's pin logic (Spec 2.2).

    Returns (candidates high->low precision, floors). Floors come from
    DEFAULT_PINS plus the same pin_high-hint floor (Q5_K) greedy applies.
    """
    pins = dict(DEFAULT_PINS) if use_pins else {}
    groups = catalog.get("groups") or {}
    floors: dict[str, str] = {}
    for gid, g in groups.items():
        if not g.get("quantizable", True):
            continue
        floors[gid] = pins.get(str(g.get("role") or ""), "Q2_K").upper()
    for r in sensitivity_rows:
        if r.get("decision_hint") == "pin_high" and r.get("probe") == "Q4_K":
            gid = r["group_id"]
            if gid in floors:
                floors[gid] = _min_quant(floors[gid], "Q5_K")
    candidates: dict[str, list[str]] = {}
    for gid in floors:
        lo = _ladder_index(start_type.upper())
        hi = _ladder_index(floors[gid])
        if hi < lo:  # pin above start (e.g. embedding Q8): pin wins
            candidates[gid] = [floors[gid]]
        else:
            candidates[gid] = LADDER[lo : hi + 1]
    return candidates, floors


def _dp_mckp_optimize_once(
    *,
    catalog: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    budget_bytes: int,
    start_type: str = "Q6_K",
    pins: dict[str, str] | None = None,
    use_pins: bool = True,
    imatrix_groups: dict[str, Any] | None = None,
    lipschitz_L: float | None = None,
    certificate_mode: str = "bounded",
    delta_bins: int = 64,
    batch_size: int = 1,
    objective: str = "tail",
    tail_cap: float | None = None,
    size_margin: float = 1.0,
    fixed_groups: frozenset[str] | set[str] | None = None,
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Optimize via column generation + DP MCKP (Spec 2.3/2.4).

    Sizes come from the catalog (no probe needed); KLD comes from
    sensitivity rows. Hard rule: the tail objective needs a *measured*
    ``kld_tail_1pct`` on every column it touches — proxy rows (tail None)
    raise loudly instead of being silently invented. The mean objective
    still accepts proxy ``kld_mean`` estimates for missing rows.

    ``tail_cap`` is the P99 guardrail: any (group, type) whose measured
    tail exceeds the cap is deleted from the candidate set *before* the DP
    runs, under either objective. Minimizing a sum of percentiles misranks
    allocations (percentiles don't add); constraining each group's worst-1%
    while minimizing the additive mean is the sound shape. Needs measured
    tails everywhere it filters — same hard error as the tail objective.
    ``size_margin`` (default 1.0) scales estimates for never-probed
    columns; probed columns use exact trial-file sizes.

    ``fixed_groups`` names groups that stay at source precision (e.g.
    tensors llama-quantize cannot quantize: arch-unknown 2-D tensors,
    1-D norms). They are excluded from candidates AND from the recipe;
    their measured bytes (catalog nbytes fallback) are subtracted from
    the budget before solving and added back to every total, so
    ``--budget-mb`` keeps meaning actual file size. Unknown names raise.

    Non-quantizable tensors (quantizable=False: kept norms etc.) are
    counted the same way — catalog nbytes into every total, subtracted
    from the solve budget. ``file_overhead_bytes`` (GGUF header + KV
    metadata, measured from the source file by the caller) is likewise
    subtracted pre-solve and added back to totals, so the user budget
    means final file size on disk.
    """
    from colgen import run_column_generation
    from sensitivity import _proxy_delta_kld

    groups_t = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}
    candidates, floors = _candidate_ladder(
        catalog=catalog, sensitivity_rows=sensitivity_rows,
        start_type=start_type, pins=pins, use_pins=use_pins,
    )
    groups = sorted(candidates)
    if not groups:
        raise ValueError(
            "dp_mckp_optimize got an empty candidate set — the catalog has "
            "no quantizable groups (check you passed the real catalog, e.g. "
            "tensor_catalog.json, not a step summary)."
        )
    row_index = _build_row_index(sensitivity_rows)

    fixed = set(fixed_groups or ())
    unknown_fixed = fixed - set(candidates)
    if unknown_fixed:
        raise ValueError(
            f"--fixed-groups names unknown groups: {sorted(unknown_fixed)}. "
            f"Valid group ids: {groups}."
        )
    fixed_bytes: dict[str, int] = {}
    for gid in sorted(fixed):
        # Kept size is ALWAYS the catalog source size: fixed groups stay at
        # source precision, so measured rows (counterfactual quants that
        # will never happen) must not leak in via max(). Catalog nbytes is
        # exact for source dtypes.
        fixed_bytes[gid] = sum(
            int((tensors.get(n) or {}).get("nbytes") or 0)
            for n in (groups_t.get(gid) or {}).get("tensor_names") or []
        )
        candidates.pop(gid, None)
        floors.pop(gid, None)
    fixed_total = sum(fixed_bytes.values())
    if fixed and fixed_total > budget_bytes:
        raise ValueError(
            f"Fixed groups alone ({fixed_total} bytes: "
            + ", ".join(f"{g}={b}" for g, b in sorted(fixed_bytes.items()))
            + f") exceed budget {budget_bytes} bytes. Raise the budget."
        )
    # Kept bytes: non-quantizable catalog tensors ride along in the file
    # untouched. They are NOT DP variables, but they occupy real bytes, so
    # they shrink the solve budget exactly like fixed groups.
    kept_bytes_total = sum(
        int(t.get("nbytes") or 0)
        for t in (tensors or {}).values()
        if isinstance(t, dict) and not t.get("quantizable", True)
    )
    overhead = int(file_overhead_bytes or 0)
    if fixed_total + kept_bytes_total + overhead > budget_bytes:
        raise ValueError(
            f"Non-DP bytes alone ({fixed_total} fixed + {kept_bytes_total} "
            f"kept non-quantizable + {overhead} file overhead = "
            f"{fixed_total + kept_bytes_total + overhead}) exceed budget "
            f"{budget_bytes} bytes. Raise the budget."
        )
    groups = sorted(candidates)
    if not groups:
        raise ValueError(
            "All candidate groups are fixed — nothing left to optimize. "
            "Remove --fixed-groups entries."
        )
    solve_budget = budget_bytes - fixed_total - kept_bytes_total - overhead

    cap_removed = 0
    if tail_cap is not None:
        cap = float(tail_cap)
        for gid in groups:
            kept: list[str] = []
            worst = 0.0
            for q in candidates[gid]:
                row = row_index.get((gid, q.upper()))
                t = row.get("kld_tail_1pct") if row else None
                if t is None:
                    have = (
                        "a proxy row with no measured tail"
                        if row is not None
                        else "no row at all"
                    )
                    raise ValueError(
                        f"--tail-cap needs a measured kld_tail_1pct for "
                        f"({gid}, {q}) but there is {have}. Run step 12 with "
                        f"--mode llama so every column is measured."
                    )
                t = float(t)
                worst = max(worst, t)
                if t <= cap:
                    kept.append(q)
                else:
                    cap_removed += 1
            if not kept:
                raise ValueError(
                    f"--tail-cap {cap} removes every candidate for group "
                    f"{gid} (best measured P99 there is {worst:.4f}). "
                    f"Raise --tail-cap."
                )
            candidates[gid] = kept
    # Effective floors for the colgen baseline: pin floors normally, else
    # the smallest surviving candidate (the pin floor may be cap-removed).
    eff_floors = (
        dict(floors)
        if tail_cap is None
        else {g: candidates[g][-1] for g in groups}
    )

    def imatrix_imp(gid: str) -> float | None:
        if imatrix_groups and gid in imatrix_groups:
            return float(imatrix_groups[gid].get("importance_mean") or 0.0)
        return None

    def probe_fn(gid: str, q: str) -> dict[str, Any]:
        row = row_index.get((gid, q.upper()))
        if row is not None and row.get("kld_tail_1pct") is not None:
            return {
                "kld_mean": float(row.get("kld_mean", row.get("delta_kld") or 0.0)),
                "kld_tail_1pct": float(row["kld_tail_1pct"]),
                "n_tokens": row.get("n_tokens"),
            }
        if objective != "mean":
            have = "a proxy row with no measured tail" if row is not None else "no row at all"
            raise ValueError(
                f"tail-KLD objective needs a measured kld_tail_1pct for "
                f"({gid}, {q}) but there is {have}. Run step 12 with "
                f"--mode llama (trial quant + perplexity on the search split) "
                f"so every column is measured; proxy estimates are refused."
            )
        if row is not None:
            return {
                "kld_mean": float(row.get("kld_mean", row.get("delta_kld") or 0.0)),
                "kld_tail_1pct": None,
                "n_tokens": row.get("n_tokens"),
            }
        g = groups_t.get(gid) or {}
        mean = _proxy_delta_kld(g, tensors, q, imatrix_group_importance=imatrix_imp(gid))
        return {"kld_mean": mean, "kld_tail_1pct": None, "n_tokens": None}

    # Sizes: exact trial-file bytes where the column was measured
    # (row "bytes_measured" from step-12 llama probes), BYTES_PER_ELEM
    # estimate otherwise. Measured values are sanity-checked against the
    # estimate — dual threshold (relative AND absolute) so tiny groups
    # don't trip on fixed metadata overhead; trips abort loudly.
    size_bytes: dict[tuple[str, str], int] = {}
    size_measured: dict[tuple[str, str], bool] = {}
    for gid in groups:
        n_elem = _group_n_elements(groups_t[gid], tensors)
        for q in candidates[gid]:
            row = row_index.get((gid, q.upper()))
            got = row.get("bytes_measured") if row else None
            if got is not None:
                est = estimate_group_nbytes(n_elem, q)
                dev = abs(int(got) - est)
                if dev > SIZE_SANITY_ABS and dev / max(est, 1) > SIZE_SANITY_REL:
                    raise ValueError(
                        f"Measured size for ({gid}, {q}) is {int(got)} bytes "
                        f"vs {est} estimated "
                        f"({dev / max(est, 1):.1%} off, threshold "
                        f"{SIZE_SANITY_REL:.0%}/{SIZE_SANITY_ABS // 1024}KiB). "
                        f"Trial file corrupt or mis-mapped — refusing to solve."
                    )
                size_bytes[(gid, q)] = int(got)
                size_measured[(gid, q)] = True
            else:
                size_bytes[(gid, q)] = int(
                    estimate_group_nbytes(n_elem, q) * size_margin
                )
                size_measured[(gid, q)] = False
    imatrix_scores = None
    if imatrix_groups:
        imatrix_scores = {
            gid: float(imatrix_groups[gid].get("importance_mean") or 0.0)
            for gid in groups
            if gid in imatrix_groups
        } or None

    result = run_column_generation(
        groups=groups, candidates=candidates, size_bytes=size_bytes,
        probe_fn=probe_fn, budget_bytes=solve_budget,
        imatrix_scores=imatrix_scores, lipschitz_L=lipschitz_L,
        delta_bins=delta_bins, mode=certificate_mode, batch_size=batch_size,
        floor_of=eff_floors, objective=objective,
    )
    result["start_type"] = start_type.upper()
    result["floors"] = eff_floors
    for e in result["cost_matrix"]["entries"]:
        e["measured"] = bool(size_measured.get((e["group"], e["type"]), False))
    result["tail_cap"] = tail_cap
    result["auto_cap"] = False
    result["pass1_mean_kld"] = None
    result["pass1_tail_kld"] = None
    result["cap_removed_columns"] = cap_removed
    result["size_margin"] = size_margin
    result["fixed_groups"] = {
        gid: {"bytes": b, "type": "source", "kld_tail": 0.0, "kld_mean": 0.0}
        for gid, b in sorted(fixed_bytes.items())
    }
    result["fixed_bytes_total"] = fixed_total
    result["kept_bytes_total"] = kept_bytes_total
    result["file_overhead_bytes"] = overhead
    result["total_bytes"] = (
        int(result["total_bytes"]) + fixed_total + kept_bytes_total + overhead
    )
    result["meets_budget"] = result["total_bytes"] <= budget_bytes
    return result


def dp_mckp_optimize(
    *,
    catalog: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    budget_bytes: int,
    start_type: str = "Q6_K",
    pins: dict[str, str] | None = None,
    use_pins: bool = True,
    imatrix_groups: dict[str, Any] | None = None,
    lipschitz_L: float | None = None,
    certificate_mode: str = "bounded",
    delta_bins: int = 64,
    batch_size: int = 1,
    objective: str = "tail",
    tail_cap: float | None = None,
    auto_cap: bool = True,
    size_margin: float = 1.0,
    fixed_groups: frozenset[str] | set[str] | None = None,
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Optimize via column generation + DP MCKP; see _dp_mckp_optimize_once.

    Under the mean objective with ``tail_cap`` unset and ``auto_cap`` on
    (the default), a two-pass guardrail runs automatically: pass 1 solves
    mean-only, T* is the worst per-group measured P99 inside the pass-1
    allocation, and pass 2 re-solves the mean subject to every group
    P99 <= T*. Pass 2 is provably feasible (the pass-1 allocation itself
    satisfies the cap). An explicit ``tail_cap`` overrides (manual single
    pass); the tail objective always runs a single pass. Auto-cap needs
    measured tails — proxy tables raise loudly.
    """
    kwargs: dict[str, Any] = dict(
        catalog=catalog, sensitivity_rows=sensitivity_rows,
        budget_bytes=budget_bytes, start_type=start_type, pins=pins,
        use_pins=use_pins, imatrix_groups=imatrix_groups,
        lipschitz_L=lipschitz_L, certificate_mode=certificate_mode,
        delta_bins=delta_bins, batch_size=batch_size, objective=objective,
        size_margin=size_margin, fixed_groups=fixed_groups,
        file_overhead_bytes=file_overhead_bytes,
    )
    if tail_cap is not None or objective != "mean" or not auto_cap:
        return _dp_mckp_optimize_once(tail_cap=tail_cap, **kwargs)
    first = _dp_mckp_optimize_once(tail_cap=None, **kwargs)
    entries = {(e["group"], e["type"]): e for e in first["cost_matrix"]["entries"]}
    worst = 0.0
    for gid, q in first["allocation"].items():
        t = entries[(gid, q)].get("kld_tail")
        if t is None:
            raise ValueError(
                "auto-cap needs a measured kld_tail_1pct for every group in "
                f"the pass-1 allocation, but ({gid}, {q}) has none. Run "
                "step 12 with --mode llama so every column is measured."
            )
        worst = max(worst, float(t))
    second = _dp_mckp_optimize_once(tail_cap=worst, **kwargs)
    second["auto_cap"] = True
    second["pass1_mean_kld"] = first["total_mean_kld"]
    second["pass1_tail_kld"] = first["total_tail_kld"]
    return second


def _yaml_escape(s: str) -> str:
    if any(c in s for c in ":#{}[]|&*!?>'%@`,"):
        return json.dumps(s)
    return s


def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _yaml_escape(v)
    return str(v)


def _render_yaml_lines(obj: Any, indent: int) -> list[str]:
    """Minimal YAML emitter for additive recipe sections (Spec 2.6)."""
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return [f"{pad}{{}}"]
        lines: list[str] = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(_render_yaml_lines(v, indent + 2))
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return lines
    if isinstance(obj, list):
        if not obj:
            return [f"{pad}[]"]
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                sub = _render_yaml_lines(item, indent + 2)
                lines.append(f"{pad}- {sub[0].strip()}")
                lines.extend(sub[1:])
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(obj)}"]


def allocation_hash(assignments: dict[str, str]) -> str:
    blob = json.dumps(sorted(assignments.items()), separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def render_recipe_yaml(
    *,
    model_ref: str,
    hf_repo_id: str | None,
    gguf_sha256: str | None,
    imatrix_sha256: str | None,
    corpus_id: str | None,
    budget_bytes: int,
    base_type: str,
    assignments: dict[str, str],
    groups: dict[str, Any],
    estimated_bytes: int,
    predicted_delta_kld: float,
    method: str,
    extras: dict[str, Any] | None = None,
) -> str:
    overrides_lines = []
    for gid, q in sorted(assignments.items()):
        g = groups.get(gid) or {}
        regex = g.get("tensor_type_regex")
        # Prefer regex from first tensor names pattern stored in sensitivity — rebuild
        names = g.get("tensor_names") or []
        if names:
            from sensitivity import tensor_type_regex

            regex = tensor_type_regex(g)
        else:
            regex = gid
        overrides_lines.append(f'  "{regex}": {q.lower()}')

    lines = [
        "# OpenDynamicGGUF recipe — odg/recipe/v1",
        "schema: odg/recipe/v1",
        "model:",
        f"  source: {_yaml_escape(model_ref)}",
        f"  hf_repo_id: {_yaml_escape(hf_repo_id) if hf_repo_id else 'null'}",
        f"  gguf_sha256: \"{gguf_sha256 or ''}\"",
        "calibration:",
        f"  corpus_id: {_yaml_escape(corpus_id or 'odg-corpus-v1')}",
        f"  imatrix_sha256: \"{imatrix_sha256 or ''}\"",
        "  splits: { calib: 0.6, search: 0.2, heldout: 0.2, seed: 42 }",
        "budget:",
        f"  target_size_bytes: {budget_bytes}",
        f"  target_size_mb: {budget_bytes / (1024 * 1024):.2f}",
        f"base_type: {base_type.lower()}",
        "overrides:",
        *overrides_lines,
        "estimate:",
        f"  size_bytes: {estimated_bytes}",
        f"  size_mb: {estimated_bytes / (1024 * 1024):.2f}",
        f"  predicted_mean_delta_kld: {predicted_delta_kld:.6f}",
        f"  method: {method}",
        "validation: {}",
    ]
    if extras:
        lines.extend(_render_yaml_lines(extras, 0))
    lines.append("")
    return "\n".join(lines)


def render_tensor_type_file(
    assignments: dict[str, str],
    groups: dict[str, Any],
) -> str:
    """llama-quantize --tensor-type-file format: REGEX=TYPE per line."""
    from sensitivity import tensor_type_regex

    lines = ["# tensor-type-file generated by odg optimize"]
    for gid, q in sorted(assignments.items()):
        g = groups.get(gid) or {}
        regex = tensor_type_regex(g) if g.get("tensor_names") else gid
        lines.append(f"{regex}={q.lower()}")
    lines.append("")
    return "\n".join(lines)


def default_budget_bytes(
    catalog: dict[str, Any], *, ratio: float = 0.72,
    size_margin: float = 1.0,
) -> int:
    """
    Budget as a fraction of all-Q6_K size, but never below the pinned floor
    (embd Q8 + attn_v Q5 + rest Q3) so greedy can actually meet it.
    Both reference sizes carry the same margin, so ratios stay meaningful.
    """
    groups = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}
    q6 = {
        gid: "Q6_K"
        for gid, g in groups.items()
        if g.get("quantizable", True)
    }
    full = _estimate_total_bytes(q6, groups, tensors, size_margin)
    # Minimum achievable with default pins + Q3 elsewhere
    floor_assign = {}
    for gid, g in groups.items():
        if not g.get("quantizable", True):
            continue
        role = str(g.get("role") or "")
        floor_assign[gid] = DEFAULT_PINS.get(role, "Q3_K")
    floor_bytes = _estimate_total_bytes(floor_assign, groups, tensors, size_margin)
    target = int(full * ratio)
    # Leave a little headroom above the pin floor
    return max(1, max(target, int(floor_bytes * 1.02)))


DEFAULT_PARETO_RATIOS = (0.55, 0.65, 0.72, 0.80, 0.90, 1.0)


def _optimize_dp_mckp(
    *,
    model_ref: str,
    out_dir: Path,
    catalog: dict[str, Any],
    sensitivity: dict[str, Any],
    budget_bytes: int,
    hf_repo_id: str | None,
    gguf_sha256: str | None,
    imatrix_sha256: str | None,
    corpus_id: str | None,
    use_pins: bool,
    kld_objective: str,
    certificate_mode: str,
    lipschitz_L: float | None,
    jobs: int,
    pareto_ratios: list[float] | None,
    imatrix_groups: dict[str, Any] | None,
    tail_cap: float | None = None,
    auto_cap: bool = True,
    size_margin: float = 1.0,
    fixed_groups: frozenset[str] | set[str] | None = None,
    file_overhead_bytes: int = 0,
) -> OptimizeResult:
    """DP-MCKP path (Spec 2.3/2.4/2.6): colgen master + DP Pareto + certificate."""
    from dp_mckp import InfeasibleBudget, solve_mckp

    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto_dir = out_dir / "pareto"
    pareto_dir.mkdir(exist_ok=True)

    rows = sensitivity.get("rows") or []
    if not rows:
        raise ValueError("sensitivity table has no rows")
    if kld_objective not in ("tail_1pct", "mean"):
        raise ValueError(f"Unknown kld_objective: {kld_objective!r}")
    obj = "mean" if kld_objective == "mean" else "tail"

    log.append(f"1. Budget → {budget_bytes} bytes ({budget_bytes / (1024**2):.1f} MiB)")
    log.append(f"2. Sensitivity rows={len(rows)} method={sensitivity.get('method')}")
    log.append(
        f"3. DP-MCKP column generation from Q6_K with role pins "
        f"(objective={kld_objective}, certificate={certificate_mode}, "
         f"lipschitz={'auto' if lipschitz_L is None else lipschitz_L}, jobs={jobs}, "
         f"tail_cap={tail_cap}, auto_cap={auto_cap}, size_margin={size_margin}, "
         f"fixed_groups={sorted(fixed_groups) if fixed_groups else []})"
    )
    notes.append(
        f"jobs={jobs}: process-level probe parallelism (each probe is a "
        f"llama-quantize + llama-perplexity subprocess; OS-scheduled, "
        f"hardware-agnostic). Proxy-mode probes are in-process."
    )

    dp = dp_mckp_optimize(
        catalog=catalog,
        sensitivity_rows=rows,
        budget_bytes=budget_bytes,
        start_type="Q6_K",
        use_pins=use_pins,
        imatrix_groups=imatrix_groups,
        lipschitz_L=lipschitz_L,
        certificate_mode=certificate_mode,
        objective=obj,
        tail_cap=tail_cap,
        auto_cap=auto_cap,
        size_margin=size_margin,
        fixed_groups=fixed_groups,
        file_overhead_bytes=file_overhead_bytes,
    )
    alloc: dict[str, str] = dp["allocation"]
    fixed_info: dict[str, dict[str, Any]] = dp.get("fixed_groups", {})
    fixed_total: int = int(dp.get("fixed_bytes_total", 0))
    kept_total: int = int(dp.get("kept_bytes_total", 0))
    overhead: int = int(dp.get("file_overhead_bytes", 0))
    if fixed_info:
        notes.append(
            "Fixed groups stay at source precision (excluded from candidates "
            "and recipe.tt; export default applies, which keeps them): "
            + ", ".join(f"{g}={v['bytes']}" for g, v in sorted(fixed_info.items()))
            + f". Fixed total {fixed_total} bytes subtracted from every "
            "budget before solving, added back to every total."
        )
    cert = dp["certificate"]
    cost_matrix = dp["cost_matrix"]
    total_tail = dp["total_tail_kld"]
    total_mean = dp["total_mean_kld"]
    groups = catalog.get("groups") or {}
    method = "dp_mckp_colgen_v1"

    def _fmt(v: Any) -> str:
        return "None" if v is None else f"{float(v):.4f}"

    log.append(
        f"4. Primary recipe size={dp['total_bytes']} "
        f"tail={_fmt(total_tail)} mean={_fmt(total_mean)} "
        f"rounds={dp['rounds']} probed={cert['probed_columns']} "
        f"excluded={cert['excluded_columns']} lambda={cert['shadow_price_lambda']:.6g}"
    )

    entry_index = {(e["group"], e["type"]): e for e in cost_matrix["entries"]}
    ntok = next(
        (e["n_tokens"] for e in cost_matrix["entries"] if e.get("n_tokens") is not None),
        None,
    )
    kld_metric = {
        "objective": kld_objective,
        "reported": ["tail_1pct", "mean"],
        "n_tokens": ntok,
    }
    allocation_list = [
        {
            "group": g,
            "type": alloc[g],
            "bytes": entry_index[(g, alloc[g])]["bytes"],
            "kld_tail": entry_index[(g, alloc[g])]["kld_tail"],
        }
        for g in sorted(alloc)
    ] + [
        {
            "group": g,
            "type": "source",
            "bytes": info["bytes"],
            "kld_tail": 0.0,
            "fixed": True,
        }
        for g, info in sorted(fixed_info.items())
    ]
    totals = {"bytes": dp["total_bytes"], "kld_mean": total_mean, "kld_tail": total_tail}

    # Pareto: DP re-solves over probed columns only (free, no new probes;
    # the termination certificate strictly covers the primary budget).
    ratios = list(pareto_ratios) if pareto_ratios else list(DEFAULT_PARETO_RATIOS)
    q6_size = default_budget_bytes(catalog, ratio=1.0, size_margin=size_margin)
    pareto_targets = sorted({int(q6_size * r) for r in ratios} | {budget_bytes})
    p_groups = sorted(cost_matrix["groups"])
    p_cand = {
        g: sorted({e["type"] for e in cost_matrix["entries"] if e["group"] == g and e["probed"]})
        for g in p_groups
    }
    p_size = {(e["group"], e["type"]): int(e["bytes"]) for e in cost_matrix["entries"] if e["probed"]}
    p_cost = {
        (e["group"], e["type"]): float(e["kld_mean"] if obj == "mean" else e["kld_tail"])
        for e in cost_matrix["entries"]
        if e["probed"]
    }
    p_mean = {(e["group"], e["type"]): float(e["kld_mean"]) for e in cost_matrix["entries"] if e["probed"]}
    p_tail = {
        (e["group"], e["type"]): (None if e["kld_tail"] is None else float(e["kld_tail"]))
        for e in cost_matrix["entries"]
        if e["probed"]
    }
    pareto_paths: list[str] = []
    pareto_summary = []
    pareto_points = []
    fixed_total = int(dp.get("fixed_bytes_total", 0))
    fixed_list = [
        {
            "group": g,
            "type": "source",
            "bytes": info["bytes"],
            "kld_tail": 0.0,
            "fixed": True,
        }
        for g, info in sorted(dp.get("fixed_groups", {}).items())
    ]
    for i, b in enumerate(pareto_targets):
        b_adj = b - fixed_total - kept_total - overhead
        if b_adj < 0:
            pareto_summary.append({"budget_bytes": b, "feasible": False})
            pareto_points.append({
                "budget_bytes": b, "kld_tail": None,
                "allocation_hash": None, "feasible": False,
            })
            continue
        try:
            alt = solve_mckp(
                groups=p_groups, candidates=p_cand, size_bytes=p_size,
                cost_tail=p_cost, budget_bytes=b_adj,
            )
        except InfeasibleBudget:
            pareto_summary.append({"budget_bytes": b, "feasible": False})
            pareto_points.append({
                "budget_bytes": b, "kld_tail": None,
                "allocation_hash": None, "feasible": False,
            })
            continue
        alt_total = int(alt["total_bytes"]) + fixed_total + kept_total + overhead
        ahash = allocation_hash(alt["allocation"])
        name = f"pareto-{i:02d}-{b // 1024}k.yaml"
        alt_mean = sum(p_mean[(g, alt["allocation"][g])] for g in p_groups)
        alt_tails = [p_tail[(g, alt["allocation"][g])] for g in p_groups]
        alt_tail = sum(alt_tails) if all(t is not None for t in alt_tails) else None
        y = render_recipe_yaml(
            model_ref=model_ref,
            hf_repo_id=hf_repo_id,
            gguf_sha256=gguf_sha256,
            imatrix_sha256=imatrix_sha256,
            corpus_id=corpus_id,
            budget_bytes=b,
            base_type="Q6_K",
            assignments=alt["allocation"],
            groups=groups,
            estimated_bytes=alt_total,
            predicted_delta_kld=alt_mean,
            method=method,
            extras={
                "optimizer": "dp_mckp",
                "kld_metric": kld_metric,
                "totals": {"bytes": alt_total, "kld_mean": alt_mean, "kld_tail": alt_tail},
                "allocation": [
                    {"group": g, "type": alt["allocation"][g],
                     "bytes": p_size[(g, alt["allocation"][g])],
                     "kld_tail": p_tail[(g, alt["allocation"][g])]}
                    for g in p_groups
                ] + [dict(e) for e in fixed_list],
            },
        )
        p = pareto_dir / name
        p.write_text(y, encoding="utf-8")
        pareto_paths.append(str(p))
        pareto_summary.append({
            "path": str(p), "budget_bytes": b, "estimated_bytes": alt_total,
            "predicted_tail_kld": alt_tail, "predicted_mean_kld": alt_mean,
            "allocation_hash": ahash, "feasible": True,
        })
        pareto_points.append({
            "budget_bytes": b, "kld_tail": alt_tail,
            "allocation_hash": ahash, "feasible": True,
        })

    extras = {
        "optimizer": "dp_mckp",
        "kld_metric": kld_metric,
        "guardrail": {
            "mode": (
                "auto"
                if dp.get("auto_cap")
                else ("manual" if dp.get("tail_cap") is not None else "none")
            ),
            "tail_cap": dp.get("tail_cap"),
            "removed_columns": dp.get("cap_removed_columns", 0),
            "pass1_mean_kld": dp.get("pass1_mean_kld"),
            "pass1_tail_kld": dp.get("pass1_tail_kld"),
            "note": (
                "Two-pass auto-cap: pass 1 minimizes the additive mean, T* "
                "is the worst per-group measured P99 inside that allocation, "
                "pass 2 re-minimizes the mean subject to every group "
                "P99 <= T*. Percentiles don't add, so the summed P99 is "
                "reported, never optimized; the certificate covers the "
                "restricted problem (Pareto points inherit it)."
                if dp.get("auto_cap")
                else (
                    "Per-group P99 ceiling: candidates violating the cap are "
                    "deleted pre-DP under either objective; the certificate "
                    "covers the restricted problem (Pareto points inherit it)."
                    if dp.get("tail_cap") is not None
                    else "No P99 guardrail."
                )
            ),
        },
        "size_margin": size_margin,
        "cost_matrix": cost_matrix,
        "allocation": allocation_list,
        "totals": totals,
        "certificate": cert,
        "pareto": pareto_points,
        "discretization": {"bin_bytes": dp["bin_bytes"], "rounding": dp["rounding"]},
    }
    recipe_yaml = render_recipe_yaml(
        model_ref=model_ref,
        hf_repo_id=hf_repo_id,
        gguf_sha256=gguf_sha256,
        imatrix_sha256=imatrix_sha256,
        corpus_id=corpus_id,
        budget_bytes=budget_bytes,
        base_type="Q6_K",
        assignments=alloc,
        groups=groups,
        estimated_bytes=dp["total_bytes"],
        predicted_delta_kld=total_mean,
        method=method,
        extras=extras,
    )
    recipe_path = out_dir / "recipe.yaml"
    recipe_path.write_text(recipe_yaml, encoding="utf-8")

    tt = render_tensor_type_file(alloc, groups)
    tt_path = out_dir / "recipe.tt"
    tt_path.write_text(tt, encoding="utf-8")

    log.append(f"5. Wrote recipe.yaml + recipe.tt + {len(pareto_paths)} Pareto recipes")
    notes.append(
        "Allocation is DP-optimal over the probed cost matrix (tail-KLD "
        "objective unless --kld-objective mean). Pareto points re-solve the "
        "DP over probed columns only — no new probes; the termination "
        "certificate strictly covers the primary budget."
    )
    notes.append(
        "Assignments from sensitivity table (proxy or measured). "
        "Export with Step 14 using recipe.tt."
    )
    if dp["total_bytes"] > budget_bytes:
        notes.append("Primary allocation exceeds budget — raise the budget.")

    (out_dir / "optimize_manifest.json").write_text(
        json.dumps(
            {
                "primary": {
                    "assignments": alloc,
                    "estimated_bytes": dp["total_bytes"],
                    "predicted_delta_kld": total_mean,
                    "total_tail_kld": total_tail,
                    "total_mean_kld": total_mean,
                    "meets_budget": dp["total_bytes"] <= budget_bytes,
                    "optimizer": "dp_mckp",
                    "kld_objective": kld_objective,
                    "tail_cap": dp.get("tail_cap"),
                    "auto_cap": dp.get("auto_cap", False),
                    "pass1_mean_kld": dp.get("pass1_mean_kld"),
                    "pass1_tail_kld": dp.get("pass1_tail_kld"),
        "size_margin": size_margin,
        "budget": {
            "target_bytes": budget_bytes,
            "file_overhead_bytes": overhead,
            "kept_nonquant_bytes": kept_total,
            "fixed_bytes": fixed_total,
            "note": (
                "Non-DP bytes (fixed groups, kept non-quantizable "
                "tensors, GGUF header/metadata overhead) are subtracted "
                "from the target before solving and added back to every "
                "total, so the budget means final file size on disk."
            ),
        },
                    "fixed_groups": dp.get("fixed_groups", {}),
                    "fixed_bytes_total": dp.get("fixed_bytes_total", 0),
                    "kept_bytes_total": dp.get("kept_bytes_total", 0),
                    "file_overhead_bytes": dp.get("file_overhead_bytes", 0),
                    "certificate": cert,
                    "rounds": dp["rounds"],
                    "floors": dp.get("floors"),
                },
                "pareto": pareto_summary,
                "budget_bytes": budget_bytes,
                "jobs": jobs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return OptimizeResult(
        model_ref=model_ref,
        method=method,
        budget_bytes=budget_bytes,
        estimated_bytes=dp["total_bytes"],
        predicted_delta_kld=total_mean,
        n_groups=len(alloc),
        recipe_path=str(recipe_path),
        tensor_type_file=str(tt_path),
        pareto_paths=pareto_paths,
        assignments=alloc,
        steps_log=log,
        notes=notes,
        optimizer="dp_mckp",
        kld_objective=kld_objective,
        total_tail_kld=total_tail,
        total_mean_kld=total_mean,
        certificate=cert,
        cost_matrix=cost_matrix,
        tail_cap=dp.get("tail_cap"),
        auto_cap=dp.get("auto_cap", False),
        pass1_mean_kld=dp.get("pass1_mean_kld"),
        pass1_tail_kld=dp.get("pass1_tail_kld"),
        cap_removed_columns=int(dp.get("cap_removed_columns", 0)),
        fixed_groups=dp.get("fixed_groups", {}),
        fixed_bytes_total=int(dp.get("fixed_bytes_total", 0)),
        kept_bytes_total=int(dp.get("kept_bytes_total", 0)),
        file_overhead_bytes=int(dp.get("file_overhead_bytes", 0)),
    )


def optimize_recipes(
    *,
    model_ref: str,
    out_dir: Path,
    catalog: dict[str, Any],
    sensitivity: dict[str, Any],
    budget_bytes: int | None = None,
    budget_ratio: float = 0.72,
    hf_repo_id: str | None = None,
    gguf_sha256: str | None = None,
    imatrix_sha256: str | None = None,
    corpus_id: str | None = None,
    use_pins: bool = True,
    optimizer: str = "dp_mckp",
    kld_objective: str = "mean",
    certificate_mode: str = "bounded",
    lipschitz_L: float | None = None,
    jobs: int = 1,
    pareto_ratios: list[float] | None = None,
    imatrix_groups: dict[str, Any] | None = None,
    tail_cap: float | None = None,
    auto_cap: bool = True,
    size_margin: float = 1.0,
    fixed_groups: frozenset[str] | set[str] | None = None,
    file_overhead_bytes: int = 0,
) -> OptimizeResult:
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto_dir = out_dir / "pareto"
    pareto_dir.mkdir(exist_ok=True)

    rows = sensitivity.get("rows") or []
    if not rows:
        raise ValueError("sensitivity table has no rows")

    if budget_bytes is None:
        budget_bytes = default_budget_bytes(
            catalog, ratio=budget_ratio, size_margin=size_margin
        )
        log.append(
            f"1. Budget from ratio={budget_ratio:.2f} → {budget_bytes} bytes "
            f"({budget_bytes / (1024**2):.1f} MiB)"
        )
    else:
        log.append(
            f"1. Budget fixed → {budget_bytes} bytes "
            f"({budget_bytes / (1024**2):.1f} MiB)"
        )

    if optimizer == "dp_mckp":
        return _optimize_dp_mckp(
            model_ref=model_ref,
            out_dir=out_dir,
            catalog=catalog,
            sensitivity=sensitivity,
            budget_bytes=budget_bytes,
            hf_repo_id=hf_repo_id,
            gguf_sha256=gguf_sha256,
            imatrix_sha256=imatrix_sha256,
            corpus_id=corpus_id,
            use_pins=use_pins,
            kld_objective=kld_objective,
            certificate_mode=certificate_mode,
            lipschitz_L=lipschitz_L,
            jobs=jobs,
            pareto_ratios=pareto_ratios,
            imatrix_groups=imatrix_groups,
            tail_cap=tail_cap,
            auto_cap=auto_cap,
            size_margin=size_margin,
            fixed_groups=fixed_groups,
            file_overhead_bytes=file_overhead_bytes,
        )
    if optimizer != "greedy":
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    log.append(f"2. Sensitivity rows={len(rows)} method={sensitivity.get('method')}")
    log.append("3. Greedy downgrade from Q6_K with role pins")

    primary = greedy_optimize(
        catalog=catalog,
        sensitivity_rows=rows,
        budget_bytes=budget_bytes,
        start_type="Q6_K",
        use_pins=use_pins,
        size_margin=size_margin,
        fixed_groups=fixed_groups,
        file_overhead_bytes=file_overhead_bytes,
    )
    log.append(
        f"4. Primary recipe size={primary['estimated_bytes']} "
        f"meets_budget={primary['meets_budget']} "
        f"pred_ΔKLD={primary['predicted_delta_kld']:.4f} "
        f"steps={len(primary['history'])}"
    )

    groups = catalog.get("groups") or {}
    method = "greedy_knapsack_v1"

    recipe_yaml = render_recipe_yaml(
        model_ref=model_ref,
        hf_repo_id=hf_repo_id,
        gguf_sha256=gguf_sha256,
        imatrix_sha256=imatrix_sha256,
        corpus_id=corpus_id,
        budget_bytes=budget_bytes,
        base_type="Q6_K",
        assignments=primary["assignments"],
        groups=groups,
        estimated_bytes=primary["estimated_bytes"],
        predicted_delta_kld=primary["predicted_delta_kld"],
        method=method,
    )
    recipe_path = out_dir / "recipe.yaml"
    recipe_path.write_text(recipe_yaml, encoding="utf-8")

    tt = render_tensor_type_file(primary["assignments"], groups)
    tt_path = out_dir / "recipe.tt"
    tt_path.write_text(tt, encoding="utf-8")

    # Pareto: optimize under several budgets
    q6_size = default_budget_bytes(catalog, ratio=1.0)
    pareto_targets = sorted(
        {
            int(q6_size * r)
            for r in DEFAULT_PARETO_RATIOS
        }
        | {budget_bytes}
    )
    pareto_paths: list[str] = []
    pareto_summary = []
    for i, b in enumerate(pareto_targets):
        alt = greedy_optimize(
            catalog=catalog,
            sensitivity_rows=rows,
            budget_bytes=b,
            start_type="Q6_K",
            use_pins=use_pins,
            fixed_groups=fixed_groups,
            file_overhead_bytes=file_overhead_bytes,
        )
        name = f"pareto-{i:02d}-{b // 1024}k.yaml"
        y = render_recipe_yaml(
            model_ref=model_ref,
            hf_repo_id=hf_repo_id,
            gguf_sha256=gguf_sha256,
            imatrix_sha256=imatrix_sha256,
            corpus_id=corpus_id,
            budget_bytes=b,
            base_type="Q6_K",
            assignments=alt["assignments"],
            groups=groups,
            estimated_bytes=alt["estimated_bytes"],
            predicted_delta_kld=alt["predicted_delta_kld"],
            method=method,
        )
        p = pareto_dir / name
        p.write_text(y, encoding="utf-8")
        pareto_paths.append(str(p))
        pareto_summary.append(
            {
                "path": str(p),
                "budget_bytes": b,
                "estimated_bytes": alt["estimated_bytes"],
                "predicted_delta_kld": alt["predicted_delta_kld"],
                "meets_budget": alt["meets_budget"],
            }
        )

    log.append(f"5. Wrote recipe.yaml + recipe.tt + {len(pareto_paths)} Pareto recipes")
    notes.append(
        "Assignments from sensitivity table (proxy or measured). "
        "Export with Step 14 using recipe.tt."
    )
    if not primary["meets_budget"]:
        notes.append(
            "Could not fully meet budget (hit pin floors). "
            "Relax pins or raise --budget-mb."
        )

    (out_dir / "optimize_manifest.json").write_text(
        json.dumps(
            {
                "primary": {
                    "assignments": primary["assignments"],
                    "estimated_bytes": primary["estimated_bytes"],
                    "predicted_delta_kld": primary["predicted_delta_kld"],
                    "meets_budget": primary["meets_budget"],
                    "history": primary["history"],
                },
                "pareto": pareto_summary,
                "budget_bytes": budget_bytes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return OptimizeResult(
        model_ref=model_ref,
        method=method,
        budget_bytes=budget_bytes,
        estimated_bytes=primary["estimated_bytes"],
        predicted_delta_kld=primary["predicted_delta_kld"],
        n_groups=len(primary["assignments"]),
        recipe_path=str(recipe_path),
        tensor_type_file=str(tt_path),
        pareto_paths=pareto_paths,
        assignments=primary["assignments"],
        steps_log=log,
        notes=notes,
    )
