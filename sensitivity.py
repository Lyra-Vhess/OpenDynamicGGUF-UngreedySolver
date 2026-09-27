"""Step 12 — sensitivity probing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math
import re
import json
from pathlib import Path
from typing import Any, Literal


# --- from sensitivity/types.py ---
@dataclass
class SensitivityResult:
    model_ref: str
    method: str  # "llama_probe" | "proxy_from_features"
    gguf_sha256: str | None
    search_path: str | None
    n_groups_probed: int
    n_rows: int
    probe_types: list[str]
    baseline_type: str
    top_efficiency: list[dict[str, Any]]
    pinned_hints: list[dict[str, Any]]
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)

# --- from sensitivity/proxy.py ---
BYTES_PER_ELEM: dict[str, float] = {
    "BF16": 2.0,
    "F16": 2.0,
    "F32": 4.0,
    "Q8_0": 34.0 / 32.0,
    "Q6_K": 210.0 / 256.0,
    "Q5_K": 176.0 / 256.0,
    "Q4_K": 144.0 / 256.0,
    "Q3_K": 110.0 / 256.0,
    "Q2_K": 84.0 / 256.0,
}

# Default trial ladder (easy → hard groups try lower first in ranking, not here)
DEFAULT_PROBE_TYPES = ["Q3_K", "Q4_K", "Q5_K", "Q6_K"]
BASELINE_TYPE = "Q6_K"

# Role → base ΔKLD scale at Q4_K (heuristic)
_ROLE_KLD: dict[str, float] = {
    "attn_q": 0.035,
    "attn_k": 0.025,
    "attn_v": 0.045,
    "attn_o": 0.030,
    "ffn_gate": 0.012,
    "ffn_up": 0.010,
    "ffn_down": 0.022,
    "embedding": 0.055,
    "lm_head": 0.040,
    "other": 0.020,
}

_DEPTH_KLD: dict[str, float] = {
    "early": 0.95,
    "middle": 1.00,
    "late": 1.20,
    "global": 1.10,
}

# Quant type → multiplier on base KLD (lower bits → higher KLD)
_QUANT_KLD_MULT: dict[str, float] = {
    "Q2_K": 3.5,
    "Q3_K": 2.2,
    "Q4_K": 1.0,
    "Q5_K": 0.55,
    "Q6_K": 0.25,
    "Q8_0": 0.08,
}


def estimate_group_nbytes(n_elements: int, quant: str) -> int:
    bpe = BYTES_PER_ELEM.get(quant.upper())
    if bpe is None:
        raise KeyError(f"Unknown quant type for size estimate: {quant}")
    return int(math.ceil(n_elements * bpe))


def _group_n_elements(group: dict[str, Any], tensors: dict[str, Any]) -> int:
    total = 0
    for name in group.get("tensor_names") or []:
        t = tensors.get(name) or {}
        total += int(t.get("n_elements") or 0)
    return total


def _feature_hardness(group: dict[str, Any], tensors: dict[str, Any]) -> float:
    """Combine weight + activation + imatrix-like signals into [0, ~2]."""
    names = group.get("tensor_names") or []
    w_outs, a_abs, a_outs = [], [], []
    for n in names:
        t = tensors.get(n) or {}
        wf = t.get("weight_features") or {}
        af = t.get("activation_features") or {}
        if wf.get("outlier_ratio") is not None:
            w_outs.append(float(wf["outlier_ratio"]))
        if af.get("absmax") is not None:
            a_abs.append(float(af["absmax"]))
        if af.get("outlier_ratio") is not None:
            a_outs.append(float(af["outlier_ratio"]))
    # Prefer precomputed group features when present
    gwf = group.get("weight_features") or {}
    gaf = group.get("activation_features") or {}
    w_out = float(gwf.get("outlier_ratio_mean") or (sum(w_outs) / len(w_outs) if w_outs else 0.0))
    a_max = float(gaf.get("absmax") or (max(a_abs) if a_abs else 0.0))
    a_out = float(gaf.get("outlier_ratio") or (sum(a_outs) / len(a_outs) if a_outs else 0.0))
    wh = float(gwf.get("hardness") or 0.0)
    ah = float(gaf.get("hardness") or 0.0)
    return 0.4 * wh + 0.4 * ah + 20.0 * w_out + 0.05 * a_max + 10.0 * a_out


def _proxy_delta_kld(
    group: dict[str, Any],
    tensors: dict[str, Any],
    quant: str,
    *,
    imatrix_group_importance: float | None = None,
) -> float:
    role = str(group.get("role") or "other")
    depth = str(group.get("depth") or "global")
    base = _ROLE_KLD.get(role, 0.02) * _DEPTH_KLD.get(depth, 1.0)
    qmult = _QUANT_KLD_MULT.get(quant.upper(), 1.0)
    hard = _feature_hardness(group, tensors)
    # Normalize hardness roughly into a 0.5–2.0 multiplier
    hard_m = 0.5 + min(hard, 5.0) / 5.0 * 1.5
    imp_m = 1.0
    if imatrix_group_importance is not None:
        imp_m = 0.7 + 0.6 * float(imatrix_group_importance)
    return max(1e-6, base * qmult * hard_m * imp_m)


def tensor_type_regex(group: dict[str, Any]) -> str:
    """
    Build a llama-quantize --tensor-type regex covering the group's tensors.
    Example: blk.(0|1|2).ffn_up.weight → pattern on shared suffix.
    """
    names = group.get("tensor_names") or []
    if not names:
        return re.escape(str(group.get("group_id") or "unknown"))
    # Common case: blk.N.ROLE.weight
    m = re.match(r"blk\.(\d+)\.(.+)$", names[0])
    if m and all(re.match(r"blk\.\d+\." + re.escape(m.group(2)) + r"$", n) for n in names):
        layers = []
        suffix = m.group(2)
        for n in names:
            mm = re.match(r"blk\.(\d+)\.", n)
            if mm:
                layers.append(mm.group(1))
        layers_sorted = sorted(layers, key=int)
        return rf"blk\.({'|'.join(layers_sorted)})\.{re.escape(suffix)}"
    # Fallback: alternation of escaped full names
    return "(?:" + "|".join(re.escape(n) for n in names) + ")"


def _qualifying_groups(
    catalog: dict[str, Any],
) -> list[tuple[str, dict[str, Any], int]]:
    """(gid, group, n_elements) for quantizable non-empty groups, sorted."""
    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    out = []
    for gid, g in sorted(groups.items()):
        if not g.get("quantizable", True):
            continue
        n_elem = _group_n_elements(g, tensors)
        if n_elem <= 0:
            continue
        out.append((gid, g, n_elem))
    return out


def _pins_ladder(
    group: dict[str, Any],
    *,
    start_type: str = BASELINE_TYPE,
) -> list[str]:
    """Pins-only candidate ladder for a group, high precision first.

    Hints only ever *narrow* ladders at solve time, so this is a superset
    of the final candidate set. Groups whose floor sits above the start
    (e.g. embedding at Q8) keep just the floor.
    """
    from optimizer import DEFAULT_PINS, LADDER, _ladder_index

    pins = dict(DEFAULT_PINS)
    floor = pins.get(str(group.get("role") or ""), "Q2_K").upper()
    lo = _ladder_index(start_type.upper())
    hi = _ladder_index(floor)
    return [floor] if hi < lo else LADDER[lo : hi + 1]


def default_probe_grid(
    catalog: dict[str, Any],
    profile_types: list[str],
    *,
    start_type: str = BASELINE_TYPE,
) -> list[str]:
    """Default grid: profile grid plus every type any group's pins-only
    ladder needs.

    A bare profile grid (e.g. Q3–Q6) covers nothing on ladders entirely
    above the baseline (e.g. embedding pinned at Q8) — that used to fail
    loudly at solve time. Unioning in ladder types fixes the default while
    per-group filtering keeps it tight; explicit --probe-types bypasses
    this (DP firewall stays as backstop). Ordered by ladder, high first.
    """
    from optimizer import LADDER

    want = {t.upper() for t in profile_types}
    for _, g, _ in _qualifying_groups(catalog):
        want.update(_pins_ladder(g, start_type=start_type))
    order = {q: i for i, q in enumerate(LADDER)}
    return sorted(want, key=lambda q: order.get(q, len(LADDER)))


def group_probe_grid(
    group: dict[str, Any],
    grid: list[str],
    *,
    start_type: str = BASELINE_TYPE,
) -> list[str]:
    """Intersect the global probe grid with the group's candidate ladder.

    Pins only (no sensitivity-hint pass: hints only ever *narrow* ladders at
    solve time, so the pins-only grid is a superset — over-probing, never
    under-probing, and the DP firewall stays as backstop). Groups whose
    floor sits above the start (e.g. embedding at Q8) keep just the floor.
    Above-ladder types (e.g. Q8 for an unpinned group) are dropped: the DP
    could never choose them, so measuring them is pure GPU waste.
    """
    ladder = _pins_ladder(group, start_type=start_type)
    return [q for q in grid if q.upper() in ladder]


def probe_groups_proxy(
    catalog: dict[str, Any],
    *,
    probe_types: list[str] | None = None,
    baseline_type: str = BASELINE_TYPE,
    imatrix_groups: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Return sensitivity rows for all quantizable groups × probe types,
    with the grid intersected per group (see group_probe_grid).
    """
    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    tensors = catalog.get("tensors") or {}
    rows: list[dict[str, Any]] = []
    grid_skipped = 0

    for gid, g, n_elem in _qualifying_groups(catalog):
        kept = group_probe_grid(g, probe_types, start_type=baseline_type)
        grid_skipped += len(probe_types) - len(kept)
        if not kept:
            raise ValueError(
                f"Probe grid {probe_types} covers nothing on {gid}'s ladder "
                f"(role={g.get('role')}). Widen --probe-types."
            )
        base_bytes = estimate_group_nbytes(n_elem, baseline_type)
        imp = None
        if imatrix_groups and gid in imatrix_groups:
            imp = float(imatrix_groups[gid].get("importance_mean") or 0.0)

        for q in kept:
            q_bytes = estimate_group_nbytes(n_elem, q)
            delta_bytes = base_bytes - q_bytes  # positive = smaller
            # If probing higher than baseline, bytes_saved may be negative
            delta_kld = _proxy_delta_kld(g, tensors, q, imatrix_group_importance=imp)
            # If quant is higher precision than baseline, KLD should be near 0 vs baseline
            if BYTES_PER_ELEM.get(q.upper(), 99) >= BYTES_PER_ELEM.get(
                baseline_type.upper(), 1
            ):
                delta_kld *= 0.15
            eps = 1e-6
            score = (delta_bytes / max(delta_kld, eps)) if delta_bytes > 0 else 0.0
            hint = "compress" if score > 5e7 and delta_kld < 0.03 else (
                "pin_high" if delta_kld > 0.04 else "neutral"
            )
            rows.append(
                {
                    "group_id": gid,
                    "role": g.get("role"),
                    "depth": g.get("depth"),
                    "probe": q,
                    "baseline": baseline_type,
                    "n_elements": n_elem,
                    "n_tensors": g.get("n_tensors"),
                    "bytes_baseline": base_bytes,
                    "bytes_probe": q_bytes,
                    "delta_bytes": delta_bytes,
                    "delta_kld": delta_kld,
                    # kld_mean is a feature estimate; kld_tail_1pct is None
                    # (no fake tail: the DP tail objective refuses unmeasured
                    # columns loudly — run mode=llama for measured tails).
                    "kld_mean": delta_kld,
                    "kld_tail_1pct": None,
                    "n_tokens": None,
                    "top_token_agree": max(0.0, 1.0 - 2.5 * delta_kld),
                    "efficiency": score,
                    "decision_hint": hint,
                    "tensor_type_regex": tensor_type_regex(g),
                    "method": "proxy_from_features",
                    "split": "search",
                }
            )
    return rows

# --- from sensitivity/probe.py ---
Mode = Literal["auto", "llama", "proxy"]


def _load_imatrix_groups(proxy_path: Path | None) -> dict[str, Any] | None:
    if not proxy_path or not proxy_path.is_file():
        return None
    data = json.loads(proxy_path.read_text())
    return data.get("groups")


def probe_groups_llama(
    catalog: dict[str, Any],
    *,
    model_gguf: str | Path,
    search_txt: str | Path,
    kl_base_bin: str | Path,
    probe_types: list[str] | None = None,
    baseline_type: str = BASELINE_TYPE,
    work_dir: str | Path,
    jobs: int = 1,
    llama_quantize: str | Path | None = None,
    llama_perplexity: str | Path | None = None,
    imatrix: str | Path | None = None,
    perplexity_args: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure (quantizable group, probe type) with real tools, where each
    group's types are the global grid intersected with its own candidate
    ladder (see group_probe_grid) — above-ladder probes are skipped, not run.

    Trial GGUF: everything at ``baseline_type`` except the group's tensors
    at the probe type. KL is measured vs the step-11 search-split base.
    Rows store *deltas* vs one all-baseline anchor run. Returns
    ``(rows, baseline_absolute)``. Any tool failure raises (hard error).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from llama_probe import measure_column

    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    tensors = catalog.get("tensors") or {}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    log_ctx = {
        "model_gguf": model_gguf, "search_txt": search_txt,
        "kl_base_bin": kl_base_bin, "baseline_type": baseline_type,
        "work_dir": work, "llama_quantize": llama_quantize,
        "llama_perplexity": llama_perplexity, "imatrix": imatrix,
        "perplexity_args": perplexity_args,
    }
    base = measure_column(
        group_regex=None, probe_type=baseline_type, tag="baseline", **log_ctx
    )

    targets: list[tuple[str, dict[str, Any], str]] = []
    grid_skipped = 0
    for gid, g, n_elem in _qualifying_groups(catalog):
        kept = group_probe_grid(g, probe_types, start_type=baseline_type)
        grid_skipped += len(probe_types) - len(kept)
        if not kept:
            raise ValueError(
                f"Probe grid {probe_types} covers nothing on {gid}'s ladder "
                f"(role={g.get('role')}). Widen --probe-types."
            )
        for q in kept:
            targets.append((gid, g, q))

    measured: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as ex:
        futs = {
            ex.submit(
                measure_column,
                group_regex=tensor_type_regex(g),
                probe_type=q,
                tag=f"{gid}-{q}".replace("@", "_").replace("/", "_"),
                **log_ctx,
            ): (gid, q)
            for gid, g, q in targets
        }
        for fut in as_completed(futs):
            gid, q = futs[fut]
            measured[(gid, q)] = fut.result()  # raises on failure: hard error

    rows: list[dict[str, Any]] = []
    for gid, g, q in targets:
        m = measured[(gid, q)]
        n_elem = _group_n_elements(g, tensors)
        base_bytes = estimate_group_nbytes(n_elem, baseline_type)
        q_bytes = estimate_group_nbytes(n_elem, q)
        delta_bytes = base_bytes - q_bytes
        delta_kld = float(m["kld_mean"]) - float(base["kld_mean"])
        delta_tail = float(m["kld_tail_1pct"]) - float(base["kld_tail_1pct"])
        eps = 1e-6
        score = (delta_bytes / max(delta_kld, eps)) if delta_bytes > 0 else 0.0
        hint = "compress" if score > 5e7 and delta_kld < 0.03 else (
            "pin_high" if delta_kld > 0.04 else "neutral"
        )
        rows.append(
            {
                "group_id": gid,
                "role": g.get("role"),
                "depth": g.get("depth"),
                "probe": q,
                "baseline": baseline_type,
                "n_elements": n_elem,
                "n_tensors": g.get("n_tensors"),
                "bytes_baseline": base_bytes,
                "bytes_probe": q_bytes,
                "delta_bytes": delta_bytes,
                "delta_kld": delta_kld,
                "kld_mean": delta_kld,
                "kld_tail_1pct": delta_tail,
                "n_tokens": None,
                "kld_p999": m.get("kld_p999"),
                "same_top_p": m.get("same_top_p"),
                "perplexity": m.get("perplexity"),
                "top_token_agree": max(0.0, 1.0 - 2.5 * delta_kld),
                "efficiency": score,
                "decision_hint": hint,
                "tensor_type_regex": tensor_type_regex(g),
                "method": "llama_probe",
                "split": "search",
            }
        )
    baseline_absolute = {
        "kld_mean": base["kld_mean"],
        "kld_tail_1pct": base["kld_tail_1pct"],
        # Columns skipped by per-group grids (above-ladder types the DP
        # could never choose) — GPU probes not spent.
        "grid_skipped": grid_skipped,
    }
    return rows, baseline_absolute


def build_sensitivity_table(
    *,
    model_ref: str,
    out_dir: Path,
    catalog: dict[str, Any],
    gguf_sha256: str | None = None,
    search_path: str | Path | None = None,
    imatrix_proxy_path: str | Path | None = None,
    mode: Mode = "auto",
    probe_types: list[str] | None = None,
    baseline_type: str = BASELINE_TYPE,
    # llama-mode inputs (required when mode="llama"):
    model_gguf: str | Path | None = None,
    kl_base_bin: str | Path | None = None,
    trials_dir: str | Path | None = None,
    jobs: int = 1,
    llama_quantize: str | Path | None = None,
    llama_perplexity: str | Path | None = None,
    imatrix_gguf: str | Path | None = None,
    perplexity_args: list[str] | None = None,
) -> tuple[SensitivityResult, list[dict[str, Any]]]:
    """
    Write sensitivity.json (+ summary). Returns (result, rows).
    """
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    log.append(f"1. Probe types={probe_types} baseline={baseline_type}")
    log.append("2. Split=search only (heldout forbidden)")

    if mode == "llama":
        if model_gguf is None or not Path(model_gguf).is_file():
            raise RuntimeError(
                "llama probe mode needs the frozen GGUF (step 09). "
                f"Got model_gguf={model_gguf!r}."
            )
        if kl_base_bin is None or not Path(kl_base_bin).is_file():
            raise RuntimeError(
                "llama probe mode needs logits-search.bin (step 11 --mode llama). "
                f"Got kl_base_bin={kl_base_bin!r}."
            )
        if not search_path or not Path(search_path).is_file():
            raise RuntimeError("llama probe mode needs search.txt (step 07).")
        log.append("3. mode=llama — trial quant + perplexity per (group, type)")
        rows, baseline_absolute = probe_groups_llama(
            catalog,
            model_gguf=model_gguf,
            search_txt=search_path,
            kl_base_bin=kl_base_bin,
            probe_types=probe_types,
            baseline_type=baseline_type,
            work_dir=trials_dir or (out_dir / "trials"),
            jobs=jobs,
            llama_quantize=llama_quantize,
            llama_perplexity=llama_perplexity,
            imatrix=imatrix_gguf,
            perplexity_args=perplexity_args,
        )
        method = "llama_probe"
        log.append(
            f"4. Measured groups={len({r['group_id'] for r in rows})} "
            f"rows={len(rows)} grid_skipped={baseline_absolute['grid_skipped']} "
            f"baseline_mean={baseline_absolute['kld_mean']:.4f} "
            f"baseline_p99={baseline_absolute['kld_tail_1pct']:.4f}"
        )
        notes.append(
            "Measured per-group trial quants vs the step-11 search KL base; "
            "rows are deltas vs the all-baseline anchor run."
        )
        return _finish_table(
            model_ref=model_ref, out_dir=out_dir, catalog=catalog,
            gguf_sha256=gguf_sha256, search_path=search_path,
            probe_types=probe_types, baseline_type=baseline_type,
            method=method, log=log, notes=notes, rows=rows,
            baseline_absolute=baseline_absolute,
        )

    method = "proxy_from_features"
    if mode == "auto":
        log.append(
            "3. auto: llama probe tools/caches not wired — proxy_from_features"
        )
    else:
        log.append("3. mode=proxy — estimating ΔKLD/Δbytes from features")

    imatrix_groups = _load_imatrix_groups(
        Path(imatrix_proxy_path) if imatrix_proxy_path else None
    )
    if imatrix_groups:
        log.append(f"4. Loaded imatrix group importance ({len(imatrix_groups)} groups)")
    else:
        log.append("4. No imatrix proxy groups — features only")

    rows = probe_groups_proxy(
        catalog,
        probe_types=probe_types,
        baseline_type=baseline_type,
        imatrix_groups=imatrix_groups,
    )
    grid_skipped = (
        len(_qualifying_groups(catalog)) * len(probe_types) - len(rows)
    )
    log.append(f"4b. Per-group grids skipped {grid_skipped} above-ladder probes")
    return _finish_table(
        model_ref=model_ref, out_dir=out_dir, catalog=catalog,
        gguf_sha256=gguf_sha256, search_path=search_path,
        probe_types=probe_types, baseline_type=baseline_type,
        method=method, log=log, notes=notes, rows=rows,
        baseline_absolute=None,
        extra_notes=[
            "proxy_from_features estimates ΔKLD — not measured. "
            "Production needs llama-quantize trial + perplexity --kl-divergence on search.",
            "Held-out must not be used in this step.",
        ],
    )


def _finish_table(
    *,
    model_ref: str,
    out_dir: Path,
    catalog: dict[str, Any],
    gguf_sha256: str | None,
    search_path: str | Path | None,
    probe_types: list[str],
    baseline_type: str,
    method: str,
    log: list[str],
    notes: list[str],
    rows: list[dict[str, Any]],
    baseline_absolute: dict[str, Any] | None,
    extra_notes: list[str] | None = None,
) -> tuple[SensitivityResult, list[dict[str, Any]]]:
    """Rank rows, write sensitivity.json, build the result (shared tail)."""
    groups_probed = len({r["group_id"] for r in rows})
    log.append(f"5. Probed groups={groups_probed} rows={len(rows)}")

    # Rank by efficiency (bytes saved per unit KLD)
    by_eff = sorted(rows, key=lambda r: r["efficiency"], reverse=True)
    top_efficiency = by_eff[:10]

    # Pin hints: high ΔKLD at Q4_K
    pinned = [
        r
        for r in rows
        if r["probe"] == "Q4_K" and r["decision_hint"] == "pin_high"
    ]
    pinned = sorted(pinned, key=lambda r: r["delta_kld"], reverse=True)[:10]

    if extra_notes:
        notes.extend(extra_notes)
    else:
        notes.append("Held-out must not be used in this step.")

    table = {
        "model_ref": model_ref,
        "method": method,
        "gguf_sha256": gguf_sha256,
        "search_path": str(search_path) if search_path else None,
        "baseline_type": baseline_type,
        "probe_types": probe_types,
        "n_groups_probed": groups_probed,
        "n_rows": len(rows),
        "rows": rows,
        "top_efficiency": top_efficiency,
        "pinned_hints": pinned,
        "notes": notes,
    }
    if baseline_absolute is not None:
        table["baseline_kl"] = baseline_absolute
    (out_dir / "sensitivity.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )
    log.append("6. Wrote sensitivity.json")

    result = SensitivityResult(
        model_ref=model_ref,
        method=method,
        gguf_sha256=gguf_sha256,
        search_path=str(search_path) if search_path else None,
        n_groups_probed=groups_probed,
        n_rows=len(rows),
        probe_types=probe_types,
        baseline_type=baseline_type,
        top_efficiency=[
            {
                "group_id": r["group_id"],
                "probe": r["probe"],
                "delta_bytes": r["delta_bytes"],
                "delta_kld": r["delta_kld"],
                "efficiency": r["efficiency"],
                "decision_hint": r["decision_hint"],
            }
            for r in top_efficiency
        ],
        pinned_hints=[
            {
                "group_id": r["group_id"],
                "probe": r["probe"],
                "delta_kld": r["delta_kld"],
                "decision_hint": r["decision_hint"],
            }
            for r in pinned
        ],
        steps_log=log,
        notes=notes,
    )
    return result, rows
