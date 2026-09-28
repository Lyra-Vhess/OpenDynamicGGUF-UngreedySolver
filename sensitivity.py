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

# Quant type → multiplier on base KLD (lower bits → higher KLD).
# F32/F16 sit above the baseline: near-zero predicted delta (a measured
# column replaces the proxy wherever pricing selects it).
_QUANT_KLD_MULT: dict[str, float] = {
    "Q2_K": 3.5,
    "Q3_K": 2.2,
    "Q4_K": 1.0,
    "Q5_K": 0.55,
    "Q6_K": 0.25,
    "Q8_0": 0.08,
    "F16": 0.03,
    "F32": 0.01,
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


def group_ladder(
    group: dict[str, Any],
    *,
    start_type: str = BASELINE_TYPE,
) -> list[str]:
    """Uniform candidate ladder for a group, high precision first.

    No role floors: every group gets the full LADDER (F32 ceiling down to
    Q2). Measurement is gated by column-generation pricing, not by the
    ladder — the ladder is the candidate universe, probing is priced.
    """
    from optimizer import LADDER

    return list(LADDER)


def default_probe_grid(
    catalog: dict[str, Any],
    profile_types: list[str],
    *,
    start_type: str = BASELINE_TYPE,
) -> list[str]:
    """Default grid: profile grid plus every ladder type.

    The grid is the candidate universe for pricing, not the probe bill:
    lazy probing measures floor columns first and prices the rest, so a
    wide grid costs nothing by itself. Explicit --probe-types narrows the
    universe instead (DP firewall stays as backstop). Ordered by ladder,
    high first.
    """
    from optimizer import LADDER

    want = {t.upper() for t in profile_types}
    for _, g, _ in _qualifying_groups(catalog):
        want.update(group_ladder(g, start_type=start_type))
    order = {q: i for i, q in enumerate(LADDER)}
    return sorted(want, key=lambda q: order.get(q, len(LADDER)))


def group_probe_grid(
    group: dict[str, Any],
    grid: list[str],
    *,
    start_type: str = BASELINE_TYPE,
) -> list[str]:
    """Intersect the global probe grid with the group's candidate ladder.

    The ladder is uniform (full LADDER for every group), so this keeps
    grid order and case-normalizes; universe narrowing happens only via
    explicit --probe-types.
    """
    ladder = group_ladder(group, start_type=start_type)
    return [q for q in grid if q.upper() in ladder]


def _check_fixed_groups(
    catalog: dict[str, Any], fixed_groups: frozenset[str] | None
) -> frozenset[str]:
    """Validate probe-time --fixed-groups (same semantics as optimize).

    Fixed groups are kept at source precision and accounted in the budget
    at optimize time — probing them is meaningless (llama-quantize may
    not even touch their tensors, tripping the probe-effect assert).
    Unknown names are rejected loudly (likely a typo).
    """
    fixed = frozenset(fixed_groups or ())
    if not fixed:
        return fixed
    known = set((catalog.get("groups") or {}))
    unknown = sorted(fixed - known)
    if unknown:
        raise ValueError(
            f"--fixed-groups names unknown groups: {unknown}. "
            f"Known: {sorted(known)}"
        )
    return fixed


def probe_groups_proxy(
    catalog: dict[str, Any],
    *,
    probe_types: list[str] | None = None,
    baseline_type: str = BASELINE_TYPE,
    imatrix_groups: dict[str, Any] | None = None,
    fixed_groups: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return sensitivity rows for all quantizable groups × probe types,
    with the grid intersected per group (see group_probe_grid).
    Fixed groups are skipped (accounted at optimize time instead).
    """
    fixed = _check_fixed_groups(catalog, fixed_groups)
    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    tensors = catalog.get("tensors") or {}
    rows: list[dict[str, Any]] = []
    grid_skipped = 0

    for gid, g, n_elem in _qualifying_groups(catalog):
        if gid in fixed:
            continue
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
                    "top_token_agree": min(1.0, max(0.0, 1.0 - 2.5 * delta_kld)),
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
    fixed_groups: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure (quantizable group, probe type) with real tools, where each
    group's types are the global grid intersected with the uniform candidate
    ladder (see group_probe_grid).
    Fixed groups are skipped (accounted at optimize time instead).

    Trial GGUF: everything at ``baseline_type`` except the group's tensors
    at the probe type. KL is measured vs the step-11 search-split base.
    Rows store *deltas* vs one all-baseline anchor run. Returns
    ``(rows, baseline_absolute)``. Any tool failure raises (hard error).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from llama_probe import measure_column

    fixed = _check_fixed_groups(catalog, fixed_groups)
    probe_types = probe_types or list(DEFAULT_PROBE_TYPES)
    tensors = catalog.get("tensors") or {}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Drop stale trial GGUFs (e.g. from a previous catalog's groups);
    # logs and .tt files are kept for audit. Current trials are rewritten.
    for stale in work.glob("trial-*.gguf"):
        stale.unlink(missing_ok=True)

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
        if gid in fixed:
            continue
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
                tag=_trial_tag(gid, q),
                group_tensors=list(g.get("tensor_names") or []),
                **log_ctx,
            ): (gid, q)
            for gid, g, q in targets
        }
        for fut in as_completed(futs):
            gid, q = futs[fut]
            measured[(gid, q)] = fut.result()  # raises on failure: hard error

    rows: list[dict[str, Any]] = []
    for gid, g, q in targets:
        rows.append(_llama_row_dict(
            gid=gid, g=g, q=q, m=measured[(gid, q)], base=base,
            n_elem=_group_n_elements(g, tensors), baseline_type=baseline_type,
        ))
    baseline_absolute = {
        "kld_mean": base["kld_mean"],
        "kld_tail_1pct": base["kld_tail_1pct"],
        # Columns skipped by the per-group grid (only when an explicit
        # --probe-types narrows the universe) — GPU probes not spent.
        "grid_skipped": grid_skipped,
        # Fixed groups skipped at probe time (accounted at optimize time).
        "fixed_skipped": sorted(fixed),
    }
    return rows, baseline_absolute

def _llama_row_dict(
    *,
    gid: str,
    g: dict[str, Any],
    q: str,
    m: dict[str, Any],
    base: dict[str, Any],
    n_elem: int,
    baseline_type: str,
) -> dict[str, Any]:
    """One sensitivity row: absolute trial metrics → deltas vs the anchor."""
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
    return {
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
        # Actual group payload bytes from the trial file's own
        # metadata (exact); None when the probe predates measurement.
        "bytes_measured": m.get("group_bytes_measured"),
        # 1-D tensors the probe legitimately skipped (llama.cpp
        # never quantizes flat tensors); empty in the normal case.
        "probe_exempt_tensors": m.get("probe_exempt_tensors", []),
        "top_token_agree": min(1.0, max(0.0, 1.0 - 2.5 * delta_kld)),
        "efficiency": score,
        "decision_hint": hint,
        "tensor_type_regex": tensor_type_regex(g),
        "method": "llama_probe",
        "split": "search",
    }


def imatrix_group_scores(
    imatrix_gguf: str | Path | None,
    catalog: dict[str, Any],
) -> dict[str, float] | None:
    """Per-group mean importance from a real imatrix.gguf (pricing scales).

    Aggregates ``reband.real_imatrix_scores`` (per-(role, layer)) over each
    group's tensors. Groups with no scored tensors are absent
    (``normalize_scales`` treats them as midpoint). Returns None when there
    is no file or no scores — pricing then runs neutral, which only widens
    bounds (more probes, never wrong exclusions). A present-but-corrupt
    file raises loudly: trials would fail on it anyway.
    """
    if imatrix_gguf is None or not Path(imatrix_gguf).is_file():
        return None
    from reband import real_imatrix_scores

    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    scores = real_imatrix_scores(str(imatrix_gguf), tensors)
    if not scores:
        return None
    out: dict[str, float] = {}
    for gid, g in groups.items():
        vals: list[float] = []
        for n in g.get("tensor_names") or []:
            t = tensors.get(n) or {}
            if t.get("layer") is None or t.get("role") is None:
                continue
            v = scores.get((str(t["role"]), int(t["layer"])))
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
        if vals:
            out[gid] = sum(vals) / len(vals)
    return out or None


def _trial_tag(gid: str, q: str) -> str:
    return f"{gid}-{q}".replace("@", "_").replace("/", "_")


#: Absolute-metric keys carried by the sidecar (everything downstream of a
#: trial file: KL metrics from the perplexity log + byte/exempt accounting
#: from trial metadata). The trial .gguf itself is redundant: a pure
#: function of (frozen model, .tt override, imatrix, baseline), all kept.
_SIDECAR_ABS_KEYS = (
    "kld_mean", "kld_tail_1pct", "kld_p999", "kld_max", "kld_median",
    "same_top_p", "perplexity", "n_tokens", "group_bytes_measured",
    "probe_exempt_tensors", "trial_tag",
)

SIDECAR_NAME = "probed.jsonl"


def _sidecar_path(work: str | Path) -> Path:
    return Path(work) / SIDECAR_NAME


def _sidecar_append(work: str | Path, record: dict[str, Any]) -> None:
    """Append one probe record (crash-safe: single line, flushed)."""
    path = _sidecar_path(work)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def _sidecar_load(work: str | Path) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    """Load cached probe records: ({(gid, q): abs}, n_ignored).

    A trailing run of unparsable lines is ignored (partial write from a
    killed process — that cell is remeasured); corruption mid-file raises
    loudly (append-only must never produce that).
    """
    path = _sidecar_path(work)
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    ignored = 0
    if not path.is_file():
        return cached, ignored
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return cached, ignored
    records: list[dict[str, Any] | None] = []
    for line in lines:
        if not line.strip():
            records.append(None)
            continue
        try:
            rec = json.loads(line)
            records.append(rec if isinstance(rec, dict) else None)
        except ValueError:
            records.append(None)
    first_bad: int | None = None
    for i, rec in enumerate(records):
        if rec is None or not isinstance(rec.get("gid"), str) or not isinstance(
            rec.get("q"), str
        ):
            first_bad = i
            break
    if first_bad is not None:
        for rec in records[first_bad:]:
            if rec is not None and isinstance(rec.get("gid"), str):
                raise ValueError(
                    f"Sidecar {path} has a good record after a corrupt "
                    f"line — append-only violated, refusing to guess."
                )
        ignored = len(records) - first_bad
        records = records[:first_bad]
    for rec in records:
        assert rec is not None
        cached[(rec["gid"], rec["q"])] = {
            k: rec.get(k) for k in _SIDECAR_ABS_KEYS
        }
    return cached, ignored


def _reparse_trial_abs(
    work: Path, tag: str, group_tensors: list[str],
    probe_type: str, baseline_type: str,
) -> dict[str, Any] | None:
    """Reparse one (trial .gguf + perplexity log) pair into absolute metrics.

    Same accounting as measure_column; None when the pair is
    missing/incomplete/unparsable (caller measures for real). Used only by
    the startup harvest — steady-state resume reads the sidecar.
    """
    from gguf_tensors import gguf_tensor_map
    from kld import parse_llama_perplexity_kl
    from llama_probe import assert_probe_applied

    trial = work / f"trial-{tag}.gguf"
    plog_path = work / f"trial-{tag}.perplexity.log"
    if not (trial.is_file() and plog_path.is_file()):
        return None
    try:
        plog = plog_path.read_text(encoding="utf-8")
        m = dict(parse_llama_perplexity_kl(plog))
        tmap = gguf_tensor_map(trial)["tensors"]
        missing = [n for n in group_tensors if n not in tmap]
        if missing:
            return None
        m["probe_exempt_tensors"] = assert_probe_applied(
            tmap, group_tensors, probe_type, baseline_type, tag=tag,
        )
        m["group_bytes_measured"] = int(
            sum(int(tmap[n]["nbytes"]) for n in group_tensors)
        )
        m["trial_tag"] = tag
        return m
    except (OSError, ValueError, KeyError):
        return None


def _harvest_known_trials(
    work: Path,
    items: list[tuple[str, str, list[str]]],
    baseline_type: str,
) -> tuple[list[tuple[tuple[str, str], dict[str, Any]]], int, int]:
    """Reparse orphan trial pairs for known (gid, q) into sidecar records.

    items: (gid, probe, group_tensors). Each successfully parsed pair is
    appended to the sidecar and its .gguf deleted (logs stay for audit).
    Returns (harvested [(key, abs)], n_deleted_ggufs, freed_bytes).
    Unparsable/missing pairs are left for normal measurement.
    """
    harvested: list[tuple[tuple[str, str], dict[str, Any]]] = []
    n_deleted = 0
    freed = 0
    for gid, q, tensors in items:
        tag = _trial_tag(gid, q)
        m = _reparse_trial_abs(work, tag, tensors, q, baseline_type)
        if m is None:
            continue
        _sidecar_append(work, {"gid": gid, "q": q, **{
            k: m.get(k) for k in _SIDECAR_ABS_KEYS
        }})
        harvested.append(((gid, q), m))
        trial = work / f"trial-{tag}.gguf"
        try:
            freed += trial.stat().st_size
            trial.unlink()
        except OSError:
            pass
        else:
            n_deleted += 1
    return harvested, n_deleted, freed


def _deleted_mb(n_deleted: int, freed: int) -> str:
    return f"{n_deleted} ggufs/{freed / (1024**2):.0f}MiB"


def probe_groups_lazy(
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
    fixed_groups: frozenset[str] | None = None,
    pricing_budget_bytes: int,
    lipschitz_L: float | None = None,
    certificate_mode: str = "bounded",
    kld_objective: str = "mean",
    imatrix_scores: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Lazy GPU probing: floor-first, then priced rounds (Spec 2.4 as wired).

    The candidate universe is the grid (default: full uniform ladder —
    wide costs nothing) intersected per group. Round 0 measures the
    all-baseline anchor plus every group's floor; then column-generation
    pricing selects attractive columns only, and only those touch the GPU.
    Unmeasured columns are bound-excluded with the termination certificate,
    exactly like step-13 selection. Resume is sidecar-based
    (``probed.jsonl``): each success appends one line, reruns skip cached
    cells, and orphan trial pairs are harvested into the sidecar on start
    (then their .ggufs deleted) — disk steady-state is ~1 in-flight trial.

    Returns ``(rows, baseline_absolute)`` shaped like probe_groups_llama,
    over probed columns only; baseline_absolute also carries the pricing
    record (reference budget, rounds, lambda history, excluded count).
    """
    from concurrent.futures import ThreadPoolExecutor

    from colgen import bitwidth, run_column_generation
    from llama_probe import measure_column
    from optimizer import LADDER

    fixed = _check_fixed_groups(catalog, fixed_groups)
    tensors = catalog.get("tensors") or {}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    universe = [t.upper() for t in (probe_types or list(LADDER))]
    groups: dict[str, dict[str, Any]] = {}
    group_tensors: dict[str, list[str]] = {}
    n_elem: dict[str, int] = {}
    for gid, g, ne in _qualifying_groups(catalog):
        if gid in fixed:
            continue
        kept = group_probe_grid(g, universe, start_type=baseline_type)
        if not kept:
            raise ValueError(
                f"Probe universe {universe} covers nothing on {gid}'s ladder "
                f"(role={g.get('role')}). Widen --probe-types."
            )
        groups[gid] = g
        group_tensors[gid] = list(g.get("tensor_names") or [])
        n_elem[gid] = ne
    if not groups:
        raise ValueError("No quantizable groups to probe.")

    log_ctx = {
        "model_gguf": model_gguf, "search_txt": search_txt,
        "kl_base_bin": kl_base_bin, "baseline_type": baseline_type,
        "work_dir": work, "llama_quantize": llama_quantize,
        "llama_perplexity": llama_perplexity, "imatrix": imatrix,
        "perplexity_args": perplexity_args,
    }
    candidates = {gid: list(group_probe_grid(
        groups[gid], universe, start_type=baseline_type,
    )) for gid in groups}

    # Sidecar resume: cached cells skip the GPU; orphan trial pairs from a
    # killed run are harvested into the sidecar (ggufs deleted after).
    abs_measured, sidecar_ignored = _sidecar_load(work)
    n_cached = len(abs_measured)
    harvest_items = [
        (gid, q, group_tensors[gid])
        for gid in groups for q in candidates[gid]
        if (gid, q) not in abs_measured
    ]
    harvested, n_deleted, freed = _harvest_known_trials(
        work, harvest_items, baseline_type,
    )
    for key, m_abs in harvested:
        abs_measured[key] = m_abs

    base_rec = abs_measured.get(("__anchor__", baseline_type))
    if base_rec is None:
        base = measure_column(
            group_regex=None, probe_type=baseline_type, tag="baseline",
            **log_ctx,
        )
        _sidecar_append(work, {"gid": "__anchor__", "q": baseline_type, **{
            k: base.get(k) for k in _SIDECAR_ABS_KEYS
        }})
        abs_measured[("__anchor__", baseline_type)] = base
    else:
        base = dict(base_rec)
    _resume_note = (
        f"sidecar cached={n_cached} harvested={len(harvested)} "
        f"({_deleted_mb(n_deleted, freed)} freed) "
        f"ignored_trailing={sidecar_ignored}"
    )

    def gpu_probe(gid: str, q: str) -> dict[str, Any]:
        key = (gid, q)
        m_abs = abs_measured.get(key)
        if m_abs is None:
            tag = _trial_tag(gid, q)
            # keep_trial=False: the trial .gguf is deleted right after its
            # bytes/exempt are extracted; the sidecar line is the resume
            # record, so disk stays at ~1 in-flight trial.
            m_abs = measure_column(
                group_regex=tensor_type_regex(groups[gid]),
                probe_type=q, tag=tag,
                group_tensors=group_tensors[gid], **log_ctx,
            )
            _sidecar_append(work, {"gid": gid, "q": q, **{
                k: m_abs.get(k) for k in _SIDECAR_ABS_KEYS
            }})
            abs_measured[key] = m_abs
        return {
            "kld_mean": float(m_abs["kld_mean"]) - float(base["kld_mean"]),
            "kld_tail_1pct": (
                float(m_abs["kld_tail_1pct"]) - float(base["kld_tail_1pct"])
            ),
            "n_tokens": None,
        }

    def gpu_batch(cols: list[tuple[str, str]]) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as ex:
            futs = [ex.submit(gpu_probe, g, q) for g, q in cols]
            return [f.result() for f in futs]  # raises on failure: hard error

    candidates = {gid: list(group_probe_grid(
        groups[gid], universe, start_type=baseline_type,
    )) for gid in groups}
    floor_of = {
        gid: min(candidates[gid], key=bitwidth) for gid in groups
    }
    size_bytes = {
        (gid, q): estimate_group_nbytes(n_elem[gid], q)
        for gid in groups for q in candidates[gid]
    }
    objective = "tail" if kld_objective == "tail_1pct" else "mean"
    colgen_res = run_column_generation(
        groups=sorted(groups), candidates=candidates, size_bytes=size_bytes,
        probe_fn=gpu_probe, budget_bytes=int(pricing_budget_bytes),
        imatrix_scores=imatrix_scores, lipschitz_L=lipschitz_L,
        mode=certificate_mode, floor_of=floor_of,
        batch_size=max(1, int(jobs)), objective=objective,
        batch_probe_fn=gpu_batch,
    )

    rows: list[dict[str, Any]] = []
    for gid in sorted(groups):
        for q in candidates[gid]:
            if (gid, q) not in abs_measured:
                continue  # bound-excluded: certified, never touched GPU
            rows.append(_llama_row_dict(
                gid=gid, g=groups[gid], q=q, m=abs_measured[(gid, q)],
                base=base, n_elem=n_elem[gid], baseline_type=baseline_type,
            ))
    total_cols = sum(len(candidates[g]) for g in groups)
    baseline_absolute = {
        "kld_mean": base["kld_mean"],
        "kld_tail_1pct": base["kld_tail_1pct"],
        "grid_skipped": total_cols - len(rows),
        "fixed_skipped": sorted(fixed),
        "lazy_probing": True,
        "pricing_budget_bytes": int(pricing_budget_bytes),
        "pricing_rounds": colgen_res["rounds"],
        "pricing_lambda_history": colgen_res["lambda_history"],
        "pricing_excluded_columns": colgen_res["certificate"]["excluded_columns"],
        "pricing_certificate_mode": certificate_mode,
        "pricing_resume_note": _resume_note,
        "pricing_sensitivity_source": (
            "imatrix" if imatrix_scores else "uniform"
        ),
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
    fixed_groups: frozenset[str] | None = None,
    # lazy-pricing inputs (llama mode only):
    pricing_budget_bytes: int | None = None,
    lipschitz_L: float | None = None,
    certificate_mode: str = "bounded",
    kld_objective: str = "mean",
) -> tuple[SensitivityResult, list[dict[str, Any]]]:
    """
    Write sensitivity.json (+ summary). Returns (result, rows).

    In llama mode the default is lazy probing (pricing decides what the
    GPU measures: anchor + floors first, then attractive columns only).
    ``certificate_mode="exhaustive"`` keeps the legacy full-universe
    measurement. ``pricing_budget_bytes`` is the intended solve budget
    for pricing (the tightness λ should assume; defaults to the derived
    pricing_reference_budget, which is the default-format intent — the
    solve budget is a hard limit and Pareto targets above it are dropped,
    so no solve is ever looser than intended; the CLI passes the derived
    reference so pricing sees real λ).
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
        log.append("3. mode=llama — lazy GPU probing (pricing-gated)")
        # Pricing scales from measured importance: real imatrix.gguf
        # preferred, proxy-JSON importance fallback, neutral otherwise
        # (neutral only widens bounds — more probes, never wrong ones).
        pricing_scores = imatrix_group_scores(imatrix_gguf, catalog)
        scores_source = "imatrix_gguf" if pricing_scores else None
        if scores_source is None and imatrix_proxy_path:
            proxy_groups = _load_imatrix_groups(Path(imatrix_proxy_path))
            if proxy_groups:
                pricing_scores = {
                    gid: float(info.get("importance_mean") or 0.0)
                    for gid, info in proxy_groups.items()
                    if isinstance(info, dict)
                } or None
                scores_source = "proxy" if pricing_scores else None
        log.append(
            f"3a. pricing scales from "
            f"{scores_source or 'uniform (no imatrix scores)'}"
        )
        lazy_kwargs: dict[str, Any] = dict(
            catalog=catalog,
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
            fixed_groups=fixed_groups,
        )
        if certificate_mode == "exhaustive":
            log.append("3b. certificate=exhaustive — measuring full universe")
            rows, baseline_absolute = probe_groups_llama(**lazy_kwargs)
        else:
            from optimizer import default_budget_bytes, pricing_reference_budget

            ref_budget = pricing_budget_bytes or pricing_reference_budget(
                catalog, intended_bytes=default_budget_bytes(catalog),
            )
            log.append(
                f"3b. pricing reference budget={ref_budget} bytes "
                f"({ref_budget / (1024**2):.1f} MiB), "
                f"certificate={certificate_mode}, objective={kld_objective}"
            )
            rows, baseline_absolute = probe_groups_lazy(
                **lazy_kwargs,
                pricing_budget_bytes=ref_budget,
                lipschitz_L=lipschitz_L,
                certificate_mode=certificate_mode,
                kld_objective=kld_objective,
                imatrix_scores=pricing_scores,
            )
            lam_hist = baseline_absolute.get("pricing_lambda_history") or []
            lam_last = f"{lam_hist[-1]:.3g}" if lam_hist else "n/a"
            log.append(
                f"3c. priced rounds={baseline_absolute['pricing_rounds']} "
                f"rows={len(rows)} "
                f"excluded={baseline_absolute['pricing_excluded_columns']} "
                f"lambda={lam_last}"
            )
        method = "llama_probe"
        resume_note = baseline_absolute.get("pricing_resume_note")
        if resume_note:
            log.append(f"3d. resume: {resume_note}")
        log.append(
            f"4. Measured groups={len({r['group_id'] for r in rows})} "
            f"rows={len(rows)} grid_skipped={baseline_absolute['grid_skipped']} "
            f"fixed_skipped={baseline_absolute['fixed_skipped']} "
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
        fixed_groups=fixed_groups,
    )
    n_probed_groups = len(
        [1 for _g in _qualifying_groups(catalog)
         if _g[0] not in _check_fixed_groups(catalog, fixed_groups)]
    )
    grid_skipped = n_probed_groups * len(probe_types) - len(rows)
    log.append(f"4b. Per-group grids skipped {grid_skipped} out-of-universe probes")
    if fixed_groups:
        log.append(f"4c. Fixed groups skipped at probe time: {sorted(fixed_groups)}")
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
