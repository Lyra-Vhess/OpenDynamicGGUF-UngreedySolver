"""Imatrix-guided depth re-banding (runs between Steps 10 and 12).

Step 04 groups tensors as role@depth with depth from hard layer thirds
(early/middle/late). That is geometric, not measured: layer 13 and 14 can
behave nothing alike while layers 0..13 are treated identically.

This pass re-cuts each role's ordered layers into contiguous bands that
minimize within-band variance of per-layer imatrix importance — exact
1-D segmentation (Fisher-Jenks), microseconds for tens of layers. Same
band count per role as the thirds it replaces (3), so group and probe
counts stay ~constant; only boundaries move onto importance cliffs.

Inputs: tensor_catalog.json (any of catalog/weight_features/
activation_features flavor) + imatrix_proxy.json (per-tensor importance,
written in every build_imatrix path). Per-tensor data is untouched; group
membership, group aggregates (recomputed with the Step 06/08 aggregators),
depth labels, and catalog_sha256 are rewritten. Tensors without a layer
index (globals like embeddings) keep their existing groups.
"""

from __future__ import annotations

import copy
import math
from typing import Any

BANDS_PER_ROLE = 3
BAND_DEPTHS_3 = ("early", "middle", "late")

#: Minimum relative SS reduction for a role's optimal cuts to be adopted.
#: Below this the scores carry no band structure (e.g. all-constant) and
#: the role keeps its existing groups instead of cutting on noise.
MIN_SS_REDUCTION = 0.10


def _total_ss(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values)


def _read_f32_blob(path: str, offset: int, count: int) -> "list[float]":
    import numpy as np

    arr = np.fromfile(path, dtype="<f4", count=count, offset=offset)
    if arr.size != count:
        raise RuntimeError(
            f"short read on {path} at offset {offset}: "
            f"got {arr.size}/{count} float32"
        )
    return [float(v) for v in arr]


def real_imatrix_scores(
    imatrix_gguf: str,
    catalog_tensors: dict[str, Any],
) -> dict[tuple[str, int], float]:
    """Per-(role, layer) mean-squared-activation from a real imatrix.gguf.

    llama-imatrix writes ``<tensor>.in_sum2`` (sum of squared inputs per
    channel) + ``<tensor>.counts`` pairs. Score = sum(in_sum2)/sum(counts),
    the standard imatrix statistic, averaged over the role's tensors in
    each layer. Only F32 entries are read; anything else raises loudly.
    Model tensor names are recovered by stripping the final suffix; names
    absent from the catalog are ignored.
    """
    from pathlib import Path

    from gguf_tensors import gguf_tensor_map

    table = gguf_tensor_map(Path(imatrix_gguf))
    entries = table.get("tensors", table) if isinstance(table, dict) else {}
    base = int(table.get("data_offset", 0)) if isinstance(table, dict) else 0
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}

    def blob(entry: dict[str, Any]) -> list[float]:
        if str(entry.get("dtype", "")).upper() != "F32":
            raise RuntimeError(
                f"imatrix.gguf entry {entry.get('name')}: expected F32, "
                f"got {entry.get('dtype')}"
            )
        return _read_f32_blob(
            imatrix_gguf, base + int(entry.get("offset", 0)),
            int(entry.get("n_elements", 0)),
        )

    for full, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if full.endswith(".in_sum2"):
            sums[full[: -len(".in_sum2")]] = sum(blob(entry))
        elif full.endswith(".counts"):
            counts[full[: -len(".counts")]] = sum(blob(entry))
    acc: dict[tuple[str, int], list[float]] = {}
    for name, s in sums.items():
        c = counts.get(name)
        t = catalog_tensors.get(name)
        if not c or t is None or t.get("layer") is None:
            continue
        acc.setdefault((str(t.get("role")), int(t["layer"])), []).append(s / c)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def role_layer_scores(
    proxy_tensors: dict[str, Any],
    catalog_tensors: dict[str, Any],
) -> dict[tuple[str, int], float]:
    """Mean per-tensor importance per (role, layer).

    Layer/role come from the catalog (proxy entries lack layer); importance
    from the proxy (catalog lacks scores). Tensors missing on either side
    or without a layer index are skipped.
    """
    acc: dict[tuple[str, int], list[float]] = {}
    for name, info in proxy_tensors.items():
        t = catalog_tensors.get(name)
        if t is None:
            continue
        layer = t.get("layer")
        role = t.get("role")
        if layer is None or role is None:
            continue
        try:
            imp = float(info.get("importance"))
        except (TypeError, ValueError):
            continue
        acc.setdefault((str(role), int(layer)), []).append(imp)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def fisher_jenks_breaks(values: list[float], k: int) -> list[int]:
    """End indices (exclusive) of the first k-1 of k contiguous segments.

    Exact DP minimizing total within-segment sum of squares. ``values`` are
    in layer order; returns ``k - 1`` cut positions ``c`` with
    ``0 < c_1 < ... < c_{k-1} < n``. Degenerate cases: k <= 1 → [];
    n <= k → every layer its own segment start ([] needed... see below).
    """
    n = len(values)
    if k <= 1 or n == 0:
        return []
    k = min(k, n)
    if k <= 1:
        return []
    # Prefix sums for O(1) segment cost: SS(i, j) over values[i:j].
    pre = [0.0] * (n + 1)
    pre2 = [0.0] * (n + 1)
    for i, v in enumerate(values):
        pre[i + 1] = pre[i] + v
        pre2[i + 1] = pre2[i] + v * v

    def cost(i: int, j: int) -> float:
        m = j - i
        if m <= 0:
            return 0.0
        s = pre[j] - pre[i]
        s2 = pre2[j] - pre2[i]
        return s2 - s * s / m

    INF = math.inf
    # dp[s][j] = min cost of s segments covering values[:j]
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    cut: list[list[int]] = [[-1] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for s in range(1, k + 1):
        for j in range(s, n + 1):
            best, bj = INF, -1
            for i in range(s - 1, j):
                v = dp[s - 1][i] + cost(i, j)
                if v < best:
                    best, bj = v, i
            dp[s][j] = best
            cut[s][j] = bj
    breaks: list[int] = []
    j = n
    for s in range(k, 1, -1):
        j = cut[s][j]
        breaks.append(j)
    return sorted(breaks)


def band_index(layer_pos: int, breaks: list[int]) -> int:
    """Which band a layer-order position falls in given cut positions."""
    for b, c in enumerate(breaks):
        if layer_pos < c:
            return b
    return len(breaks)


def _is_flat_tensor(t: dict[str, Any]) -> bool:
    """1-D (or scalar) tensors llama.cpp never quantizes, by design.

    Same rule as ``llama_probe.assert_probe_applied``: at most one dim
    above 1. Such tensors must never form a probed group on their own —
    the probe would measure a no-op and record a bogus zero-delta row.
    """
    shape = t.get("shape")
    if shape is None:
        raise ValueError(
            f"Per-tensor grouping needs tensor shapes; {t.get('name')!r} "
            "has none. Refusing instead of guessing flat vs quantizable."
        )
    return sum(1 for d in shape if d and d > 1) <= 1


def _rebuild_groups(
    tensors: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild group records from tensor membership (group_id on tensors).

    Shared by the banded path and the per-tensor explode path so group
    records (aggregates, byte totals, quantizable flag) are computed one
    way.
    """
    from activation_features import aggregate_activation_group
    from weight_features import aggregate_group_features

    members: dict[str, list[str]] = {}
    for name, t in tensors.items():
        members.setdefault(str(t.get("group_id")), []).append(name)
    groups: dict[str, Any] = {}
    for gid in sorted(members):
        names = sorted(members[gid])
        m0 = tensors[names[0]]
        wfeats = [
            tensors[n].get("weight_features")
            for n in names
            if tensors[n].get("weight_features")
        ]
        afeats = [
            tensors[n].get("activation_features")
            for n in names
            if tensors[n].get("activation_features")
        ]
        g: dict[str, Any] = {
            "group_id": gid,
            "role": m0.get("role"),
            "depth": m0.get("depth"),
            "quantizable": any(
                bool(tensors[n].get("quantizable", True)) for n in names
            ),
            "n_tensors": len(names),
            "total_nbytes": sum(
                int(tensors[n].get("nbytes") or 0) for n in names
            ),
            "tensor_names": names,
        }
        if wfeats:
            g["weight_features"] = aggregate_group_features(wfeats)
        if afeats:
            g["activation_features"] = aggregate_activation_group(afeats)
        groups[gid] = g
    return groups


def explode_per_tensor(
    catalog: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Explode every group into single-tensor groups (gold-run grouping).

    Group id = tensor name; role/depth ride along for display and
    imatrix mapping. Tensors llama.cpp cannot quantize (1-D/flat, same
    rule as the probe assertion) become ``quantizable=False`` singletons
    instead of probed groups: a probe of an all-flat group measures a
    no-op and would record a bogus zero-delta "free compression" row
    that corrupts the DP. Their bytes ride along at source precision
    exactly like norm groups today. Non-flat singletons keep the
    tensor's own quantizable flag.

    Returns (new_catalog, report).
    """
    from weight_features import _catalog_sha256

    catalog = copy.deepcopy(catalog)
    tensors = catalog.get("tensors") or {}
    prev_groups = sorted((catalog.get("groups") or {}).keys())
    prev_sha = catalog.get("catalog_sha256")

    for name, t in tensors.items():
        t["group_id"] = name
    groups = _rebuild_groups(tensors)

    forced_fixed: list[str] = []
    for gid, g in groups.items():
        names = g.get("tensor_names") or []
        if len(names) == 1 and _is_flat_tensor(tensors[names[0]]):
            g["quantizable"] = False
            forced_fixed.append(gid)
    catalog["groups"] = groups
    catalog["catalog_sha256"] = _catalog_sha256(catalog)
    report = {
        "grouping": "per-tensor",
        "score_source": "per-tensor (no banding)",
        "roles_rebanded": [],
        "skipped_roles": {},
        "assumed_layers": {},
        "boundaries": {},
        "n_groups_before": len(prev_groups),
        "n_groups_after": len(groups),
        "forced_fixed_flat": sorted(forced_fixed),
        "prev_groups": prev_groups,
        "prev_sha256": prev_sha,
        "catalog_sha256": catalog["catalog_sha256"],
    }
    return catalog, report


def reband_catalog(
    catalog: dict[str, Any],
    proxy: dict[str, Any],
    *,
    bands_per_role: int = BANDS_PER_ROLE,
    imatrix_gguf: str | None = None,
    min_ss_reduction: float = MIN_SS_REDUCTION,
    per_tensor: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite group membership by imatrix-guided bands.

    Score source: the real ``imatrix.gguf`` per-channel statistics when
    provided (preferred — measured, and it already excludes junk tensors
    like rope_freqs), else per-tensor proxy importance with non-finite
    entries skipped. Roles whose optimal cuts explain too little variance
    (relative SS reduction below ``min_ss_reduction``) keep their existing
    groups instead of cutting on noise.

    Returns (new_catalog, report). The report records per-role boundaries,
    skipped roles with reasons, score source, previous group ids, and sha
    before/after for auditability.

    ``per_tensor=True`` bypasses banding entirely and explodes to
    single-tensor groups (gold-run grouping); see ``explode_per_tensor``.
    """
    if per_tensor:
        return explode_per_tensor(catalog)
    from activation_features import aggregate_activation_group
    from weight_features import _catalog_sha256, aggregate_group_features

    catalog = copy.deepcopy(catalog)
    tensors = catalog.get("tensors") or {}

    score_source = "proxy"
    scores: dict[tuple[str, int], float] = {}
    if imatrix_gguf:
        scores = real_imatrix_scores(imatrix_gguf, tensors)
        score_source = "imatrix_gguf" if scores else "proxy(empty_gguf)"
    if score_source != "imatrix_gguf":
        proxy_tensors = proxy.get("tensors") or {}
        scores = role_layer_scores(proxy_tensors, tensors)

    # Per-role ordered layer lists that have scores.
    role_layers: dict[str, list[int]] = {}
    for (role, layer) in scores:
        role_layers.setdefault(role, []).append(layer)
    for layers in role_layers.values():
        layers.sort()

    # New band assignment per (role, layer); roles without scores keep
    # their existing groups (assignment stays None).
    assignment: dict[tuple[str, int], tuple[str, str]] = {}
    boundaries: dict[str, list[list[int]]] = {}
    skipped_roles: dict[str, str] = {}
    assumed_layers: dict[str, list[int]] = {}
    # Full layer set per role from the catalog (scored or not).
    catalog_role_layers: dict[str, set[int]] = {}
    for t in tensors.values():
        if t.get("layer") is not None and t.get("role") is not None:
            catalog_role_layers.setdefault(str(t["role"]), set()).add(
                int(t["layer"])
            )
    for role, layers in sorted(role_layers.items()):
        vals = [scores[(role, lyr)] for lyr in layers]
        if _total_ss(vals) <= 0.0:
            skipped_roles[role] = "zero variance — kept existing groups"
            continue
        breaks = fisher_jenks_breaks(vals, bands_per_role)
        nbands = len(breaks) + 1
        seg_ss = sum(
            _total_ss(vals[(breaks[i - 1] if i else 0):(breaks[i] if i < len(breaks) else len(vals))])
            for i in range(nbands)
        )
        if 1.0 - seg_ss / _total_ss(vals) < min_ss_reduction:
            skipped_roles[role] = (
                f"cuts explain too little variance "
                f"({100.0 * (1.0 - seg_ss / _total_ss(vals)):.1f}% < "
                f"{100.0 * min_ss_reduction:.0f}%) — kept existing groups"
            )
            continue
        if nbands == 3:
            depths = list(BAND_DEPTHS_3)
        else:
            depths = [f"band{i}" for i in range(nbands)]
        bounds: list[list[int]] = []
        start = 0
        for b, depth in enumerate(depths):
            end = breaks[b] if b < len(breaks) else len(layers)
            for lyr in layers[start:end]:
                assignment[(role, lyr)] = (f"{role}@{depth}", depth)
            bounds.append([layers[start], layers[end - 1]])
            start = end
        boundaries[role] = bounds
        # Unscored layers of a rebanded role join the nearest scored
        # layer's band (contiguity preserved; explicit, not silent).
        # Without this they would keep thirds-era groups while sharing
        # group *names* with the new bands — mixed semantics.
        scored_pos = {lyr: band_index(i, breaks) for i, lyr in enumerate(layers)}
        assumed: list[int] = []
        for lyr in sorted(catalog_role_layers.get(role, ())):
            if (role, lyr) in scores or (role, lyr) in assignment:
                continue
            near = min(layers, key=lambda s: (abs(s - lyr), s))
            b = scored_pos[near]
            depth = list(BAND_DEPTHS_3)[b] if len(breaks) + 1 == 3 else f"band{b}"
            assignment[(role, lyr)] = (f"{role}@{depth}", depth)
            assumed.append(lyr)
        if assumed:
            assumed_layers[role] = assumed

    prev_groups = sorted((catalog.get("groups") or {}).keys())
    prev_sha = catalog.get("catalog_sha256")

    # Rewrite tensor membership; untouched tensors keep group_id/depth.
    for name, t in tensors.items():
        if t.get("layer") is None:
            continue
        key = (str(t.get("role")), int(t["layer"]))
        if key not in assignment:
            continue
        gid, depth = assignment[key]
        t["group_id"] = gid
        t["depth"] = depth

    # Rebuild groups from membership (shared helper: identical records
    # whether banded or exploded).
    groups = _rebuild_groups(tensors)
    catalog["groups"] = groups
    catalog["catalog_sha256"] = _catalog_sha256(catalog)

    report = {
        "bands_per_role": bands_per_role,
        "score_source": score_source,
        "roles_rebanded": sorted(set(role_layers) - set(skipped_roles)),
        "skipped_roles": skipped_roles,
        "assumed_layers": assumed_layers,
        "boundaries": boundaries,
        "n_groups_before": len(prev_groups),
        "n_groups_after": len(groups),
        "prev_groups": prev_groups,
        "prev_sha256": prev_sha,
        "catalog_sha256": catalog["catalog_sha256"],
    }
    return catalog, report
