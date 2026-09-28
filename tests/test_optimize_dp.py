"""Phase 4 — optimize_recipes integration: dp_mckp default, greedy preserved."""

import itertools
import json
import math

import pytest

from dp_mckp import BIN_BYTES, bytes_to_bins
from optimizer import (
    SIZE_SANITY_ABS,
    SIZE_SANITY_REL,
    _estimate_total_bytes,
    dp_mckp_optimize,
    optimize_recipes,
)
from sensitivity import estimate_group_nbytes, probe_groups_proxy


def tiny_catalog():
    groups = {}
    tensors = {}
    specs = [
        ("attn_q@early", "attn_q", "early", 30_000_000),
        ("ffn_up@middle", "ffn_up", "middle", 50_000_000),
        ("ffn_down@late", "ffn_down", "late", 40_000_000),
    ]
    for gid, role, depth, n in specs:
        name = f"blk.0.{role}.weight"
        groups[gid] = {
            "role": role, "depth": depth, "quantizable": True,
            "n_tensors": 1, "tensor_names": [name],
        }
        tensors[name] = {"n_elements": n, "nbytes": n * 2,
                         "group_id": gid, "quantizable": True}
    return {"groups": groups, "tensors": tensors}


def tiny_sensitivity(catalog):
    rows = probe_groups_proxy(
        catalog, probe_types=["Q4_K", "Q5_K", "Q6_K"], baseline_type="Q6_K",
    )
    return {"method": "proxy_from_features", "rows": rows}


def tiny_measured_sensitivity(catalog):
    """Proxy rows stamped with stand-in *measured* tails (as step 12
    --mode llama would produce via update_row_with_measured_kl), extended
    to the full Q6..Q2 ladder so the tail firewall is satisfied."""
    import copy

    sens = tiny_sensitivity(catalog)
    have = {(r["group_id"], r["probe"]) for r in sens["rows"]}
    extra = []
    for r in sens["rows"]:
        if r["probe"] != "Q4_K":
            continue
        for q, mult in (("Q3_K", 1.5), ("Q2_K", 2.5)):
            if (r["group_id"], q) in have:
                continue
            c = copy.deepcopy(r)
            c["probe"] = q
            c["delta_kld"] = float(r["delta_kld"]) * mult
            c["kld_mean"] = float(r.get("kld_mean", r["delta_kld"])) * mult
            extra.append(c)
    sens["rows"] = sens["rows"] + extra
    for r in sens["rows"]:
        r["kld_tail_1pct"] = float(r["delta_kld"]) * 8.0
        r["n_tokens"] = 1000
    sens["method"] = "llama_probe"
    return sens


def test_dp_is_default_and_emits_certificate(tmp_path):
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "dp", catalog=catalog,
        sensitivity=tiny_measured_sensitivity(catalog), budget_ratio=0.8,
    )
    assert res.optimizer == "dp_mckp"
    assert res.method == "dp_mckp_colgen_v1"
    assert res.certificate is not None
    assert res.certificate["attractive_at_termination"] == []
    assert res.total_tail_kld is not None and res.total_mean_kld is not None
    assert res.estimated_bytes <= res.budget_bytes
    recipe = (tmp_path / "dp" / "recipe.yaml").read_text()
    for key in ("optimizer: dp_mckp", "kld_metric:", "cost_matrix:",
                "certificate:", "totals:", "allocation:", "pareto:"):
        assert key in recipe
    assert (tmp_path / "dp" / "recipe.tt").is_file()
    assert list((tmp_path / "dp" / "pareto").glob("*.yaml"))
    manifest = json.loads((tmp_path / "dp" / "optimize_manifest.json").read_text())
    assert manifest["primary"]["certificate"]["attractive_at_termination"] == []
    assert manifest["jobs"] == 1


def test_dp_tail_refuses_proxy_rows(tmp_path):
    """Tail objective hard-errors on unmeasured columns (user decision b)."""
    catalog = tiny_catalog()
    with pytest.raises(ValueError, match="measured kld_tail_1pct"):
        optimize_recipes(
            model_ref="test", out_dir=tmp_path / "refuse", catalog=catalog,
            sensitivity=tiny_sensitivity(catalog), budget_ratio=0.8,
            kld_objective="tail_1pct",
        )


def test_greedy_path_unchanged(tmp_path):
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "gr", catalog=catalog,
        sensitivity=tiny_sensitivity(catalog), budget_ratio=0.8,
        optimizer="greedy",
    )
    assert res.method == "greedy_knapsack_v1"
    assert res.certificate is None
    recipe = (tmp_path / "gr" / "recipe.yaml").read_text()
    assert "certificate:" not in recipe
    assert "predicted_mean_delta_kld" in recipe  # legacy fields intact


def test_dp_options_plumbed(tmp_path):
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "ex", catalog=catalog,
        sensitivity=tiny_sensitivity(catalog), budget_ratio=0.8,
        certificate_mode="exhaustive", kld_objective="mean",
        pareto_ratios=[0.7, 1.0], jobs=2, auto_cap=False,
    )
    assert res.kld_objective == "mean"
    assert res.certificate["mode"] == "exhaustive"
    assert res.certificate["excluded_columns"] == 0
    assert res.total_tail_kld is None  # proxy rows carry no measured tail
    assert res.total_mean_kld is not None
    manifest = json.loads((tmp_path / "ex" / "optimize_manifest.json").read_text())
    assert manifest["jobs"] == 2
    assert len(manifest["pareto"]) == 3  # 0.7, 1.0 + primary budget


def test_bad_options_rejected(tmp_path):
    catalog = tiny_catalog()
    sens = tiny_sensitivity(catalog)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        optimize_recipes(model_ref="t", out_dir=tmp_path / "x", catalog=catalog,
                         sensitivity=sens, optimizer="simulated_annealing")
    with pytest.raises(ValueError, match="Unknown kld_objective"):
        optimize_recipes(model_ref="t", out_dir=tmp_path / "y", catalog=catalog,
                         sensitivity=sens, kld_objective="median")


def _measured_index(catalog):
    sens = tiny_measured_sensitivity(catalog)
    idx = {(r["group_id"], r["probe"]): r for r in sens["rows"]}
    return sens, idx


def test_tail_cap_restricts_to_brute_force_optimum():
    """Mean + guardrail: allocation equals brute force over the cap-kept
    candidates (bin-space, same ceil convention as the solver)."""
    catalog = tiny_catalog()
    sens, idx = _measured_index(catalog)
    groups = sorted(catalog["groups"])
    ladder = ["Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]
    # Cap keeps every group's best-P99 candidate and removes at least one.
    cap = max(min(idx[(g, q)]["kld_tail_1pct"] for q in ladder) for g in groups)
    kept = {g: [q for q in ladder if idx[(g, q)]["kld_tail_1pct"] <= cap]
            for g in groups}
    assert all(kept.values())
    assert any(len(v) < len(ladder) for v in kept.values())
    n_elem = {g: sum(catalog["tensors"][n].get("n_elements") or 0
                     for n in catalog["groups"][g]["tensor_names"])
              for g in groups}
    size_b = {(g, q): estimate_group_nbytes(n_elem[g], q)
              for g in groups for q in ladder}
    mean_c = {(g, q): float(idx[(g, q)]["kld_mean"])
              for g in groups for q in ladder}
    budget_bins = sum(bytes_to_bins(size_b[(g, kept[g][-1])]) for g in groups) + 4
    best, best_alloc = math.inf, None
    for combo in itertools.product(*(kept[g] for g in groups)):
        alloc = dict(zip(groups, combo))
        s = sum(bytes_to_bins(size_b[(g, alloc[g])]) for g in groups)
        c = sum(mean_c[(g, alloc[g])] for g in groups)
        if s <= budget_bins and c < best:
            best, best_alloc = c, alloc
    assert best_alloc is not None
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        tail_cap=cap, size_margin=1.0, budget_bytes=budget_bins * BIN_BYTES,
    )
    assert res["allocation"] == best_alloc
    assert res["total_mean_kld"] == pytest.approx(best)
    assert res["tail_cap"] == cap
    assert res["cap_removed_columns"] == sum(
        len(ladder) - len(kept[g]) for g in groups)


def test_tail_cap_too_tight_names_group():
    catalog = tiny_catalog()
    sens, _ = _measured_index(catalog)
    with pytest.raises(ValueError, match="Raise --tail-cap"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
            tail_cap=1e-9, size_margin=1.0, budget_bytes=10 * BIN_BYTES,
        )


def test_tail_cap_drops_unmeasured_columns():
    """Guardrail drops columns with no measured P99 (lazy bound-excluded)
    instead of erroring; fully-unmeasured groups still fail loudly."""
    catalog = tiny_catalog()
    sens, _ = _measured_index(catalog)  # Q6..Q2 measured; ceiling not
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        tail_cap=1e9, size_margin=1.0, budget_bytes=200 * BIN_BYTES,
    )
    assert res["cap_dropped_unmeasured"] == 3 * 3  # F32/F16/Q8 × 3 groups
    assert all(
        e["type"] not in ("F32", "F16", "Q8_0")
        for e in res["cost_matrix"]["entries"]
    )
    # All-proxy table: every column unmeasured → nothing to solve over.
    sens_proxy = tiny_sensitivity(catalog)
    with pytest.raises(ValueError, match="removes every candidate"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens_proxy["rows"],
            objective="mean", tail_cap=1e9, size_margin=1.0,
            budget_bytes=200 * BIN_BYTES,
        )


def test_size_margin_default_is_neutral():
    """No margin constant: default 1.0 is a no-op; explicit margin scales."""
    catalog = tiny_catalog()
    groups, tensors = catalog["groups"], catalog["tensors"]
    assign = {g: "Q6_K" for g in groups}
    base = _estimate_total_bytes(assign, groups, tensors)
    assert _estimate_total_bytes(
        assign, groups, tensors, size_margin=1.0) == base
    assert _estimate_total_bytes(
        assign, groups, tensors, size_margin=2.0) == int(base * 2.0)


def test_measured_sizes_preferred_and_audited():
    """Row bytes_measured replaces the estimate; entries carry measured:true."""
    catalog = tiny_catalog()
    sens, idx = _measured_index(catalog)
    n_elem = {g: sum(catalog["tensors"][n].get("n_elements") or 0
                     for n in catalog["groups"][g]["tensor_names"])
              for g in catalog["groups"]}
    # Stamp exact measured bytes within sanity bounds of the estimate.
    for r in sens["rows"]:
        g, q = r["group_id"], r["probe"]
        r["bytes_measured"] = estimate_group_nbytes(n_elem[g], q)
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        auto_cap=False, budget_bytes=200 * BIN_BYTES,
    )
    by_key = {(e["group"], e["type"]): e for e in res["cost_matrix"]["entries"]}
    # Stamped rows carry measured:true; unmeasured ceiling columns exist
    # with measured:false (priced, possibly proxy-filled, never invented).
    for r in sens["rows"]:
        assert by_key[(r["group_id"], r["probe"])]["measured"] is True
    assert by_key[("attn_q@early", "F32")]["measured"] is False


def test_measured_size_sanity_trip_names_column():
    """A wildly-off measured size aborts loudly naming (group, type)."""
    catalog = tiny_catalog()
    sens, idx = _measured_index(catalog)
    gid = sorted(catalog["groups"])[0]
    for r in sens["rows"]:
        if r["group_id"] == gid and r["probe"] == "Q2_K":
            r["bytes_measured"] = 10  # absurd vs ~MiB-scale estimate
    with pytest.raises(ValueError, match=rf"\({gid}, Q2_K\)"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
            auto_cap=False, budget_bytes=200 * BIN_BYTES,
        )


def test_measured_size_within_bounds_passes():
    """Sanity constants are sane: 15% rel, 256KiB abs floor."""
    assert SIZE_SANITY_REL == pytest.approx(0.15)
    assert SIZE_SANITY_ABS == 256 * 1024
    assert BIN_BYTES == 256 * 1024
    assert bytes_to_bins(BIN_BYTES) == 1
    assert bytes_to_bins(BIN_BYTES + 1) == 2


def test_auto_cap_two_pass_beats_nothing_and_respects_tstar():
    """Auto-cap (default under mean): pass 1 mean-only, T* = worst P99 in
    that allocation, pass 2 re-solves mean subject to P99 <= T* and equals
    brute force over the T*-kept candidates."""
    catalog = tiny_catalog()
    sens, idx = _measured_index(catalog)
    groups = sorted(catalog["groups"])
    ladder = ["Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]
    n_elem = {g: sum(catalog["tensors"][n].get("n_elements") or 0
                     for n in catalog["groups"][g]["tensor_names"])
              for g in groups}
    size_b = {(g, q): estimate_group_nbytes(n_elem[g], q)
              for g in groups for q in ladder}
    mean_c = {(g, q): float(idx[(g, q)]["kld_mean"])
              for g in groups for q in ladder}
    tail_c = {(g, q): float(idx[(g, q)]["kld_tail_1pct"])
              for g in groups for q in ladder}
    # Generous budget: pass 1 lands on all-Q6 (min mean everywhere).
    budget_bins = sum(bytes_to_bins(size_b[(g, "Q6_K")]) for g in groups)
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        size_margin=1.0, budget_bytes=budget_bins * BIN_BYTES,
    )
    assert res["auto_cap"] is True
    assert res["allocation"] == {g: "Q6_K" for g in groups}
    tstar = max(tail_c[(g, "Q6_K")] for g in groups)
    assert res["tail_cap"] == pytest.approx(tstar)
    assert res["pass1_mean_kld"] == pytest.approx(res["total_mean_kld"])
    # Brute force over T*-kept candidates must agree with pass 2.
    kept = {g: [q for q in ladder if tail_c[(g, q)] <= tstar] for g in groups}
    assert all(kept.values())
    best, best_alloc = math.inf, None
    for combo in itertools.product(*(kept[g] for g in groups)):
        alloc = dict(zip(groups, combo))
        s = sum(bytes_to_bins(size_b[(g, alloc[g])]) for g in groups)
        c = sum(mean_c[(g, alloc[g])] for g in groups)
        if s <= budget_bins and c < best:
            best, best_alloc = c, alloc
    assert res["allocation"] == best_alloc
    assert res["total_mean_kld"] == pytest.approx(best)
    for g, q in res["allocation"].items():
        assert tail_c[(g, q)] <= tstar


def test_auto_cap_refuses_proxy_tails():
    """Auto-cap needs measured tails: proxy tables raise loudly."""
    catalog = tiny_catalog()
    sens = tiny_sensitivity(catalog)
    with pytest.raises(ValueError, match="auto-cap needs a measured"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
            size_margin=1.0, budget_bytes=200 * BIN_BYTES,
        )


def _fixed_bytes(catalog, sens, gid):
    # Fixed groups stay at source precision: catalog nbytes, never measured.
    return sum(catalog["tensors"][n].get("nbytes") or 0
               for n in catalog["groups"][gid]["tensor_names"])


def test_fixed_groups_excluded_but_counted_dp():
    from optimizer import dp_mckp_optimize

    catalog = tiny_catalog()
    sens = tiny_measured_sensitivity(catalog)
    fixed = {"ffn_down@late"}
    fixed_b = _fixed_bytes(catalog, sens, "ffn_down@late")
    # Brute force over the remaining groups with the reduced budget.
    groups = sorted(g for g in catalog["groups"] if g not in fixed)
    ladder = ["Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]
    idx = {(r["group_id"], r["probe"]): r for r in sens["rows"]}
    n_elem = {g: sum(catalog["tensors"][n].get("n_elements") or 0
                     for n in catalog["groups"][g]["tensor_names"])
              for g in groups}
    size_b = {(g, q): estimate_group_nbytes(n_elem[g], q)
              for g in groups for q in ladder}
    mean_c = {(g, q): float(idx[(g, q)]["kld_mean"])
              for g in groups for q in ladder}
    budget = 500 * BIN_BYTES
    best, best_alloc = math.inf, None
    for combo in itertools.product(*(ladder for _ in groups)):
        alloc = dict(zip(groups, combo))
        s = sum(bytes_to_bins(size_b[(g, alloc[g])]) for g in groups)
        c = sum(mean_c[(g, alloc[g])] for g in groups)
        if s <= bytes_to_bins(budget - fixed_b) and c < best:
            best, best_alloc = c, alloc
    assert best_alloc is not None
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        size_margin=1.0, budget_bytes=budget, fixed_groups=fixed,
        auto_cap=False, certificate_mode="exhaustive",
    )
    assert "ffn_down@late" not in res["allocation"]
    assert res["allocation"] == best_alloc
    assert res["total_mean_kld"] == pytest.approx(best)
    assert res["fixed_groups"]["ffn_down@late"]["bytes"] == fixed_b
    assert res["fixed_bytes_total"] == fixed_b
    assert res["total_bytes"] <= budget
    # Binned DP total plus fixed bytes brackets the exact byte sum
    # (per-group ceil can only undershoot by < 1 bin each).
    exact = sum(size_b[(g, res["allocation"][g])] for g in groups) + fixed_b
    assert res["total_bytes"] <= exact
    assert res["total_bytes"] > exact - len(groups) * BIN_BYTES


def test_fixed_groups_unknown_name_rejected():
    from optimizer import dp_mckp_optimize

    catalog = tiny_catalog()
    sens = tiny_measured_sensitivity(catalog)
    with pytest.raises(ValueError, match="unknown groups"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
            budget_bytes=200 * BIN_BYTES, fixed_groups={"nope@nowhere"},
        )


def test_fixed_groups_over_budget_rejected():
    from optimizer import dp_mckp_optimize

    catalog = tiny_catalog()
    sens = tiny_measured_sensitivity(catalog)
    with pytest.raises(ValueError, match="exceed budget"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
            budget_bytes=1, fixed_groups={"ffn_down@late"},
        )


def test_fixed_groups_greedy_skips_and_counts():
    from optimizer import greedy_optimize

    catalog = tiny_catalog()
    sens = tiny_sensitivity(catalog)
    res = greedy_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"],
        budget_bytes=10**15, fixed_groups={"ffn_down@late"},
    )
    assert "ffn_down@late" not in res["assignments"]
    # catalog nbytes for the fixed group still ride along in the total
    fixed_nb = sum(catalog["tensors"][n].get("nbytes") or 0
                   for n in catalog["groups"]["ffn_down@late"]["tensor_names"])
    assert res["estimated_bytes"] >= fixed_nb


def _catalog_with_kept_norm(extra_nbytes=1_500_000):
    catalog = tiny_catalog()
    catalog["tensors"]["norm.weight"] = {
        "n_elements": extra_nbytes // 2, "nbytes": extra_nbytes,
        "group_id": "norm@global", "quantizable": False,
    }
    return catalog, extra_nbytes


def test_kept_nonquant_bytes_counted_and_subtracted():
    """Non-quantizable tensors shrink the solve budget and ride in totals."""
    from optimizer import dp_mckp_optimize

    catalog, kept = _catalog_with_kept_norm()
    sens = tiny_measured_sensitivity(catalog)
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        size_margin=1.0, budget_bytes=500 * BIN_BYTES,
        auto_cap=False, certificate_mode="exhaustive",
    )
    assert res["kept_bytes_total"] == kept
    assert res["total_bytes"] >= kept
    # Kept bytes alone exceeding the budget fail loudly, naming the cause.
    with pytest.raises(ValueError, match="Non-DP bytes alone"):
        dp_mckp_optimize(
            catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
            budget_bytes=kept - 1, auto_cap=False,
            certificate_mode="exhaustive",
        )


def test_file_overhead_subtracted_and_recorded():
    """file_overhead_bytes counts toward the budget on both paths."""
    from optimizer import dp_mckp_optimize, greedy_optimize

    catalog = tiny_catalog()
    sens = tiny_measured_sensitivity(catalog)
    # Generous but bin-safe: 10**15 bytes would be billions of 256 KiB
    # bins (OOM); 200k bins still leaves every allocation feasible.
    big = 200_000 * BIN_BYTES
    base = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        budget_bytes=big, auto_cap=False, certificate_mode="exhaustive",
    )
    over = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=sens["rows"], objective="mean",
        budget_bytes=big, auto_cap=False, certificate_mode="exhaustive",
        file_overhead_bytes=2_000_000,
    )
    assert over["file_overhead_bytes"] == 2_000_000
    # Huge budget → identical allocation; totals differ by exactly overhead.
    assert over["allocation"] == base["allocation"]
    assert over["total_bytes"] - base["total_bytes"] == 2_000_000

    gbase = greedy_optimize(
        catalog=catalog, sensitivity_rows=tiny_sensitivity(catalog)["rows"],
        budget_bytes=10**15,
    )
    gover = greedy_optimize(
        catalog=catalog, sensitivity_rows=tiny_sensitivity(catalog)["rows"],
        budget_bytes=10**15, file_overhead_bytes=2_000_000,
    )
    assert gover["estimated_bytes"] - gbase["estimated_bytes"] == 2_000_000
