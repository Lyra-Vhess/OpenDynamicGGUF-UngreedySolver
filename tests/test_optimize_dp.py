"""Phase 4 — optimize_recipes integration: dp_mckp default, greedy preserved."""

import json

import pytest

from optimizer import optimize_recipes
from sensitivity import probe_groups_proxy


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
        pareto_ratios=[0.7, 1.0], jobs=2,
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
