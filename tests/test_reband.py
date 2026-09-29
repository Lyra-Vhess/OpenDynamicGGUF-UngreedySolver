"""Reband: exact segmentation optimality + catalog rewrite invariants."""

import itertools

import pytest

from reband import (
    BANDS_PER_ROLE,
    band_index,
    explode_per_tensor,
    fisher_jenks_breaks,
    reband_catalog,
    role_layer_scores,
)


def ss_cost(vals):
    m = len(vals)
    if m == 0:
        return 0.0
    mean = sum(vals) / m
    return sum((v - mean) ** 2 for v in vals)


def cuts_cost(vals, cuts, k):
    bounds = (0,) + tuple(cuts) + (len(vals),)
    return sum(ss_cost(vals[bounds[i]:bounds[i + 1]]) for i in range(k))


def brute_best_cost(vals, k):
    n = len(vals)
    return min(
        cuts_cost(vals, cuts, k)
        for cuts in itertools.combinations(range(1, n), k - 1)
    )


def test_fisher_jenks_matches_brute_force():
    profiles = [
        [0.10, 0.12, 0.09, 0.80, 0.85, 0.78],
        [0.9, 0.85, 0.2, 0.22, 0.18, 0.9, 0.88],
        [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4],
    ]
    for vals in profiles:
        for k in (2, 3):
            # Equal-cost ties may cut at different positions; compare cost.
            got = fisher_jenks_breaks(vals, k)
            assert cuts_cost(vals, got, k) == pytest.approx(
                brute_best_cost(vals, k)
            )


def test_fisher_jenks_degenerate():
    assert fisher_jenks_breaks([], 3) == []
    assert fisher_jenks_breaks([0.5], 3) == []
    assert fisher_jenks_breaks([0.1, 0.9], 1) == []
    assert band_index(0, [2, 4]) == 0
    assert band_index(2, [2, 4]) == 1
    assert band_index(9, [2, 4]) == 2


def tiny_catalog_proxy():
    tensors, proxy = {}, {}
    # attn_q: cliff between layers 2 and 3; ffn_up: flat-ish.
    q_scores = [0.10, 0.12, 0.09, 0.80, 0.85, 0.78]
    for lyr in range(6):
        for role, base in (("attn_q", q_scores[lyr]), ("ffn_up", 0.4 + 0.02 * lyr)):
            name = f"blk.{lyr}.{role}.weight"
            tensors[name] = {
                "role": role, "layer": lyr, "group_id": f"{role}@middle",
                "depth": "middle", "nbytes": 1000, "n_elements": 500,
                "quantizable": True,
                "weight_features": {"mean": 0.0, "variance": 1.0,
                                    "outlier_ratio": base},
                "activation_features": {"absmax": base, "outlier_ratio": 0.01},
            }
            proxy[name] = {"importance": base, "role": role,
                           "group_id": f"{role}@middle"}
    # Global tensor: no layer → untouched by rebanding.
    tensors["token_embd.weight"] = {
        "role": "embedding", "layer": None, "group_id": "embedding@global",
        "depth": "global", "nbytes": 500, "n_elements": 250,
        "quantizable": True,
    }
    proxy["token_embd.weight"] = {"importance": 0.99, "role": "embedding",
                                  "group_id": "embedding@global"}
    catalog = {"tensors": tensors,
               "groups": {"attn_q@middle": {}, "ffn_up@middle": {},
                          "embedding@global": {}},
               "catalog_sha256": "prev"}
    return {"tensors": tensors, "groups": catalog["groups"],
            "catalog_sha256": "prev"}, {"tensors": proxy}


def test_reband_cuts_on_cliff_and_preserves_coverage():
    catalog, proxy = tiny_catalog_proxy()
    new, report = reband_catalog(catalog, proxy)
    # Cliff role splits layers 0-2 vs 3-5.
    bounds = report["boundaries"]["attn_q"]
    assert bounds[0][1] == 2 and bounds[1][0] == 3
    assert report["roles_rebanded"] == ["attn_q", "ffn_up"]
    # Same band count as thirds.
    assert report["n_groups_after"] == 2 * BANDS_PER_ROLE + 1
    # Coverage preserved: every tensor exactly once.
    names = [n for g in new["groups"].values() for n in g["tensor_names"]]
    assert sorted(names) == sorted(catalog["tensors"].keys())
    # Global tensor untouched.
    assert new["tensors"]["token_embd.weight"]["group_id"] == "embedding@global"
    # Aggregates recomputed on new membership.
    early = new["groups"]["attn_q@early"]
    assert early["n_tensors"] == 3
    assert early["total_nbytes"] == 3000
    assert "weight_features" in early and "activation_features" in early
    assert new["catalog_sha256"] != "prev"
    assert report["prev_sha256"] == "prev"
    # Scores helper sanity.
    scores = role_layer_scores(proxy["tensors"], catalog["tensors"])
    assert scores[("attn_q", 0)] == pytest.approx(0.10)


def test_zero_variance_role_keeps_existing_groups():
    catalog, proxy = tiny_catalog_proxy()
    # Flatten ffn_up scores: no band structure → keep thirds.
    for name, info in proxy["tensors"].items():
        if info["role"] == "ffn_up":
            info["importance"] = 0.5
    new, report = reband_catalog(catalog, proxy)
    assert "ffn_up" in report["skipped_roles"]
    assert "attn_q" in report["roles_rebanded"]
    # Untouched role keeps its group.
    assert new["tensors"]["blk.0.ffn_up.weight"]["group_id"] == "ffn_up@middle"


def test_nonfinite_proxy_scores_skipped():
    catalog, proxy = tiny_catalog_proxy()
    proxy["tensors"]["blk.0.attn_q.weight"]["importance"] = float("inf")
    scores = role_layer_scores(proxy["tensors"], catalog["tensors"])
    # inf poisons the layer mean (mirrors the rope_freqs incident) —
    # reband must not crash; layer 0 simply scores inf.
    assert scores[("attn_q", 0)] == float("inf")


def test_real_imatrix_aggregation(monkeypatch):
    import reband

    table = {
        "data_offset": 100,
        "tensors": {
            "blk.3.attn_q.weight.in_sum2": {
                "name": "x", "dtype": "F32", "offset": 0, "n_elements": 2,
            },
            "blk.3.attn_q.weight.counts": {
                "name": "y", "dtype": "F32", "offset": 8, "n_elements": 2,
            },
            "junk.other": {"name": "z", "dtype": "F32", "offset": 16,
                           "n_elements": 1},
        },
    }
    # real_imatrix_scores imports gguf_tensors inside the function,
    # so patch that module attr.
    import gguf_tensors
    monkeypatch.setattr(gguf_tensors, "gguf_tensor_map", lambda p: table)
    monkeypatch.setattr(reband, "_read_f32_blob",
                        lambda path, off, n: [4.0, 12.0] if off == 100 else [1.0, 3.0])
    catalog_tensors = {"blk.3.attn_q.weight": {"role": "attn_q", "layer": 3}}
    out = reband.real_imatrix_scores("dummy.gguf", catalog_tensors)
    # sum(in_sum2)=16 / sum(counts)=4 → 4.0; junk name absent from catalog.
    assert out == {("attn_q", 3): pytest.approx(4.0)}


def _shaped_catalog():
    catalog, proxy = tiny_catalog_proxy()
    for name, t in catalog["tensors"].items():
        t["shape"] = [8, 8]  # non-flat 2-D
    # A 1-D bias and a norm: both must ride at source precision.
    catalog["tensors"]["blk.0.attn_q.bias"] = {
        "role": "attn_q", "layer": 0, "group_id": "attn_q@middle",
        "depth": "middle", "shape": [8], "nbytes": 32, "n_elements": 8,
        "quantizable": True,
    }
    catalog["tensors"]["blk.0.attn_norm.weight"] = {
        "role": "norm", "layer": 0, "group_id": "norm@middle",
        "depth": "middle", "shape": [8], "nbytes": 32, "n_elements": 8,
        "quantizable": False,
    }
    catalog["groups"]["norm@middle"] = {}
    return catalog, proxy


def test_explode_per_tensor_groups():
    catalog, proxy = _shaped_catalog()
    new, report = explode_per_tensor(catalog)
    tensors = catalog["tensors"]
    # One group per tensor, gid = tensor name, full coverage.
    assert report["n_groups_after"] == len(tensors)
    assert sorted(new["groups"]) == sorted(tensors)
    names = [n for g in new["groups"].values() for n in g["tensor_names"]]
    assert sorted(names) == sorted(tensors)
    # Singletons carry role/depth and exact byte totals.
    g = new["groups"]["blk.0.attn_q.weight"]
    assert g["n_tensors"] == 1 and g["role"] == "attn_q"
    assert g["total_nbytes"] == 1000
    assert g["quantizable"] is True
    assert report["n_groups_before"] == 4
    assert new["catalog_sha256"] != "prev"


def test_explode_flat_tensors_ride_fixed():
    catalog, proxy = _shaped_catalog()
    new, report = explode_per_tensor(catalog)
    # 1-D bias: quantizable tensor but unprobed group (no zero-delta row).
    assert new["groups"]["blk.0.attn_q.bias"]["quantizable"] is False
    assert new["groups"]["blk.0.attn_norm.weight"]["quantizable"] is False
    assert sorted(report["forced_fixed_flat"]) == [
        "blk.0.attn_norm.weight", "blk.0.attn_q.bias",
    ]


def test_explode_missing_shape_raises():
    catalog, proxy = tiny_catalog_proxy()  # no shapes
    with pytest.raises(ValueError, match="[Ss]hape"):
        explode_per_tensor(catalog)


def test_reband_per_tensor_flag_bypasses_banding():
    catalog, proxy = _shaped_catalog()
    new, report = reband_catalog(catalog, proxy, per_tensor=True)
    assert report["grouping"] == "per-tensor"
    assert report["n_groups_after"] == len(catalog["tensors"])
    # Banded path untouched by the flag's existence (clean catalog: the
    # shaped one carries an extra bias tensor that would join the band).
    clean, proxy2 = tiny_catalog_proxy()
    new2, _ = reband_catalog(clean, proxy2)
    assert new2["groups"]["attn_q@early"]["n_tensors"] == 3
