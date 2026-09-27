"""Spec 4(i) — tail-KLD correctness. Unit test, runnable with pytest."""

import math

import numpy as np
import pytest

from kld import (
    compute_kld_metrics,
    load_kld_array,
    parse_llama_perplexity_kl,
    update_row_with_kld_array,
    update_row_with_measured_kl,
)


def test_hand_computed_six_elements():
    k = [0.01, 0.02, 0.05, 0.1, 0.5, 2.0]
    m = compute_kld_metrics(k)
    assert m["n_tokens"] == 6
    assert m["kld_mean"] == pytest.approx(sum(k) / 6, rel=1e-12)
    # ceil(0.01*6) = 1 -> tail = max = 2.0
    assert m["kld_tail_1pct"] == pytest.approx(2.0, rel=1e-12)


def test_n100_known_distribution():
    # 1..100: mean = 50.5, top ceil(1) = 1 -> tail = 100.0
    k = list(range(1, 101))
    m = compute_kld_metrics(k)
    assert m["n_tokens"] == 100
    assert m["kld_mean"] == pytest.approx(50.5, rel=1e-12)
    assert m["kld_tail_1pct"] == pytest.approx(100.0, rel=1e-12)
    # 200 elements -> top 2 -> mean(199, 200) = 199.5
    k2 = list(range(1, 201))
    m2 = compute_kld_metrics(k2)
    assert m2["kld_tail_1pct"] == pytest.approx(199.5, rel=1e-12)


def test_does_not_mutate_input():
    k = np.array([0.5, 2.0, 0.01, 0.1, 0.05, 0.02])
    before = k.copy()
    compute_kld_metrics(k)
    np.testing.assert_array_equal(k, before)
    lst = [0.01, 0.02, 0.05, 0.1, 0.5, 2.0]
    snapshot = list(lst)
    compute_kld_metrics(lst)
    assert lst == snapshot


def test_n0_returns_nan():
    m = compute_kld_metrics([])
    assert m["n_tokens"] == 0
    assert math.isnan(m["kld_mean"])
    assert math.isnan(m["kld_tail_1pct"])


def test_proxy_rows_have_no_fictional_tail():
    """Proxy rows must not fabricate a tail: kld_tail_1pct is None."""
    from sensitivity import probe_groups_proxy

    catalog = {
        "tensors": {"blk.0.ffn_up.weight": {"n_elements": 1024}},
        "groups": {
            "ffn_up@early": {
                "role": "ffn_up",
                "depth": "early",
                "quantizable": True,
                "n_tensors": 1,
                "tensor_names": ["blk.0.ffn_up.weight"],
            }
        },
    }
    rows = probe_groups_proxy(catalog, probe_types=["Q4_K"], baseline_type="Q6_K")
    assert rows, "expected at least one proxy row"
    for r in rows:
        assert r["kld_mean"] == pytest.approx(r["delta_kld"])
        assert r["kld_tail_1pct"] is None
        assert r["n_tokens"] is None


def test_update_row_with_measured_array():
    row = {"kld_mean": 0.0, "kld_tail_1pct": 0.0, "n_tokens": None}
    update_row_with_kld_array(row, [0.01, 0.02, 0.05, 0.1, 0.5, 2.0])
    assert row["kld_tail_1pct"] == pytest.approx(2.0)
    assert row["n_tokens"] == 6
    assert row["kld_mean"] == pytest.approx(2.68 / 6)


def test_load_kld_array_txt_json(tmp_path):
    p = tmp_path / "k.txt"
    p.write_text("0.1 0.2\n0.3,0.4\n")
    arr = load_kld_array(p)
    assert arr.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    q = tmp_path / "k.json"
    q.write_text('{"kld": [1.0, 2.0]}')
    assert load_kld_array(q).tolist() == pytest.approx([1.0, 2.0])


def test_parse_stock_llama_perplexity_kl():
    log = (
        "Final estimate: PPL = 14.457662 +/- 0.467335\n"
        "Mean KLD: 1.178221 +- 0.028615\n"
        "Median KLD: 0.135805\n"
        "99.0% KLD: 10.516757\n"
        "99.9% KLD: 15.574794\n"
        "Maximum KLD: 20.082871\n"
        "Same top p: 73.575%\n"
    )
    m = parse_llama_perplexity_kl(log)
    assert m["kld_mean"] == pytest.approx(1.178221, rel=1e-9)
    assert m["kld_tail_1pct"] == pytest.approx(10.516757, rel=1e-9)
    assert m["kld_p999"] == pytest.approx(15.574794, rel=1e-9)
    assert m["n_tokens"] is None  # stock binary prints no token count


def test_parse_missing_tail_raises_hard_error():
    with pytest.raises(ValueError, match="99.0% KLD"):
        parse_llama_perplexity_kl("Mean KLD: 0.5 +- 0.01\nMedian KLD: 0.1\n")
    with pytest.raises(ValueError, match="Mean KLD"):
        parse_llama_perplexity_kl("99.0% KLD: 3.0\n")


def test_update_row_with_measured_kl():
    row = {"kld_mean": 0.0, "kld_tail_1pct": None, "n_tokens": None}
    update_row_with_measured_kl(
        row, {"kld_mean": 0.5, "kld_tail_1pct": 3.0, "n_tokens": None}
    )
    assert row["kld_mean"] == pytest.approx(0.5)
    assert row["kld_tail_1pct"] == pytest.approx(3.0)


def test_per_group_grids_drop_above_ladder():
    from sensitivity import group_probe_grid, probe_groups_proxy

    catalog = {
        "tensors": {
            "token_embd.weight": {"n_elements": 1000},
            "blk.0.attn_q.weight": {"n_elements": 1000},
        },
        "groups": {
            "embedding@global": {
                "role": "embedding", "depth": "global", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["token_embd.weight"],
            },
            "attn_q@early": {
                "role": "attn_q", "depth": "early", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["blk.0.attn_q.weight"],
            },
        },
    }
    grid = ["Q2_K", "Q4_K", "Q6_K", "Q8_0"]
    emb = catalog["groups"]["embedding@global"]
    assert group_probe_grid(emb, grid) == ["Q8_0"]  # floor above start
    assert group_probe_grid(catalog["groups"]["attn_q@early"], grid) == [
        "Q2_K", "Q4_K", "Q6_K"]  # Q8_0 dropped: DP could never choose it
    rows = probe_groups_proxy(catalog, probe_types=grid, baseline_type="Q6_K")
    assert len(rows) == 4  # 1 + 3, not 2 x 4
    assert {r["probe"] for r in rows if r["group_id"] == "embedding@global"} == {"Q8_0"}

    import pytest as _pt
    with _pt.raises(ValueError, match="covers nothing"):
        probe_groups_proxy(catalog, probe_types=["Q4_K"], baseline_type="Q6_K")


def test_default_grid_covers_pinned_ladders():
    from sensitivity import default_probe_grid

    catalog = {
        "tensors": {
            "token_embd.weight": {"n_elements": 1000},
            "blk.0.attn_q.weight": {"n_elements": 1000},
        },
        "groups": {
            "embedding@global": {
                "role": "embedding", "depth": "global", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["token_embd.weight"],
            },
            "attn_q@early": {
                "role": "attn_q", "depth": "early", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["blk.0.attn_q.weight"],
            },
        },
    }
    # Bare profile grid (Q3-Q6): covers nothing on embedding's Q8 ladder.
    grid = default_probe_grid(catalog, ["Q3_K", "Q4_K", "Q5_K", "Q6_K"])
    assert grid == ["Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]
    # Every group's per-group grid is non-empty under the default.
    from sensitivity import group_probe_grid

    for gid, g in catalog["groups"].items():
        assert group_probe_grid(g, grid), gid


def _two_group_catalog():
    return {
        "tensors": {
            "blk.0.attn_q.weight": {"n_elements": 1024},
            "token_embd.weight": {"n_elements": 512},
        },
        "groups": {
            "attn_q@early": {
                "role": "attn_q", "depth": "early", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["blk.0.attn_q.weight"],
            },
            "embedding@global": {
                "role": "embedding", "depth": "global", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["token_embd.weight"],
            },
        },
    }


def test_fixed_groups_skipped_proxy():
    from sensitivity import probe_groups_proxy

    catalog = _two_group_catalog()
    rows = probe_groups_proxy(
        catalog, probe_types=["Q4_K"], baseline_type="Q6_K",
        fixed_groups=frozenset({"embedding@global"}),
    )
    assert {r["group_id"] for r in rows} == {"attn_q@early"}
    with pytest.raises(ValueError, match="unknown groups"):
        probe_groups_proxy(
            catalog, probe_types=["Q4_K"], fixed_groups=frozenset({"nope"}),
        )


def test_fixed_groups_skipped_llama(tmp_path, monkeypatch):
    import llama_probe
    from sensitivity import probe_groups_llama

    def fake_measure_column(**kw):
        if kw.get("group_regex") is None:
            return {"kld_mean": 0.01, "kld_tail_1pct": 0.10}
        return {
            "kld_mean": 0.02, "kld_tail_1pct": 0.20,
            "group_bytes_measured": 12345,
        }

    monkeypatch.setattr(llama_probe, "measure_column", fake_measure_column)
    for name in ("model.gguf", "search.txt", "kl.bin"):
        (tmp_path / name).write_bytes(b"x")
    rows, base = probe_groups_llama(
        _two_group_catalog(),
        model_gguf=tmp_path / "model.gguf",
        search_txt=tmp_path / "search.txt",
        kl_base_bin=tmp_path / "kl.bin",
        probe_types=["Q4_K"], baseline_type="Q6_K",
        work_dir=tmp_path / "trials", jobs=1,
        fixed_groups=frozenset({"embedding@global"}),
    )
    assert {r["group_id"] for r in rows} == {"attn_q@early"}
    assert base["fixed_skipped"] == ["embedding@global"]


def test_fixed_groups_logged_proxy_table(tmp_path):
    from sensitivity import build_sensitivity_table

    result, rows = build_sensitivity_table(
        model_ref="t", out_dir=tmp_path / "s",
        catalog=_two_group_catalog(), mode="proxy",
        probe_types=["Q4_K"],
        fixed_groups=frozenset({"embedding@global"}),
    )
    assert {r["group_id"] for r in rows} == {"attn_q@early"}
    assert any("4c" in line for line in result.steps_log)
