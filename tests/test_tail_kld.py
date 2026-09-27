"""Spec 4(i) — tail-KLD correctness. Unit test, runnable with pytest."""

import math

import numpy as np
import pytest

from kld import (
    PROXY_TAIL_MULTIPLIER,
    compute_kld_metrics,
    load_kld_array,
    proxy_tail_from_mean,
    update_row_with_kld_array,
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


def test_proxy_rows_carry_both_fields():
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
        assert r["kld_tail_1pct"] == pytest.approx(
            r["delta_kld"] * PROXY_TAIL_MULTIPLIER
        )
        assert "n_tokens" in r


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


def test_proxy_tail_helper():
    assert proxy_tail_from_mean(0.01) == pytest.approx(0.01 * PROXY_TAIL_MULTIPLIER)
