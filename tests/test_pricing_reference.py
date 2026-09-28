"""Derived pricing reference (no sensitivity budget flags) + step-13 loud note.

Step 12 prices columns against a reference budget that must be *loose*
relative to every budget step 13 might solve. That reference is derived —
``max(intended solve budget, Pareto-top coverage)`` — never user-supplied.
And when a step-13 allocation rests on proxy-estimated (never measured)
KLD columns, the manifest says so loudly instead of silently.
"""

import json

import pytest

from optimizer import (
    default_budget_bytes,
    optimize_recipes,
    pricing_reference_budget,
)
from test_optimize_dp import (
    tiny_catalog,
    tiny_measured_sensitivity,
    tiny_sensitivity,
)


def test_reference_is_intended_only():
    """The solve budget is a hard limit: Pareto targets above it are
    dropped, so no solve is ever looser than intended and the reference
    is the intended budget, nothing more."""
    catalog = tiny_catalog()
    assert pricing_reference_budget(
        catalog, intended_bytes=default_budget_bytes(catalog, ratio=1.15)
    ) == default_budget_bytes(catalog, ratio=1.15)
    assert pricing_reference_budget(
        catalog, intended_bytes=default_budget_bytes(catalog, ratio=0.55)
    ) == default_budget_bytes(catalog, ratio=0.55)


def _manifest(tmp_path, run_dir):
    return json.loads(
        (tmp_path / run_dir / "optimize_manifest.json").read_text())


def _floor_only_sensitivity(catalog):
    """Mixed table as a too-tight step-12 reference leaves it: measured
    floors only, everything above the floor proxy-estimated."""
    sens = tiny_measured_sensitivity(catalog)
    sens["rows"] = [r for r in sens["rows"] if r["probe"] == "Q2_K"]
    assert sens["rows"], "expected measured floor rows"
    return sens


def test_loud_note_fires_on_proxy_chosen(tmp_path):
    """Mixed table (measured floors, proxy rest) solved looser than the
    reference covered: upgrades rest on estimates → WARNING naming them.
    (With auto_cap on, the guardrail itself hard-errors on proxy picks
    before the note is reachable.)"""
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "mixed", catalog=catalog,
        sensitivity=_floor_only_sensitivity(catalog), budget_ratio=1.0,
        kld_objective="mean", auto_cap=False,
    )
    manifest = _manifest(tmp_path, "mixed")
    assert manifest["primary"]["proxy_kld_columns"], (
        "expected a loose mixed-table solve to rest on unmeasured columns")
    assert any("WARNING" in n and "proxy-estimated" in n for n in res.notes)


def test_loud_note_quiet_on_measured(tmp_path):
    """Fully measured table at a normal budget: no warning, empty list."""
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "meas", catalog=catalog,
        sensitivity=tiny_measured_sensitivity(catalog), budget_ratio=0.8,
        kld_objective="mean",
    )
    manifest = _manifest(tmp_path, "meas")
    assert manifest["primary"]["proxy_kld_columns"] == []
    assert not any("proxy-estimated" in n for n in res.notes)


def test_pareto_capped_at_hard_budget(tmp_path):
    """Ratios above the solve budget are dropped, not solved: every
    emitted point is at or below budget, and the manifest says what was
    dropped."""
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "cap", catalog=catalog,
        sensitivity=tiny_measured_sensitivity(catalog), budget_ratio=0.8,
        kld_objective="mean", pareto_ratios=[0.55, 0.72, 1.0],
    )
    manifest = _manifest(tmp_path, "cap")
    assert manifest["dropped_pareto_above_budget"] == [
        default_budget_bytes(catalog, ratio=1.0)]
    for point in manifest["pareto"]:
        assert point["budget_bytes"] <= res.budget_bytes
    assert any("hard limit" in n for n in res.notes)


def test_pareto_default_span_capped(tmp_path):
    """The default span tops at 1.0×Q6: with a 0.72 budget the top points
    are dropped and the primary + lower points remain."""
    catalog = tiny_catalog()
    res = optimize_recipes(
        model_ref="test", out_dir=tmp_path / "capdef", catalog=catalog,
        sensitivity=tiny_measured_sensitivity(catalog), budget_ratio=0.72,
        kld_objective="mean",
    )
    manifest = _manifest(tmp_path, "capdef")
    assert len(manifest["dropped_pareto_above_budget"]) == 3  # 0.8, 0.9, 1.0
    emitted = [p["budget_bytes"] for p in manifest["pareto"]]
    assert emitted and max(emitted) <= res.budget_bytes
