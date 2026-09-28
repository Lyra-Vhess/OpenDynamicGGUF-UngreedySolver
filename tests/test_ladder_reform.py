"""Ladder reform: uniform ladder (no role pins), F32/F16 ceiling, lazy pricing.

Covers the pin-gutting + ceiling extension + lazy-GPU-driver contracts:
ladders are role-blind, negative-delta ceiling columns are choosable,
agreement never exceeds 1.0, and the lazy driver measures floors plus
attractive-only columns (resume-aware).
"""

import pytest

from optimizer import _candidate_ladder, dp_mckp_optimize
from sensitivity import (
    _llama_row_dict,
    default_probe_grid,
    estimate_group_nbytes,
    group_ladder,
    probe_groups_lazy,
)


def _catalog():
    groups, tensors = {}, {}
    specs = [
        ("embedding@global", "embedding", "global", 1_000_000),
        ("attn_v@early", "attn_v", "early", 1_000_000),
        ("ffn_up@middle", "ffn_up", "middle", 1_000_000),
    ]
    for gid, role, depth, n in specs:
        name = f"t.{role}.weight"
        groups[gid] = {
            "role": role, "depth": depth, "quantizable": True,
            "n_tensors": 1, "tensor_names": [name],
        }
        tensors[name] = {"n_elements": n, "nbytes": n * 2,
                         "group_id": gid, "quantizable": True}
    return {"groups": groups, "tensors": tensors}


def test_ladder_is_uniform_full_ladder():
    """No role floors: embedding/attn_v get the same full ladder as all."""
    from optimizer import LADDER

    catalog = _catalog()
    candidates, floors = _candidate_ladder(
        catalog=catalog, sensitivity_rows=[], start_type="Q6_K",
    )
    for gid in catalog["groups"]:
        assert candidates[gid] == list(LADDER), gid
        assert floors[gid] == "Q2_K", gid
    assert LADDER[:3] == ["F32", "F16", "Q8_0"]  # ceiling present


def test_pin_high_floor_survives_gutting():
    """Measured pin_high hints still floor at Q5_K (the kept mechanism)."""
    catalog = _catalog()
    rows = [{
        "group_id": "ffn_up@middle", "probe": "Q4_K",
        "decision_hint": "pin_high", "delta_kld": 0.09,
    }]
    candidates, floors = _candidate_ladder(
        catalog=catalog, sensitivity_rows=rows, start_type="Q6_K",
    )
    assert floors["ffn_up@middle"] == "Q5_K"
    assert candidates["ffn_up@middle"] == ["F32", "F16", "Q8_0", "Q6_K", "Q5_K"]
    assert floors["embedding@global"] == "Q2_K"


def test_negative_delta_ceiling_column_wins_loose_budget():
    """An F32 column beating the anchor (negative delta) is choosable."""
    from dp_mckp import BIN_BYTES

    catalog = _catalog()
    rows = [
        {
            "group_id": gid, "probe": "F32",
            "kld_mean": -0.05, "delta_kld": -0.05,
            "kld_tail_1pct": -0.02, "n_tokens": 1000,
        }
        for gid in catalog["groups"]
    ]
    n_elem = 1_000_000
    huge = (
        3 * estimate_group_nbytes(n_elem, "F32") + 10 * BIN_BYTES
    )
    res = dp_mckp_optimize(
        catalog=catalog, sensitivity_rows=rows, objective="mean",
        auto_cap=False, budget_bytes=huge,
    )
    assert set(res["allocation"].values()) == {"F32"}
    assert res["total_mean_kld"] < 0


def test_agreement_clamped_for_above_baseline_probe():
    """Negative ΔKLD (better than anchor) never yields >100% agreement."""
    g = {"role": "other", "depth": "global", "n_tensors": 1}
    base = {"kld_mean": 1.0, "kld_tail_1pct": 0.5}
    m = {
        "kld_mean": 0.99, "kld_tail_1pct": 0.49,  # better than anchor
        "kld_p999": None, "same_top_p": 0.999, "perplexity": 9.0,
        "group_bytes_measured": 4000, "probe_exempt_tensors": [],
    }
    row = _llama_row_dict(
        gid="other@global", g=g, q="F32", m=m, base=base,
        n_elem=1000, baseline_type="Q6_K",
    )
    assert row["delta_kld"] < 0
    assert row["top_token_agree"] == 1.0
    assert row["efficiency"] == 0.0  # negative bytes saved → no score
    assert row["decision_hint"] == "neutral"


def test_default_grid_is_full_ladder():
    catalog = _catalog()
    grid = default_probe_grid(catalog, ["Q3_K", "Q4_K", "Q5_K", "Q6_K"])
    from optimizer import LADDER

    assert grid == list(LADDER)


def _fake_measure_factory(log):
    """Stub measure_column: absolute KLD by probe type; anchor ~0.

    F32/F16 match Q8 exactly (zero marginal gain over Q8), so the bound
    model prices them at ub=0 and they are never attractive.
    """
    table = {"Q2_K": 1.0, "Q3_K": 0.5, "Q4_K": 0.2, "Q5_K": 0.08,
             "Q6_K": 0.05, "Q8_0": 0.01, "F16": 0.01, "F32": 0.01}

    def fake(*, group_regex=None, probe_type, tag, **kw):
        log.append((group_regex is not None, probe_type, tag))
        v = 0.0 if group_regex is None else table[probe_type.upper()]
        return {
            "kld_mean": v, "kld_tail_1pct": v, "kld_p999": v,
            "same_top_p": 0.99, "perplexity": 9.0,
            "group_bytes_measured": 100, "probe_exempt_tensors": [],
            "trial_tag": tag,
        }

    return fake


def test_lazy_driver_skips_unattractive_ceiling(tmp_path, monkeypatch):
    """    Tight pricing budget: floors measured, the F32/F16/Q8 ceiling priced
    out before touching the GPU (zero marginal gain over Q8 at prohibitive
    byte cost). Rows cover probed columns only.
    """
    import llama_probe

    catalog = _catalog()
    for g in catalog["groups"].values():
        name = g["tensor_names"][0]
        catalog["tensors"][name]["n_elements"] = 100_000_000
    log: list = []
    monkeypatch.setattr(
        llama_probe, "measure_column", _fake_measure_factory(log))

    n = 100_000_000
    floor_total = 3 * estimate_group_nbytes(n, "Q2_K")
    rows, absinfo = probe_groups_lazy(
        catalog, model_gguf="m.gguf", search_txt="s.txt",
        kl_base_bin="k.bin", probe_types=None, baseline_type="Q6_K",
        work_dir=tmp_path / "trials", jobs=2,
        pricing_budget_bytes=floor_total + 20 * 1024 * 1024,
        certificate_mode="bounded", kld_objective="mean",
    )
    probed_types = {r["probe"] for r in rows}
    assert "Q2_K" in probed_types  # floors mandatory
    assert not ({"F32", "F16", "Q8_0"} & probed_types)  # ceiling priced out
    assert not any(call[1] in ("F32", "F16", "Q8_0") for call in log if call[0])
    assert absinfo["lazy_probing"] is True
    assert absinfo["pricing_excluded_columns"] > 0
    assert absinfo["grid_skipped"] == absinfo["pricing_excluded_columns"]
    assert {r["method"] for r in rows} == {"llama_probe"}
    assert {r["split"] for r in rows} == {"search"}


def test_lazy_driver_resume_skips_measured(tmp_path, monkeypatch):
    """Second run reuses sidecar records: zero GPU calls, same rows."""
    import llama_probe

    catalog = _catalog()
    log: list = []
    monkeypatch.setattr(
        llama_probe, "measure_column", _fake_measure_factory(log))

    n = 1_000_000
    kwargs = dict(
        model_gguf="m.gguf", search_txt="s.txt", kl_base_bin="k.bin",
        probe_types=["Q2_K", "Q6_K"], baseline_type="Q6_K",
        work_dir=tmp_path / "trials", jobs=1,
        pricing_budget_bytes=3 * estimate_group_nbytes(n, "Q6_K"),
        certificate_mode="bounded", kld_objective="mean",
    )
    rows1, _ = probe_groups_lazy(catalog, **kwargs)
    assert [c for c in log if c[0]]  # something real was measured
    sidecar = tmp_path / "trials" / "probed.jsonl"
    assert sidecar.is_file()
    log.clear()

    def _boom(*args, **kwargs):
        raise AssertionError("GPU must not be touched on resume")

    monkeypatch.setattr(llama_probe, "measure_column", _boom)
    rows2, abs2 = probe_groups_lazy(catalog, **kwargs)
    assert log == []  # anchor too comes from the sidecar
    assert [(r["group_id"], r["probe"]) for r in rows2] == [
        (r["group_id"], r["probe"]) for r in rows1
    ]
    assert "cached=" in abs2["pricing_resume_note"]


def _layered_catalog():
    catalog = _catalog()
    layers = {"embedding@global": None, "attn_v@early": 0,
              "ffn_up@middle": 5}
    for gid, layer in layers.items():
        for n in catalog["groups"][gid]["tensor_names"]:
            catalog["tensors"][n]["role"] = catalog["groups"][gid]["role"]
            catalog["tensors"][n]["layer"] = layer
    return catalog


def test_imatrix_group_scores_aggregate_per_group(tmp_path, monkeypatch):
    """Real imatrix scores aggregate to per-group means; unscored groups
    absent; missing file gives neutral None."""
    import reband
    import sensitivity as sens_mod

    fake_gguf = tmp_path / "imatrix.gguf"
    fake_gguf.write_bytes(b"fake")
    monkeypatch.setattr(
        reband, "real_imatrix_scores",
        lambda path, tensors: {("attn_v", 0): 4.0, ("ffn_up", 5): 10.0},
    )
    catalog = _layered_catalog()
    scores = sens_mod.imatrix_group_scores(fake_gguf, catalog)
    assert scores == {"attn_v@early": 4.0, "ffn_up@middle": 10.0}
    assert sens_mod.imatrix_group_scores(tmp_path / "missing.gguf", catalog) is None
    assert sens_mod.imatrix_group_scores(None, catalog) is None


def test_lazy_driver_uses_imatrix_scales(tmp_path, monkeypatch):
    """Scores passed through: audit records the imatrix source."""
    import llama_probe

    catalog = _catalog()
    log: list = []
    monkeypatch.setattr(
        llama_probe, "measure_column", _fake_measure_factory(log))
    n = 1_000_000
    _, absinfo = probe_groups_lazy(
        catalog, model_gguf="m.gguf", search_txt="s.txt",
        kl_base_bin="k.bin", probe_types=["Q2_K", "Q6_K"],
        baseline_type="Q6_K", work_dir=tmp_path / "trials", jobs=1,
        pricing_budget_bytes=3 * estimate_group_nbytes(n, "Q6_K"),
        certificate_mode="bounded", kld_objective="mean",
        imatrix_scores={"embedding@global": 9.0, "attn_v@early": 1.0,
                        "ffn_up@middle": 2.0},
    )
    assert absinfo["pricing_sensitivity_source"] == "imatrix"
    _, absinfo2 = probe_groups_lazy(
        catalog, model_gguf="m.gguf", search_txt="s.txt",
        kl_base_bin="k.bin", probe_types=["Q2_K", "Q6_K"],
        baseline_type="Q6_K", work_dir=tmp_path / "trials2", jobs=1,
        pricing_budget_bytes=3 * estimate_group_nbytes(n, "Q6_K"),
        certificate_mode="bounded", kld_objective="mean",
    )
    assert absinfo2["pricing_sensitivity_source"] == "uniform"


def test_sidecar_roundtrip_and_corruption_rules(tmp_path):
    """Append/load roundtrips; trailing partial lines ignored, mid-file
    corruption raises."""
    from sensitivity import _sidecar_append, _sidecar_load, _sidecar_path

    work = tmp_path / "trials"
    work.mkdir()
    rec = {"gid": "g@early", "q": "Q2_K", "kld_mean": 1.0,
           "kld_tail_1pct": 2.0}
    _sidecar_append(work, rec)
    _sidecar_append(work, {"gid": "__anchor__", "q": "Q6_K",
                           "kld_mean": 0.0, "kld_tail_1pct": 0.0})
    cached, ignored = _sidecar_load(work)
    assert ignored == 0
    assert cached[("g@early", "Q2_K")]["kld_mean"] == 1.0
    assert ("__anchor__", "Q6_K") in cached
    # Trailing partial write (killed process): ignored, cell remeasured.
    with open(_sidecar_path(work), "a", encoding="utf-8") as f:
        f.write('{"gid": "g@early", "q": "Q4_K", "kld')
    cached, ignored = _sidecar_load(work)
    assert ignored == 1
    assert ("g@early", "Q4_K") not in cached
    # Mid-file corruption after good data: loud, never guessed.
    import pytest as _pt

    with open(_sidecar_path(work), "w", encoding="utf-8") as f:
        f.write('{"gid": "a", "q": "Q2_K"}\nNOT-JSON\n'
                '{"gid": "b", "q": "Q2_K"}\n')
    with _pt.raises(ValueError, match="append-only violated"):
        _sidecar_load(work)
    assert _sidecar_load(tmp_path / "nope") == ({}, 0)


def test_harvest_recovers_orphan_trials(tmp_path, monkeypatch):
    """Orphan (gguf + log) pairs become sidecar lines; ggufs deleted."""
    import sensitivity as sens_mod

    work = tmp_path / "trials"
    work.mkdir()
    gguf = work / "trial-g_early-Q2_K.gguf"
    gguf.write_bytes(b"x" * 4096)
    (work / "trial-g_early-Q2_K.perplexity.log").write_text("log")
    canned = {"kld_mean": 1.0, "kld_tail_1pct": 2.0, "kld_p999": 3.0,
              "same_top_p": 0.9, "perplexity": 9.0,
              "group_bytes_measured": 100, "probe_exempt_tensors": [],
              "trial_tag": "t"}
    def _fake_reparse(work_p, tag, tensors, qt, bt):
        if tag != "g_early-Q2_K":
            return None
        return dict(canned)

    monkeypatch.setattr(sens_mod, "_reparse_trial_abs", _fake_reparse)
    harvested, n_deleted, freed = sens_mod._harvest_known_trials(
        work, [("g@early", "Q2_K", ["t"])], "Q6_K",
    )
    assert len(harvested) == 1
    assert harvested[0][0] == ("g@early", "Q2_K")
    assert n_deleted == 1 and freed == 4096
    assert not gguf.exists()
    cached, _ = sens_mod._sidecar_load(work)
    assert cached[("g@early", "Q2_K")]["kld_mean"] == 1.0
    # Missing pair: left for measurement, nothing deleted.
    harvested, n_deleted, _ = sens_mod._harvest_known_trials(
        work, [("g@early", "Q4_K", ["t"])], "Q6_K",
    )
    assert harvested == [] and n_deleted == 0
