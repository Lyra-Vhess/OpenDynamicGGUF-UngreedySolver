"""Source anchor + background-rung contracts.

The run's zero point is the frozen source file against itself (never a
quantized trial); trial deltas subtract it, so any constant shift leaves
the DP's mind unchanged (proven by test_dp_cost_shift_invariance, not
assumed). Background rung (trial build target) rides on baseline_type.
"""

import pytest

import llama_probe
from dp_mckp import solve_mckp
from test_ladder_reform import (
    _catalog,
    _fake_anchor_factory,
    _fake_measure_factory,
)

KL_LOG = "Mean KLD: 0.000001\n99.0% KLD: 0.000010\nMean PPL(Q) : 2.700000\n"


def _patch_ppl(monkeypatch, log_text=KL_LOG, seen=None):
    import logits

    monkeypatch.setattr(logits, "find_llama_perplexity",
                        lambda explicit=None: "/bin/llama-perplexity")

    def fake_run(cmd, *, what):
        if seen is not None:
            seen.append(cmd)
        return log_text

    monkeypatch.setattr(llama_probe, "_run", fake_run)


def test_anchor_runs_on_frozen_file(tmp_path, monkeypatch):
    """No trial build: perplexity straight on the frozen GGUF, KL parsed."""
    seen = []
    _patch_ppl(monkeypatch, seen=seen)
    out = llama_probe.measure_source_anchor(
        model_gguf=tmp_path / "frozen.gguf", search_txt="s.txt",
        kl_base_bin="k.bin", work_dir=tmp_path / "trials",
    )
    cmd = seen[0]
    assert cmd[1:4] == ["-m", str(tmp_path / "frozen.gguf"), "-f"]
    assert "--kl-divergence" in cmd and "--kl-divergence-base" in cmd
    assert "llama-quantize" not in " ".join(cmd)  # no trial build
    assert out["kld_mean"] == pytest.approx(1e-6)
    assert out["kld_tail_1pct"] == pytest.approx(1e-5)
    assert (tmp_path / "trials" / "trial-anchor.perplexity.log").is_file()


def test_anchor_missing_kl_lines_fails(tmp_path, monkeypatch):
    """No KL lines = failed anchor = failed run (no silent zero)."""
    _patch_ppl(monkeypatch, log_text="perplexity = 2.7, no KL here\n")
    with pytest.raises(ValueError, match="missing required KL lines"):
        llama_probe.measure_source_anchor(
            model_gguf="m.gguf", search_txt="s.txt", kl_base_bin="k.bin",
            work_dir=tmp_path / "trials",
        )


def test_anchor_nonzero_self_kl_fails(tmp_path, monkeypatch):
    """Source must reproduce its own logits: big self-KL = corrupt chain."""
    _patch_ppl(monkeypatch, log_text="Mean KLD: 1.5\n99.0% KLD: 3.0\n")
    with pytest.raises(RuntimeError, match="does not reproduce its own"):
        llama_probe.measure_source_anchor(
            model_gguf="m.gguf", search_txt="s.txt", kl_base_bin="k.bin",
            work_dir=tmp_path / "trials",
        )


def test_anchor_missing_binary_fails(tmp_path, monkeypatch):
    import logits

    monkeypatch.setattr(logits, "find_llama_perplexity", lambda explicit=None: None)
    with pytest.raises(RuntimeError, match="llama-perplexity not found"):
        llama_probe.measure_source_anchor(
            model_gguf="m.gguf", search_txt="s.txt", kl_base_bin="k.bin",
            work_dir=tmp_path / "trials",
        )


def test_dp_cost_shift_invariance():
    """Adding a constant to every cost cannot move the DP optimum.

    This is the load-bearing property behind the source anchor: the zero
    point cancels out of every comparison the optimizer makes.
    """
    groups = ["a", "b"]
    cands = {"a": ["Q2_K", "Q6_K"], "b": ["Q2_K", "Q6_K"]}
    sizes = {(g, q): (10 if q == "Q2_K" else 20) for g in groups
             for q in cands[g]}
    costs = {("a", "Q2_K"): 0.9, ("a", "Q6_K"): 0.1,
             ("b", "Q2_K"): 0.5, ("b", "Q6_K"): 0.4}
    kw = dict(groups=groups, candidates=cands, size_bytes=sizes,
              budget_bytes=30, bin_bytes=1)
    r1 = solve_mckp(cost_tail=costs, **kw)
    shifted = {k: v + 5.0 for k, v in costs.items()}
    r2 = solve_mckp(cost_tail=shifted, **kw)
    assert r1["allocation"] == r2["allocation"]


def test_lazy_rows_subtract_source_anchor(tmp_path, monkeypatch):
    """Rows are trial-minus-anchor: a nonzero anchor shifts every row."""
    from sensitivity import estimate_group_nbytes, probe_groups_lazy

    catalog = _catalog()
    log: list = []
    monkeypatch.setattr(
        llama_probe, "measure_column", _fake_measure_factory(log))
    # Source self-KL is ~0 in production; nonzero here proves subtraction.
    monkeypatch.setattr(
        llama_probe, "measure_source_anchor",
        _fake_anchor_factory(mean=0.25, tail=0.5))
    n = 1_000_000
    rows, absinfo = probe_groups_lazy(
        catalog, model_gguf="m.gguf", search_txt="s.txt",
        kl_base_bin="k.bin", probe_types=["Q2_K"],
        baseline_type="Q6_K", work_dir=tmp_path / "trials", jobs=1,
        pricing_budget_bytes=3 * estimate_group_nbytes(n, "Q6_K"),
        certificate_mode="bounded", kld_objective="mean",
    )
    assert absinfo["anchor"] == "source"
    assert absinfo["kld_mean"] == pytest.approx(0.25)
    floor = [r for r in rows if r["probe"] == "Q2_K"]
    assert floor and all(r["kld_mean"] == pytest.approx(1.0 - 0.25)
                         for r in floor)


def test_background_rung_threads_to_trial_build(tmp_path, monkeypatch):
    """baseline Q8: trials build on Q8, bytes reference Q8, anchor untouched."""
    from sensitivity import estimate_group_nbytes, probe_groups_lazy

    catalog = _catalog()
    seen: list = []

    def fake_measure(*, group_regex=None, probe_type, tag,
                     baseline_type, **kw):
        seen.append((group_regex, probe_type, baseline_type))
        return {
            "kld_mean": 0.01, "kld_tail_1pct": 0.01, "kld_p999": 0.01,
            "same_top_p": 0.99, "perplexity": 9.0,
            "group_bytes_measured": 100, "probe_exempt_tensors": [],
            "trial_tag": tag,
        }

    monkeypatch.setattr(llama_probe, "measure_column", fake_measure)
    monkeypatch.setattr(
        llama_probe, "measure_source_anchor", _fake_anchor_factory())
    n = 1_000_000
    rows, _ = probe_groups_lazy(
        catalog, model_gguf="m.gguf", search_txt="s.txt",
        kl_base_bin="k.bin", probe_types=["Q2_K"],
        baseline_type="Q8_0", work_dir=tmp_path / "trials", jobs=1,
        pricing_budget_bytes=3 * estimate_group_nbytes(n, "Q8_0"),
        certificate_mode="bounded", kld_objective="mean",
    )
    trials = [s for s in seen if s[0] is not None]
    assert trials and all(s[2] == "Q8_0" for s in trials)
    assert {r["baseline"] for r in rows} == {"Q8_0"}
    assert all(r["bytes_baseline"] == estimate_group_nbytes(n, "Q8_0")
               for r in rows)
