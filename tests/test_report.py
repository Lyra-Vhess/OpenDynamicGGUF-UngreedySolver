"""Feature 03 — report data extraction and self-contained HTML rendering."""

import json

import pytest

from report import build_report_data, generate_report, render_html


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run_dir(tmp_path):
    """Minimal but complete synthetic run mirroring the store layout."""
    root = tmp_path / "run"
    _write(
        root / "run.json",
        {
            "run_id": "20260101-test",
            "model_ref": "test:model",
            "status": "done",
            "quant_format": "q4_k_m",
            "quant_label": "Q4_K_M",
        },
    )
    _write(
        root / "steps/01_resolve/output.json",
        {
            "user_ref": "test:model",
            "hf_repo_id": "org/test-model",
            "source_sha256": "aa11",
            "descriptor": {
                "family": "gemma3",
                "layer_count": 2,
                "embedding_length": 64,
                "parameter_count": 1000,
                "context_length": 8192,
            },
        },
    )
    _write(
        root / "steps/05_catalog/tensor_catalog.json",
        {
            "tensors": {
                "blk.0.ffn_up.weight": {"n_elements": 1000, "group_id": "ffn_up@early"},
                "blk.0.attn_v.weight": {"n_elements": 500, "group_id": "attn_v@early"},
            },
            "groups": {
                "ffn_up@early": {"tensor_names": ["blk.0.ffn_up.weight"]},
                "attn_v@early": {"tensor_names": ["blk.0.attn_v.weight"]},
            },
        },
    )
    _write(
        root / "steps/12_sensitivity/sensitivity.json",
        {
            "method": "proxy_from_features",
            "baseline_type": "Q6_K",
            "probe_types": ["Q3_K", "Q4_K"],
            "rows": [
                {
                    "group_id": "ffn_up@early",
                    "probe": "Q3_K",
                    "delta_kld": 0.004,
                    "delta_bytes": 300,
                },
                {
                    "group_id": "attn_v@early",
                    "probe": "Q4_K",
                    "delta_kld": 0.037,
                    "delta_bytes": 45,
                },
            ],
        },
    )
    _write(
        root / "steps/13_optimize/output.json",
        {
            "method": "greedy_knapsack_v1",
            "budget_bytes": 5000,
            "estimated_bytes": 4800,
            "predicted_delta_kld": 0.01,
            "recipe_path": "recipe.yaml",
            "assignments": {"ffn_up@early": "Q3_K", "attn_v@early": "Q5_K"},
        },
    )
    _write(
        root / "steps/13_optimize/optimize_manifest.json",
        {
            "primary": {
                "estimated_bytes": 4800,
                "history": [
                    {
                        "group_id": "ffn_up@early",
                        "to": "Q3_K",
                        "delta_bytes": 300.0,
                        "delta_kld_inc": 0.004,
                        "efficiency": 75000.0,
                        "size_after": 4800,
                    }
                ],
            },
            "pareto": [
                {"estimated_bytes": 4800, "predicted_delta_kld": 0.01, "meets_budget": True},
                {"estimated_bytes": 6000, "predicted_delta_kld": 0.004, "meets_budget": False},
            ],
        },
    )
    _write(
        root / "steps/16_pareto_sweep/frontier.json",
        {
            "n_measured": 2,
            "n_params": 1500,
            "noise_floor_kld": 0.001,
            "ppl_base": 9.5,
            "points": [
                {
                    "actual_bytes": 600,
                    "tier1": {
                        "metrics": {
                            "mean_kld": 0.02,
                            "p99_kld": 0.2,
                            "top1_agree": 0.97,
                        },
                        "perplexity": 9.8,
                    },
                },
                {
                    "actual_bytes": 750,
                    "tier1": {
                        "metrics": {
                            "mean_kld": 0.01,
                            "p99_kld": 0.1,
                            "top1_agree": 0.98,
                        },
                        "perplexity": 9.6,
                    },
                },
            ],
            "xl": {
                "actual_bytes": 900,
                "tier1": {
                    "metrics": {
                        "mean_kld": 0.015,
                        "p99_kld": 0.15,
                        "top1_agree": 0.975,
                    },
                    "perplexity": 9.7,
                },
            },
        },
    )
    _write(
        root / "steps/15_validate/output.json",
        {
            "verdict": "PROVISIONAL",
            "method": "proxy_gates",
            "tier1": {
                "gates": {"mean_kld_max": 0.5, "top1_agree_min": 0.5},
                "metrics": {"mean_kld": 0.01, "top1_agree": 0.98},
                "pass": True,
                "note": "proxy",
            },
        },
    )
    _write(
        root / "experiment/comparison.json",
        {
            "schema": "odg/experiment/v1",
            "model": "google/functiongemma-270m-it",
            "quality": [
                {
                    "task": "mmlu",
                    "label": "MMLU",
                    "metric": "acc,none",
                    "scores": {
                        "bf16": 0.321,
                        "q4_k_m": 0.298,
                        "q5_k_m": 0.310,
                        "q6_k": 0.318,
                        "odg": 0.315,
                    },
                    "best_quantized": "q6_k",
                }
            ],
            "compression": [
                {"variant": "bf16", "label": "BF16", "bytes": 540_000_000},
                {"variant": "q4_k_m", "label": "Q4_K_M", "bytes": 180_000_000},
                {"variant": "odg", "label": "OpenDynamicGGUF", "bytes": 185_000_000},
            ],
        },
    )
    return root


class TestExtraction:
    def test_all_sections_available(self, run_dir):
        data = build_report_data(run_dir)
        assert all(v["available"] for v in data.values()), {
            k: v["available"] for k, v in data.items()
        }

    def test_allocation_reasons_trace_to_evidence(self, run_dir):
        rows = {r["group"]: r for r in build_report_data(run_dir)["allocations"]["rows"]}
        # downgraded group cites the greedy decision
        assert "greedy downgrade" in rows["ffn_up@early"]["reason"]
        assert rows["ffn_up@early"]["delta_kld"] == 0.004
        # no invented pins: unmeasured assignment cites the optimizer
        assert rows["attn_v@early"]["reason"] == "assigned by optimizer"

    def test_gates_extracted_with_thresholds(self, run_dir):
        gates = {g["metric"]: g for g in build_report_data(run_dir)["gates"]["gates"]}
        assert gates["mean_kld"]["pass"] is True
        assert gates["top1_agree"]["higher_better"] is True

    def test_partial_run_degrades_not_crashes(self, tmp_path):
        root = tmp_path / "partial"
        _write(root / "run.json", {"run_id": "x", "model_ref": "m", "status": "running"})
        data = build_report_data(root)
        assert data["summary"]["available"]
        assert not data["allocations"]["available"]
        html = render_html(data)  # must render "not run" sections
        assert "not run" in html


class TestRendering:
    def test_generate_writes_self_contained_html(self, run_dir):
        result = generate_report(run_dir)
        html = (run_dir / "report.html").read_text()
        assert result.sections_rendered == 8 and not result.sections_missing
        # self-contained: no external fetches (SVG xmlns is a namespace id, not a fetch)
        for external in ('<script src', "<link rel", 'src="http', "@import", "url(http"):
            assert external not in html
        assert "<svg" in html and "<style>" in html
        # key content present
        for needle in ("test:model", "PROVISIONAL", "ffn_up@early", "Q3_K", "MMLU"):
            assert needle in html
        # sections caption their source artifacts
        assert "steps/12_sensitivity/sensitivity.json" in html

    def test_sweep_frontier_charts(self, run_dir):
        from report import extract_sweep

        sw = extract_sweep(run_dir)
        assert sw["available"]
        assert [c["key"] for c in sw["charts"]] == [
            "mean_kld", "p99_kld", "top1_agree", "perplexity"]
        mean = sw["charts"][0]
        assert mean["log_y"] is True
        assert mean["ref"] == 0.001
        assert sw["charts"][1]["ref"] == 0.001  # record P99 floor absent → mean fallback
        assert mean["points"][-1].get("xl") is True
        # real bpw: 600 bytes * 8 / 1500 params = 3.20
        assert mean["points"][0]["bpw"] == "3.20"
        top1 = sw["charts"][2]
        assert top1["ref"] is None
        ppl = sw["charts"][3]
        assert ppl["ref"] == 9.5

    def test_sweep_charts_rendered_without_pick(self, run_dir):
        html = render_html(build_report_data(run_dir))
        assert "Frontier — Mean KLD vs size" in html
        assert "Frontier — 99% KLD vs size" in html
        assert "Frontier — Top-1 agreement vs size" in html
        assert "Frontier — Perplexity vs size" in html
        assert "(log scale)" in html
        assert "noise floor (all-Q8 background)" in html
        assert "BF16 base" in html
        assert "★" in html  # XL star, no winner diamond anywhere
        assert "chosen" not in html and "pick" not in html.lower()

    def test_sweep_fallbacks_from_run_artifacts(self, tmp_path):
        # No record-level baselines: noise floor from sensitivity rows,
        # ppl base from sibling Tier-1 logs, bpw from catalog elements.
        from report import extract_sweep

        root = tmp_path / "run"
        _write(root / "run.json", {"run_id": "x"})
        _write(root / "steps/05_catalog/tensor_catalog.json",
               {"tensors": {"a": {"n_elements": 1000}}, "groups": {}})
        _write(root / "steps/12_sensitivity/sensitivity.json",
               {"baseline_type": "Q8_0",
                "rows": [{"probe": "Q8_0", "kld_mean": 0.002},
                         {"probe": "Q4_K", "kld_mean": 0.05}]})
        _write(root / "steps/16_pareto_sweep/frontier.json",
               {"points": [{"actual_bytes": 1000,
                             "tier1": {"metrics": {"mean_kld": 0.02,
                                                   "p99_kld": 0.2,
                                                   "top1_agree": 0.9},
                                       "perplexity": 9.0}}]})
        logdir = root / "steps/16_pareto_sweep/exports/00-1k"
        logdir.mkdir(parents=True)
        (logdir / "llama-perplexity-tier1.log").write_text(
            "Mean PPL(Q)                   :   9.000000\n"
            "Mean PPL(base)                :   9.250000\n")
        sw = extract_sweep(root)
        assert sw["available"]
        assert sw["charts"][0]["ref"] == 0.002
        # no kld_tail_1pct in rows → P99 chart falls back to the mean floor
        assert sw["charts"][1]["ref"] == 0.002
        assert sw["charts"][0]["points"][0]["bpw"] == "8.00"
        assert sw["charts"][3]["ref"] == 9.25

    def test_scatter_sig_figs_and_labels(self):
        # Same engine as the PNGs: all labels land in the markup.
        from report import _svg_scatter

        pts = [
            {"size_mb": 3000.0, "kld": 9.833, "bpw": "3.42"},
            {"size_mb": 3000.5, "kld": 9.831, "bpw": "3.57"},
            {"size_mb": 3000.6, "kld": 9.829, "bpw": "3.90"},
            {"size_mb": 3000.7, "kld": 9.827, "bpw": "4.06"},
            {"size_mb": 6000.0, "kld": 9.700, "bpw": "5.40"},
        ]
        svg3 = _svg_scatter(pts, sig_figs=3)
        assert "9.83," in svg3
        svg2 = _svg_scatter(pts, sig_figs=2)
        assert "9.8," in svg2
        assert svg2.count("bpw") == 5  # none dropped
        assert svg2.lstrip().startswith("<svg")
        lonely = _svg_scatter(
            [{"size_mb": 3000.0, "kld": 1.0, "bpw": "3.42"},
             {"size_mb": 6000.0, "kld": 1.0, "bpw": "5.40"}])
        assert lonely.count("bpw") == 2
        # dots without bpw still plot (predicted-manifest fallback)
        bare = _svg_scatter([{"size_mb": 3000.0, "kld": 1.0},
                             {"size_mb": 6000.0, "kld": 2.0}])
        assert bare.lstrip().startswith("<svg")
        assert "bpw" not in bare

    def test_custom_out_path(self, run_dir, tmp_path):
        out = tmp_path / "elsewhere" / "r.html"
        result = generate_report(run_dir, out_path=out)
        assert out.is_file() and result.report_path == str(out)

    def test_not_a_run_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run.json"):
            generate_report(tmp_path)
