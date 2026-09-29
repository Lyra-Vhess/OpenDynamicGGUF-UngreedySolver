"""Report card: run-baseline sizing + measured-vs-predicted quality."""
import pytest

from validate import (
    build_report_card_data,
    write_quantization_report_card,
)


def _card_catalog():
    return {
        "n_layers": 1,
        "groups": {
            "ffn_up@global": {
                "role": "ffn_up", "depth": "global", "quantizable": True,
                "n_tensors": 1, "tensor_names": ["blk.0.ffn_up.weight"],
            },
        },
        "tensors": {
            "blk.0.ffn_up.weight": {"n_elements": 1_000_000,
                                    "quantizable": True},
        },
    }


def _card_inputs():
    rows = [{
        "group_id": "ffn_up@global", "probe": "Q4_K", "baseline": "Q8_0",
        "delta_kld": 0.01, "decision_hint": "neutral",
    }]
    manifest = {"primary": {
        "predicted_delta_kld": 0.476,
        "predicted_delta_kld_corrected": 0.009,
        "background_penalty_kld": 0.001233,
    }}
    payload = {
        "verdict": "RELEASE",
        "tier1": {
            "method": "llama_heldout",
            "perplexity": 9.5,
            "metrics": {
                "mean_kld": 0.006, "p99_kld": 0.08, "p999_kld": 0.3,
                "max_kld": 1.0, "top1_agree": 0.984,
            },
        },
    }
    return rows, manifest, payload


def test_card_uses_run_baseline_and_quality():
    rows, manifest, payload = _card_inputs()
    data = build_report_card_data(
        model_ref="m", catalog=_card_catalog(),
        assignments={"ffn_up@global": "Q4_K"},
        sensitivity_rows=rows, optimize_manifest=manifest,
        validate_payload=payload, sensitivity_baseline="Q8_0",
    )
    assert data["baseline"] == "Q8_0"
    assert "Q6" not in data["baseline"]
    q = data["quality"]
    assert q["predicted_delta_kld_corrected"] == pytest.approx(0.009)
    assert q["background_penalty_kld"] == pytest.approx(0.001233)
    assert q["measured_mean_kld"] == pytest.approx(0.006)
    assert q["measured_p99_kld"] == pytest.approx(0.08)
    assert q["measured_top1_agree"] == pytest.approx(0.984)
    assert q["measured_perplexity"] == pytest.approx(9.5)
    assert q["measured_method"] == "llama_heldout"
    assert any("Q8_0" in n for n in data["notes"])


def test_card_explicit_baseline_wins_and_measured_optional(tmp_path):
    rows, manifest, _payload = _card_inputs()
    data = build_report_card_data(
        model_ref="m", catalog=_card_catalog(),
        assignments={"ffn_up@global": "Q4_K"},
        sensitivity_rows=rows, optimize_manifest=manifest,
        validate_payload={"verdict": "FAIL", "tier1": {}},
        baseline="Q6_K", sensitivity_baseline="Q8_0",
    )
    assert data["baseline"] == "Q6_K"
    assert data["quality"]["measured_mean_kld"] is None
    paths = write_quantization_report_card(
        tmp_path, model_ref="m", catalog=_card_catalog(),
        assignments={"ffn_up@global": "Q4_K"},
        sensitivity_rows=rows, optimize_manifest=manifest,
        validate_payload={"verdict": "FAIL", "tier1": {}},
        sensitivity_baseline="Q8_0",
    )
    md = (tmp_path / "quantization_report_card.md").read_text()
    assert "background-corrected" in md
    assert "Measured held-out" in md
    html = (tmp_path / "quantization_report_card.html").read_text()
    assert "Measured quality" in html
    assert paths["json"].endswith(".json")
