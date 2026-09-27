"""imatrix proxy: non-finite raw scores must not poison normalization."""

import json

from imatrix import build_proxy_importance


def tiny_catalog():
    tensors = {
        "blk.0.attn_q.weight": {
            "role": "attn_q", "layer": 0, "group_id": "attn_q@early",
            "quantizable": True,
            "weight_features": {"outlier_ratio": 0.01, "variance": 1.0},
            "activation_features": {"absmax": 2.0, "outlier_ratio": 0.001},
        },
        "rope_freqs.weight": {
            "role": "other", "layer": None, "group_id": "other@global",
            "quantizable": True,
            # Spectral-class blowup: infinite outlier ratio.
            "weight_features": {"outlier_ratio": float("inf"), "variance": 1.0},
            "activation_features": {"absmax": 1.0, "outlier_ratio": 0.0},
        },
    }
    return {"tensors": tensors}


def test_inf_tensor_excluded_not_normalized(tmp_path):
    out = tmp_path / "imatrix_proxy.json"
    payload = build_proxy_importance(
        tiny_catalog(), out_path=out, gguf_sha256="abc", calib_path="calib.txt",
    )
    assert "rope_freqs.weight" in payload["excluded_nonfinite"]
    assert "rope_freqs.weight" not in payload["tensors"]
    # Sole finite tensor normalizes to 1.0 (not 0.0 via /inf).
    assert payload["tensors"]["blk.0.attn_q.weight"]["importance"] == 1.0
    # File round-trips the exclusion audit.
    assert json.loads(out.read_text())["excluded_nonfinite"] == [
        "rope_freqs.weight"
    ]
