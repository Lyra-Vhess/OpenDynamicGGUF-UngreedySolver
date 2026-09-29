"""odg export --recipe: select a frontier/pareto recipe for export."""
import argparse

import pytest

import cli
from cli import _pipeline_expected_inputs, _resolve_recipe
from optimizer import (
    parse_recipe_overrides,
    render_recipe_yaml,
    render_tt_from_overrides,
)


def _recipe_text(tmp_path, extras=None):
    groups = {
        "g1": {"tensor_names": ["blk.0.ffn_up.weight"]},
        "g2": {"tensor_names": ["blk.0.ffn_down.weight"]},
    }
    text = render_recipe_yaml(
        model_ref="m", hf_repo_id=None, gguf_sha256="ab",
        imatrix_sha256=None, corpus_id=None, budget_bytes=1000,
        base_type="Q6_K", assignments={"g1": "Q4_K", "g2": "Q5_K"},
        groups=groups, estimated_bytes=900, predicted_delta_kld=0.01,
        method="dp_mckp_colgen_v1", extras=extras,
    )
    p = tmp_path / "recipe.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_overrides_round_trip_with_extras(tmp_path):
    """Quoted-key parsing survives extras with nested allocation lists."""
    extras = {
        "optimizer": "dp_mckp",
        "allocation": [
            {"group": "g1", "type": "Q4_K", "bytes": 100,
             "kld_tail": 0.02},
        ],
        "pareto": [{"budget_bytes": 1000, "feasible": True}],
    }
    p = _recipe_text(tmp_path, extras=extras)
    out = parse_recipe_overrides(p)
    # Keys are tensor regexes (dots escaped) — they feed the .tt as-is.
    assert out == {
        "blk\\.(0)\\.ffn_up\\.weight": "Q4_K",
        "blk\\.(0)\\.ffn_down\\.weight": "Q5_K",
    }
    tt = render_tt_from_overrides(out)
    assert "blk\\.(0)\\.ffn_up\\.weight=q4_k" in tt.splitlines()
    assert "blk\\.(0)\\.ffn_down\\.weight=q5_k" in tt.splitlines()


def test_overrides_empty_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("schema: odg/recipe/v1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="[Nn]o overrides"):
        parse_recipe_overrides(p)


def test_resolve_recipe_search_order(tmp_path):
    opt = tmp_path / "optimize"
    (opt / "pareto").mkdir(parents=True)
    (opt / "recipe.yaml").write_text("x", encoding="utf-8")
    (opt / "pareto" / "frontier-bpw-4.yaml").write_text("y", encoding="utf-8")
    elsewhere = tmp_path / "other.yaml"
    elsewhere.write_text("z", encoding="utf-8")
    assert _resolve_recipe(opt, "frontier-bpw-4.yaml").name == (
        "frontier-bpw-4.yaml")
    assert _resolve_recipe(opt, "recipe.yaml").parent == opt
    assert _resolve_recipe(opt, str(elsewhere)) == elsewhere
    with pytest.raises(ValueError, match="Available step-13 recipes"):
        _resolve_recipe(opt, "nope.yaml")


def test_recipe_flag_parity(monkeypatch, tmp_path):
    seen: dict[str, argparse.Namespace] = {}

    def fake(name):
        def _fn(args: argparse.Namespace) -> int:
            seen[name] = args
            return 0
        return _fn

    monkeypatch.setattr(cli, "cmd_export", fake("cmd_export"))
    assert cli.main(["export", "--model", "m",
                     "--recipe", "frontier-bpw-4.yaml"]) == 0
    assert seen["cmd_export"].recipe == "frontier-bpw-4.yaml"

    monkeypatch.setattr(cli, "cmd_fit", fake("cmd_fit"))
    assert cli.main(["fit", "--model", "m", "--device", "d",
                     "--recipe", "frontier-bpw-4.yaml"]) == 0
    assert seen["cmd_fit"].recipe == "frontier-bpw-4.yaml"

    for step_fn in ("cmd_resolve", "cmd_load", "cmd_enumerate", "cmd_classify",
                    "cmd_catalog", "cmd_weight_features", "cmd_corpus",
                    "cmd_activation_features", "cmd_freeze_gguf", "cmd_imatrix",
                    "cmd_reband", "cmd_reference_logits", "cmd_sensitivity",
                    "cmd_optimize", "cmd_export", "cmd_validate"):
        monkeypatch.setattr(cli, step_fn, fake(step_fn))
    code = cli.main([
        "--artifacts", str(tmp_path),
        "run", "--model", "m", "--quant", "q4_k_m", "--no-ask",
        "--until", "validate", "--recipe", "pareto-03-5120k.yaml",
    ])
    assert code == 0
    assert seen["cmd_export"].recipe == "pareto-03-5120k.yaml"


def test_recipe_in_export_stale_check():
    exp = _pipeline_expected_inputs(
        "export", argparse.Namespace(recipe="frontier-bpw-4.yaml"), fmt=None)
    assert exp["recipe"] == "frontier-bpw-4.yaml"
    exp = _pipeline_expected_inputs("export", argparse.Namespace(), fmt=None)
    assert exp["recipe"] == "primary"
