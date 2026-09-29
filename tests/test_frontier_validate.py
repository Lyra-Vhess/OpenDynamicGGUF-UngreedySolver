"""Step-15 frontier mode: curated recipe set, transient exports, plots."""
import argparse
import json
from pathlib import Path

import pytest

import cli
from cli import _pipeline_expected_inputs, _validate_frontier
from pareto_sweep import parse_recipe_points, run_sweep


def _recipe(path: Path, *, budget: int, pred: float, bodies: dict[str, str]):
    lines = [
        "schema: odg/recipe/v1",
        "budget:",
        f"  target_size_bytes: {budget}",
        "base_type: q6_k",
        "overrides:",
        *(f'  "{rx}": {q}' for rx, q in bodies.items()),
        "estimate:",
        f"  size_bytes: {budget}",
        f"  predicted_mean_delta_kld: {pred:.6f}",
        "validation: {}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_parse_patterns_include_frontier(tmp_path):
    _recipe(tmp_path / "pareto-00-100k.yaml", budget=100,
            pred=0.02, bodies={"a": "Q4_K"})
    _recipe(tmp_path / "frontier-bpw-4.yaml", budget=200,
            pred=0.01, bodies={"a": "Q5_K"})
    pareto_only = parse_recipe_points(tmp_path)
    assert [Path(p["path"]).name for p in pareto_only] == ["pareto-00-100k.yaml"]
    both = parse_recipe_points(tmp_path, ("pareto-*.yaml", "frontier-*.yaml"))
    assert len(both) == 2


def _stub_export(*, recipe_yaml, recipe_tt, gguf_in, imatrix, out_path,
                 base_type, llama_quantize):
    out_path = Path(out_path)
    out_path.write_bytes(b"gguf-stub")
    return str(out_path), 9


def _stub_measure(candidate, *, heldout_txt, heldout_bin, out_dir, tag,
                  llama_perplexity=None, extra_args=None):
    return {
        "method": "stub",
        "metrics": {"mean_kld": 0.01, "p99_kld": 0.1,
                    "top1_agree": 0.98},
        "perplexity": 9.0,
    }


def test_sweep_curated_files_drop_winner(tmp_path):
    opt = tmp_path / "opt"
    opt.mkdir()
    f1 = _recipe(opt / "frontier-bpw-3.yaml", budget=100,
                 pred=0.02, bodies={"a": "Q4_K"})
    f2 = _recipe(opt / "recipe.yaml", budget=200,
                 pred=0.01, bodies={"a": "Q4_K"})
    rec = run_sweep(
        optimize_dir=opt, gguf_in=tmp_path / "f.gguf", imatrix=None,
        heldout_txt=tmp_path / "h.txt", heldout_bin=tmp_path / "h.bin",
        out_dir=tmp_path / "front", recipe_files=[f1, f2],
        keep_winner=False, export_fn=_stub_export,
        measure_fn=_stub_measure,
    )
    # Exact-duplicate allocations fold: one measured point.
    assert rec["n_measured"] == 1
    assert rec["winner"]["kept_gguf"] is None
    assert list((tmp_path / "front").rglob("*.gguf")) == []
    assert (tmp_path / "front" / "frontier.json").is_file()
    assert len(rec["plots"]) == 4


def _stub_store(tmp_path, manifest):
    opt = tmp_path / "opt"
    (opt / "pareto").mkdir(parents=True)
    (opt / "optimize_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")

    class S:
        def step_path(self, run_id, step):
            return opt if step == "optimize" else tmp_path / step

        def read_step_output(self, run_id, step):
            return {}

    return S(), opt


def test_frontier_gate_requires_exhaustive(tmp_path):
    store, _opt = _stub_store(tmp_path, {
        "primary": {"certificate": {"mode": "bounded"}},
        "frontier": {"selection": {}},
    })
    meta = argparse.Namespace(run_id="r")
    with pytest.raises(ValueError, match="exhaustive"):
        _validate_frontier(
            store, meta, tmp_path / "val", {"primary": {
                "certificate": {"mode": "bounded"}}},
            {}, {}, argparse.Namespace(), keep_winner=False,
        )


def test_validate_frontier_flags(monkeypatch, tmp_path):
    seen: dict[str, argparse.Namespace] = {}

    def fake(name):
        def _fn(args: argparse.Namespace) -> int:
            seen[name] = args
            return 0
        return _fn

    monkeypatch.setattr(cli, "cmd_validate", fake("cmd_validate"))
    assert cli.main(["validate", "--model", "m", "--frontier",
                     "--budget-mb", "frontier"]) == 0
    assert seen["cmd_validate"].frontier is True
    assert seen["cmd_validate"].budget_mb == "frontier"

    for step_fn in ("cmd_resolve", "cmd_load", "cmd_enumerate", "cmd_classify",
                    "cmd_catalog", "cmd_weight_features", "cmd_corpus",
                    "cmd_activation_features", "cmd_freeze_gguf", "cmd_imatrix",
                    "cmd_reband", "cmd_reference_logits", "cmd_sensitivity",
                    "cmd_optimize", "cmd_export", "cmd_validate"):
        monkeypatch.setattr(cli, step_fn, fake(step_fn))
    code = cli.main([
        "--artifacts", str(tmp_path),
        "run", "--model", "m", "--quant", "q4_k_m", "--no-ask",
        "--until", "validate", "--frontier",
    ])
    assert code == 0
    assert seen["cmd_validate"].frontier is True

    exp = _pipeline_expected_inputs(
        "validate", argparse.Namespace(frontier=True, budget_mb="frontier"),
        fmt=None)
    assert exp["frontier"] is True
    assert exp["budget_mb"] == "frontier"
    exp = _pipeline_expected_inputs("validate", argparse.Namespace(), fmt=None)
    assert exp["frontier"] is False
    assert exp["budget_mb"] is None
