"""Step-15 frontier mode: recipe parsing, dedupe, pick, transient exports."""
import argparse
import json
from pathlib import Path

import pytest

import cli
from cli import _pipeline_expected_inputs, _validate_frontier
from validate import dedupe_points, parse_recipe_points, pick_winner, run_sweep


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


OLD_OPT = Path(
    "artifacts/runs/20260927-091332-home-lyra-ai-opencode-"
    "opendynamicgguf-artifacts-/steps/13_optimize"
)
needs_old = pytest.mark.skipif(
    not (OLD_OPT / "pareto").is_dir(), reason="old optimize dir missing")


@needs_old
def test_parse_real_pareto_recipes():
    from optimizer import LADDER

    pts = parse_recipe_points(OLD_OPT / "pareto")
    assert len(pts) == 12
    ladder = set(LADDER)
    for p in pts:
        assert p["budget_bytes"] > 0
        assert len(p["overrides"]) > 20  # banded recipe: ~25 groups
        assert {v.upper() for v in p["overrides"].values()} <= ladder
    assert pts[0]["predicted_mean_kld"] is not None


@needs_old
def test_dedupe_real_recipes_with_manifest():
    manifest = json.loads((OLD_OPT / "optimize_manifest.json").read_text())
    hashes = {int(e["budget_bytes"]): str(e["allocation_hash"])
              for e in manifest["pareto"]
              if e.get("feasible") and e.get("allocation_hash")}
    pts = parse_recipe_points(OLD_OPT / "pareto")
    feas = [p for p in pts if p["budget_bytes"] in hashes]
    kept = dedupe_points(feas, hashes)
    # The old manifest shows two budgets sharing one hash (dup solve).
    assert len(kept) < len(feas)
    assert len({k for k in
                [hashes[p["budget_bytes"]] for p in kept]}) == len(kept)


def test_dedupe_without_manifest_by_recipe_text():
    base = {"a": "Q4_K", "b": "Q6_K"}
    pts = [
        {"budget_bytes": 100, "overrides": dict(base)},
        {"budget_bytes": 200, "overrides": dict(base)},
        {"budget_bytes": 300, "overrides": {"a": "Q2_K", "b": "Q6_K"}},
    ]
    kept = dedupe_points(pts)
    assert [p["budget_bytes"] for p in kept] == [100, 300]


def _m(mean, nbytes):
    return {"actual_bytes": nbytes,
            "tier1": {"metrics": {"mean_kld": mean}}}


def test_pick_winner_lowest_mean_under_cap():
    ms = [_m(0.02, 100), _m(0.01, 200), _m(0.005, 300)]
    assert pick_winner(ms, size_cap=250)["actual_bytes"] == 200
    assert pick_winner(ms, size_cap=50) is None
    assert pick_winner([], size_cap=10**12) is None


def _fake_yaml(path: Path, budget: int, predicted: float,
               overrides: dict[str, str]) -> None:
    lines = ["schema: odg/recipe/v1",
             f"  target_size_bytes: {budget}",
             "overrides:"]
    lines += [f'  "{rx}": {q}' for rx, q in sorted(overrides.items())]
    lines += ["estimate:",
              f"  size_bytes: {budget - 1000}",
              f"  predicted_mean_delta_kld: {predicted:.6f}",
              "validation: {}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_full_flow_keeps_only_winner(tmp_path):
    opt = tmp_path / "opt"
    (opt / "pareto").mkdir(parents=True)
    over_a = {"g1": "Q4_K", "g2": "Q6_K"}
    over_b = {"g1": "Q2_K", "g2": "Q6_K"}
    _fake_yaml(opt / "pareto" / "pareto-00-100k.yaml", 102400, 0.02, over_a)
    _fake_yaml(opt / "pareto" / "pareto-01-200k.yaml", 204800, 0.01, over_a)
    _fake_yaml(opt / "pareto" / "pareto-02-300k.yaml", 307200, 0.005, over_b)
    (opt / "optimize_manifest.json").write_text(json.dumps({"pareto": [
        {"budget_bytes": 102400, "feasible": True, "allocation_hash": "aa"},
        {"budget_bytes": 204800, "feasible": True, "allocation_hash": "aa"},
        {"budget_bytes": 307200, "feasible": True, "allocation_hash": "bb"},
    ]}), encoding="utf-8")

    means = {102400: 0.02, 307200: 0.005}

    def fake_export(*, recipe_yaml, recipe_tt, gguf_in, imatrix,
                    out_path, base_type, llama_quantize):
        out_path.write_bytes(b"G" * 1000)
        return str(out_path), 1000

    def fake_measure(candidate, *, heldout_txt, heldout_bin, out_dir,
                     tag, llama_perplexity, extra_args):
        b = 102400 if "00-" in tag else 307200
        return {"method": "fake", "split": "heldout",
                "metrics": {"mean_kld": means[b], "p99_kld": 0.05,
                            "p999_kld": None, "max_kld": None,
                            "top1_agree": 0.99},
                "pass": True, "pass_detail": {},
                "perplexity": 9.0}

    rec = run_sweep(
        optimize_dir=opt, gguf_in=tmp_path / "f.gguf", imatrix=None,
        heldout_txt=tmp_path / "h.txt", heldout_bin=tmp_path / "h.bin",
        out_dir=tmp_path / "sweep", size_cap=10**9,
        export_fn=fake_export, measure_fn=fake_measure,
    )
    # Dedupe collapsed the dup solve; winner = lowest measured mean.
    assert rec["n_measured"] == 2
    assert rec["winner"]["budget_bytes"] == 307200
    ggufs = list((tmp_path / "sweep").rglob("*.gguf"))
    assert [g.name for g in ggufs] == ["best-under-1000MB.gguf"]
    assert (tmp_path / "sweep" / "best-under-1000MB.recipe.yaml").is_file()
    assert (tmp_path / "sweep" / "best-under-1000MB.recipe.tt").is_file()
    assert (tmp_path / "sweep" / "best-under-1000MB.provenance.json").is_file()
    plots = rec["plots"]
    assert len(plots) == 4 and all(Path(p).is_file() for p in plots)
    front = json.loads((tmp_path / "sweep" / "frontier.json").read_text())
    assert front["winner"]["kept_gguf"].endswith("best-under-1000MB.gguf")
