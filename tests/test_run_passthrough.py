"""Single-command wiring: `odg run` / `odg fit` flag parity + stale-checkpoint guard.

`odg run` must forward the newer step flags (jobs, fixed-groups,
perplexity/imatrix passthrough, probe grid, certificate choices, ...) instead
of silently running every step on hardcoded defaults. When a step is already
checkpointed as done but the pipeline flags disagree with its recorded
input.json, the pipeline warns and asks before re-running (warn-and-confirm);
non-interactive sessions warn and keep the checkpoint.
"""

import argparse

import pytest

import cli
from cli import (
    _check_stale_step,
    _pipeline_expected_inputs,
)


def _pipeline_ns(**over):
    base = dict(
        command="run",
        artifacts=None,
        model="m",
        quant=None,
        no_ask=True,
        new_run=False,
        run=None,
        force=False,
        prefer_hf=False,
        download_weights=False,
        until=None,
        from_step=None,
        quiet=True,
        no_explain=True,
        mode="auto",
        target_tokens=50_000,
        seed=42,
        max_docs=32,
        freeze_mode="auto",
        convert_script=None,
        require_bf16=False,
        chunks=64,
        imatrix_args=None,
        perplexity_args=None,
        bands_per_role=3,
        jobs=1,
        probe_types=None,
        fixed_groups=None,
        optimizer="dp_mckp",
        kld_objective="mean",
        certificate="bounded",
        lipschitz=None,
        pareto_ratios=None,
        budget_mb=None,
        budget_ratio=None,
        export_mode="auto",
        base_type=None,
        validate_mode="auto",
        strict=False,
        only_quantizable=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_expected_inputs_sensitivity():
    args = _pipeline_ns(
        jobs=3, perplexity_args="-ngl 99", fixed_groups="other@global",
        probe_types="Q2_K,Q4_K", mode="llama",
    )
    exp = _pipeline_expected_inputs("sensitivity", args, fmt=None)
    assert exp["mode"] == "llama"
    assert exp["perplexity_args"] == ["-ngl", "99"]
    assert exp["fixed_groups"] == ["other@global"]
    assert exp["probe_types"] == ["Q2_K", "Q4_K"]
    # Lazy-pricing knobs are result-affecting: part of the comparison.
    assert exp["kld_objective"] == "mean"
    assert exp["certificate"] == "bounded"
    assert exp["lipschitz"] is None
    exp2 = _pipeline_expected_inputs(
        "sensitivity",
        _pipeline_ns(
            kld_objective="tail_1pct", certificate="exhaustive",
            lipschitz=2.5),
        fmt=None,
    )
    assert     exp2["kld_objective"] == "tail_1pct"
    assert exp2["certificate"] == "exhaustive"
    assert exp2["lipschitz"] == 2.5
    # Intended solve budget steers pricing tightness: result-affecting.
    from quant_formats import get_format as _gf

    exp3 = _pipeline_expected_inputs(
        "sensitivity", _pipeline_ns(), _gf("q4_k_m"))
    assert exp3["budget_mb"] is None
    assert exp3["budget_ratio"] == pytest.approx(0.72)
    exp4 = _pipeline_expected_inputs(
        "sensitivity", _pipeline_ns(budget_mb=4885.0), _gf("q4_k_m"))
    assert exp4["budget_mb"] == pytest.approx(4885.0)
    # jobs is parallelism-only: never part of the comparison.
    assert "jobs" not in exp
    # Without a resolved format, format-derived keys are omitted, not guessed.
    assert "baseline" not in exp
    assert "quant_format" not in exp


def test_expected_inputs_optimize_budget_resolution():
    from quant_formats import get_format

    fmt = get_format("q4_k_m")
    exp = _pipeline_expected_inputs("optimize", _pipeline_ns(), fmt)
    assert exp["budget_mb"] is None
    assert exp["budget_ratio"] == pytest.approx(float(fmt.budget_ratio))
    assert exp["optimizer"] == "dp_mckp"
    assert exp["fixed_groups"] == []
    exp2 = _pipeline_expected_inputs(
        "optimize", _pipeline_ns(budget_mb=4888.0, budget_ratio=0.72), fmt
    )
    assert exp2["budget_mb"] == pytest.approx(4888.0)
    assert exp2["budget_ratio"] == pytest.approx(0.72)


def test_expected_inputs_chunks_normalization():
    assert _pipeline_expected_inputs("imatrix", _pipeline_ns(), None)["chunks"] == 64
    assert _pipeline_expected_inputs(
        "imatrix", _pipeline_ns(chunks=0), None)["chunks"] is None


def test_expected_inputs_bad_passthrough_raises():
    with pytest.raises(ValueError, match="--perplexity-args"):
        _pipeline_expected_inputs(
            "sensitivity", _pipeline_ns(perplexity_args="'unterminated"), None
        )


def _done_step(store, run_id, step_id, input_data):
    store.begin_step(run_id, step_id, input_data)
    store.complete_step(run_id, step_id, {"ok": True})


def test_stale_check_matching_inputs_no_prompt(monkeypatch, tmp_path):
    from store import RunStore

    store = RunStore(tmp_path)
    meta = store.create_run("m")
    recorded = {"mode": "llama", "jobs": 2,
                "perplexity_args": ["-ngl", "99"], "fixed_groups": []}
    _done_step(store, meta.run_id, "sensitivity", recorded)

    def no_input(_prompt=""):
        raise AssertionError("must not prompt when inputs match")

    monkeypatch.setattr("builtins.input", no_input)
    expected = {"mode": "llama", "perplexity_args": ["-ngl", "99"],
                "fixed_groups": []}
    assert _check_stale_step(store, meta.run_id, "sensitivity", expected,
                             interactive=True) is False


def test_stale_check_mismatch_confirm_yes(monkeypatch, tmp_path):
    from store import RunStore

    store = RunStore(tmp_path)
    meta = store.create_run("m")
    _done_step(store, meta.run_id, "optimize",
               {"optimizer": "greedy", "budget_mb": None})
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    expected = {"optimizer": "dp_mckp", "budget_mb": None}
    assert _check_stale_step(store, meta.run_id, "optimize", expected,
                             interactive=True) is True


def test_stale_check_mismatch_confirm_no(monkeypatch, tmp_path):
    from store import RunStore

    store = RunStore(tmp_path)
    meta = store.create_run("m")
    _done_step(store, meta.run_id, "optimize",
               {"optimizer": "greedy", "budget_mb": None})
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    expected = {"optimizer": "dp_mckp", "budget_mb": None}
    assert _check_stale_step(store, meta.run_id, "optimize", expected,
                             interactive=True) is False


def test_stale_check_noninteractive_keeps_checkpoint(monkeypatch, tmp_path):
    from store import RunStore

    store = RunStore(tmp_path)
    meta = store.create_run("m")
    _done_step(store, meta.run_id, "reband", {"bands_per_role": 3})

    def no_input(_prompt=""):
        raise AssertionError("must not prompt when non-interactive")

    monkeypatch.setattr("builtins.input", no_input)
    assert _check_stale_step(store, meta.run_id, "reband",
                             {"bands_per_role": 4},
                             interactive=False) is False


def test_stale_check_not_done_or_no_expected(tmp_path):
    from store import RunStore

    store = RunStore(tmp_path)
    meta = store.create_run("m")
    # Step never ran: no input.json, not done → proceed normally, no prompt.
    assert _check_stale_step(store, meta.run_id, "sensitivity",
                             {"mode": "llama"},
                             interactive=True) is False
    assert _check_stale_step(store, meta.run_id, "sensitivity", {},
                             interactive=True) is False


def test_stale_check_ignores_keys_missing_from_record(tmp_path):
    # Old checkpoints predate some keys; comparison is limited to keys the
    # recorded input.json actually has.
    from store import RunStore

    store = RunStore(tmp_path)
    meta = store.create_run("m")
    _done_step(store, meta.run_id, "sensitivity", {"mode": "auto"})
    assert _check_stale_step(
        store, meta.run_id, "sensitivity",
        {"mode": "auto", "fixed_groups": ["other@global"]},
        interactive=True,
    ) is False


def test_run_threads_flags_to_step_namespaces(monkeypatch, tmp_path):
    seen: dict[str, argparse.Namespace] = {}

    def fake(name):
        def _fn(args: argparse.Namespace) -> int:
            seen[name] = args
            return 0
        return _fn

    for step_fn in ("cmd_resolve", "cmd_load", "cmd_enumerate", "cmd_classify",
                    "cmd_catalog", "cmd_weight_features", "cmd_corpus",
                    "cmd_activation_features", "cmd_freeze_gguf", "cmd_imatrix",
                    "cmd_reband", "cmd_reference_logits", "cmd_sensitivity",
                    "cmd_optimize", "cmd_export", "cmd_validate"):
        monkeypatch.setattr(cli, step_fn, fake(step_fn))

    code = cli.main([
        "--artifacts", str(tmp_path),
        "run", "--model", "m", "--quant", "q4_k_m", "--no-ask",
        "--until", "validate",
        "--jobs", "3",
        "--fixed-groups", "other@global",
        "--perplexity-args", "-ngl 99",
        "--imatrix-args", "-ngl 99",
        "--probe-types", "Q2_K,Q4_K",
        "--certificate", "exhaustive",
        "--bands-per-role", "4",
        "--budget-mb", "4888",
        "--strict",
    ])
    assert code == 0

    sens = seen["cmd_sensitivity"]
    assert sens.jobs == 3
    assert sens.fixed_groups == "other@global"
    assert sens.perplexity_args == "-ngl 99"
    assert sens.probe_types == "Q2_K,Q4_K"
    # Lazy-pricing knobs reach sensitivity (shared step namespace).
    assert sens.certificate == "exhaustive"
    assert sens.kld_objective == "mean"
    assert sens.lipschitz is None

    opt = seen["cmd_optimize"]
    assert opt.jobs == 3
    assert opt.fixed_groups == "other@global"
    assert opt.certificate == "exhaustive"
    assert opt.budget_mb == pytest.approx(4888.0)

    assert seen["cmd_imatrix"].imatrix_args == "-ngl 99"
    assert seen["cmd_reference_logits"].perplexity_args == "-ngl 99"
    assert seen["cmd_reband"].bands_per_role == 4
    assert seen["cmd_export"].mode == "auto"
    assert seen["cmd_validate"].strict is True
    # Global probe mode reaches the llama-capable steps untouched.
    assert seen["cmd_imatrix"].mode == "auto"
    assert seen["cmd_sensitivity"].mode == "auto"


def test_no_pins_flag_is_gone():
    """--no-pins was removed with the role pins: all parsers reject it."""
    for argv in (
        ["optimize", "--model", "m", "--no-pins"],
        ["run", "--model", "m", "--no-pins"],
        ["fit", "--model", "m", "--device", "dummy", "--no-pins"],
        ["sensitivity", "--model", "m", "--no-pins"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_run_per_step_modes(monkeypatch, tmp_path):
    seen: dict[str, argparse.Namespace] = {}

    def fake(name):
        def _fn(args: argparse.Namespace) -> int:
            seen[name] = args
            return 0
        return _fn

    for step_fn in ("cmd_resolve", "cmd_load", "cmd_enumerate", "cmd_classify",
                    "cmd_catalog", "cmd_weight_features", "cmd_corpus",
                    "cmd_activation_features", "cmd_freeze_gguf", "cmd_imatrix",
                    "cmd_reband", "cmd_reference_logits", "cmd_sensitivity",
                    "cmd_optimize", "cmd_export", "cmd_validate"):
        monkeypatch.setattr(cli, step_fn, fake(step_fn))

    code = cli.main([
        "--artifacts", str(tmp_path),
        "run", "--model", "m", "--no-ask", "--until", "validate",
        "--mode", "llama",
        "--export-mode", "dry-run",
        "--validate-mode", "proxy",
    ])
    assert code == 0
    assert seen["cmd_sensitivity"].mode == "llama"
    assert seen["cmd_imatrix"].mode == "llama"
    assert seen["cmd_reference_logits"].mode == "llama"
    assert seen["cmd_export"].mode == "dry-run"
    assert seen["cmd_validate"].mode == "proxy"
