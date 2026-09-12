"""Same-model experiment: pinned config, metric extraction, comparison table."""

import json

import pytest

from experiment import (
    TASKS,
    VARIANTS,
    build_eval_command,
    build_comparison,
    compression_stats,
    config_fingerprint,
    default_config,
    extract_quality,
    locate_source_gguf,
    n_params_from_safetensors,
    parse_llama_bench_json,
    parse_perplexity_log,
    pick_metric,
    render_comparison_markdown,
    run_variant,
)


def test_config_is_pinned_and_does_not_override_shots():
    cfg = default_config()
    assert cfg["model"] == "functiongemma:latest"
    assert tuple(cfg["tasks"]) == TASKS
    assert cfg["batch_size"] == 8
    assert cfg["seed"] == 0
    assert cfg["num_fewshot"] is None  # harness per-task defaults
    assert "mmlu" in cfg["tasks"] and "gsm8k" in cfg["tasks"]


def test_eval_command_is_identical_except_model_args(tmp_path):
    cfg = default_config()
    common = dict(
        cfg=cfg, root=tmp_path, out_dir=tmp_path / "out", limit=32, device="cpu"
    )
    bf16 = build_eval_command(variant="bf16", **common)
    q4 = build_eval_command(variant="q4_k_m", **common)

    def pin(cmd):
        return cmd[cmd.index("--tasks") :]

    assert pin(bf16) == pin(q4)
    assert "--num_fewshot" not in bf16
    assert "--seed" in bf16 and "0" in bf16
    assert "--batch_size" in bf16 and "8" in bf16
    assert "mmlu,gsm8k,hellaswag,arc_challenge,truthfulqa_mc2" in bf16
    assert bf16[bf16.index("--model") + 1] == "gguf"
    assert any("q4_k_m" in a or a.endswith(".gguf") for a in q4)


def test_locate_source_gguf_accepts_a_path(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    assert locate_source_gguf(str(gguf)) == gguf


def test_fingerprint_changes_when_limit_changes():
    cfg = default_config()
    a = config_fingerprint(cfg, suite="dev", limit=32)
    b = config_fingerprint(cfg, suite="paper", limit=None)
    assert a != b
    assert a == config_fingerprint(cfg, suite="dev", limit=32)


def test_pick_metric_prefers_task_primary():
    assert pick_metric("mmlu", {"acc,none": 0.72, "acc_norm,none": 0.8}) == (
        "acc,none",
        0.72,
    )
    assert pick_metric("hellaswag", {"acc,none": 0.5, "acc_norm,none": 0.82}) == (
        "acc_norm,none",
        0.82,
    )
    assert pick_metric("gsm8k", {"exact_match,strict-match": 0.4})[1] == 0.4


def test_extract_quality_uses_groups_for_mmlu():
    raw = {
        "results": {"mmlu_abstract_algebra": {"acc,none": 0.1}},
        "groups": {"mmlu": {"acc,none": 0.55}},
        "n-shot": {"mmlu": 5},
        "versions": {"mmlu": 2},
    }
    q = extract_quality(raw, ["mmlu"])
    assert q["tasks"]["mmlu"]["score"] == 0.55
    assert q["skipped"] is False


def test_compression_stats():
    s = compression_stats(
        nbytes=4_800_000_000, n_params=7_000_000_000, bf16_bytes=14_000_000_000
    )
    assert s["bytes_per_parameter"] == pytest.approx(
        4_800_000_000 / 7_000_000_000, rel=1e-4
    )
    assert s["compression_ratio_vs_bf16"] == pytest.approx(14 / 4.8, rel=1e-3)


def test_parse_perplexity_and_llama_bench():
    ppl = parse_perplexity_log(
        "Final estimate: PPL = 12.345 ± 0.1\nMean KL divergence: 0.0123\nMean n_same_top: 0.987\n"
    )
    assert ppl["perplexity"] == 12.345
    assert ppl["kl_divergence"] == 0.0123
    assert ppl["token_agreement"] == 0.987

    bench = parse_llama_bench_json(
        "noise "
        + json.dumps(
            [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.5, "backends": "Metal"},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 20.25},
            ]
        )
    )
    assert bench["pp_tps"] == 100.5
    assert bench["tg_tps"] == 20.25
    assert parse_llama_bench_json("not json") is None


def test_n_params_from_safetensors_header(tmp_path):
    header = {
        "tok.weight": {"dtype": "BF16", "shape": [100, 8], "data_offsets": [0, 1600]},
        "__metadata__": {"x": "y"},
    }
    blob = json.dumps(header).encode()
    p = tmp_path / "w.safetensors"
    p.write_bytes(len(blob).to_bytes(8, "little") + blob + b"\x00" * 16)
    assert n_params_from_safetensors(p) == 800


def _write_result(root, variant, scores, nbytes):
    d = root / "results" / variant
    d.mkdir(parents=True)
    payload = {
        "schema": "odg/experiment/v1",
        "variant": variant,
        "label": variant,
        "config_fingerprint": "abc",
        "quality": {
            "skipped": False,
            "tasks": {
                tid: {"score": sc, "metric": "acc,none"} for tid, sc in scores.items()
            },
        },
        "behavior": {"skipped": True},
        "throughput": {"pp_tps": 10.0, "tg_tps": 2.0},
        "memory": {
            "bytes": nbytes,
            "mb": nbytes / (1024**2),
            "bytes_per_parameter": 4.0,
        },
    }
    (d / "result.json").write_text(json.dumps(payload))


def test_compare_builds_table(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(default_config()))
    scores = {
        "mmlu": 0.70,
        "gsm8k": 0.40,
        "hellaswag": 0.50,
        "arc_challenge": 0.30,
        "truthfulqa_mc2": 0.25,
    }
    _write_result(tmp_path, "bf16", scores, 500_000_000)
    q4 = dict(scores)
    q4["mmlu"] = 0.65
    _write_result(tmp_path, "q4_k_m", q4, 180_000_000)
    odg = dict(scores)
    odg["mmlu"] = 0.68
    _write_result(tmp_path, "odg", odg, 185_000_000)

    (tmp_path / "manifest.json").write_text(
        json.dumps({"n_params": 268_000_000, "bf16_bytes": 500_000_000})
    )
    comp = build_comparison(tmp_path)
    assert comp["pin_ok"] is True
    mmlu = next(r for r in comp["quality"] if r["task"] == "mmlu")
    assert mmlu["scores"]["bf16"] == 0.70
    assert mmlu["best_quantized"] == "odg"
    md = render_comparison_markdown(comp)
    assert "MMLU" in md and "Q4_K_M" in md and "OpenDynamicGGUF" in md
    assert "**68.0**" in md  # ODG best among quantized on MMLU


def test_unknown_variant_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown variant"):
        run_variant("nope", tmp_path)


def test_variants_cover_the_claimed_baselines():
    assert VARIANTS == ("bf16", "q4_k_m", "q5_k_m", "q6_k", "odg")
