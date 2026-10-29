"""Same-model experiment: pinned config, metric extraction, comparison table."""

import json
import os
import time

import pytest

import freeze
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


# --- llama.cpp converter cache (offline: fake git on PATH) ---

FAKE_GIT = """#!/usr/bin/env python3
import os, pathlib, sys

args = sys.argv[1:]
if log := os.environ.get("FAKE_GIT_LOG"):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\\n")
if os.environ.get("FAKE_GIT_FAIL"):
    sys.exit(1)
if args[:1] == ["ls-remote"]:
    sys.stdout.write(os.environ.get("FAKE_GIT_TAGS", ""))
elif args[:1] == ["clone"]:
    dest = pathlib.Path(args[-1])
    (dest / ".git").mkdir(parents=True, exist_ok=True)
    (dest / "convert_hf_to_gguf.py").write_text("# fake\\n")
    (dest / "conversion").mkdir(exist_ok=True)
    (dest / "conversion" / "__init__.py").write_text("")
    (dest / "gguf-py" / "gguf").mkdir(parents=True, exist_ok=True)
    (dest / "gguf-py" / "gguf" / "__init__.py").write_text("")
elif "-C" in args and "rev-parse" in args:
    print(os.environ.get("FAKE_GIT_SHA", "a" * 40))
sys.exit(0)
"""


def _wire_git(monkeypatch, tmp_path, *, tags=(), sha="a" * 40, fail=False):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "git"
    exe.write_text(FAKE_GIT, encoding="utf-8")
    exe.chmod(0o755)
    log = tmp_path / "git.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.delenv("LLAMA_CPP", raising=False)
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.setenv("FAKE_GIT_TAGS", "".join(f"0\trefs/tags/{t}\n" for t in tags))
    monkeypatch.setenv("FAKE_GIT_SHA", sha)
    monkeypatch.setenv("FAKE_GIT_FAIL", "1" if fail else "")
    return log


def _seed_cache(tmp_path, *, tag="v0.8.0", sha="a" * 40, checked_at=None):
    tree = tmp_path / "cache" / "odg" / "llama.cpp" / sha
    (tree / "conversion").mkdir(parents=True)
    (tree / "gguf-py" / "gguf").mkdir(parents=True)
    (tree / "convert_hf_to_gguf.py").write_text("# fake\n", encoding="utf-8")
    (tree / "meta.json").write_text(json.dumps({"tag": tag, "sha": sha}), encoding="utf-8")
    state = tmp_path / "cache" / "odg" / "llama.cpp" / "state.json"
    state.write_text(
        json.dumps(
            {
                "tag": tag,
                "sha": sha,
                "checked_at": time.time() if checked_at is None else checked_at,
            }
        ),
        encoding="utf-8",
    )
    return tree / "convert_hf_to_gguf.py"


def _log_text(log):
    return log.read_text(encoding="utf-8") if log.exists() else ""


def test_latest_stable_tag_uses_numeric_order(monkeypatch, tmp_path):
    _wire_git(
        monkeypatch,
        tmp_path,
        tags=("v0.9.9", "v0.10.0", "v0.10.0-rc1", "b11976", "v0.8.0"),
    )
    assert freeze._latest_stable_tag() == "v0.10.0"


def test_converter_cache_hit_never_shells_out(monkeypatch, tmp_path):
    script = _seed_cache(tmp_path)
    log = _wire_git(monkeypatch, tmp_path, tags=("v0.8.0",))
    assert freeze.find_convert_script() == script
    assert _log_text(log) == ""


def test_converter_rechecks_within_ttl_but_reuses_tree(monkeypatch, tmp_path):
    script = _seed_cache(tmp_path, checked_at=0)
    log = _wire_git(monkeypatch, tmp_path, tags=("v0.8.0",))
    assert freeze.find_convert_script() == script
    text = _log_text(log)
    assert "ls-remote" in text and "clone" not in text


def test_converter_fetches_new_stable_tag(monkeypatch, tmp_path):
    _seed_cache(tmp_path, checked_at=0)
    new_sha = "b" * 40
    log = _wire_git(monkeypatch, tmp_path, tags=("v0.8.0", "v0.9.0"), sha=new_sha)
    found = freeze.find_convert_script()
    assert found == tmp_path / "cache" / "odg" / "llama.cpp" / new_sha / "convert_hf_to_gguf.py"
    assert found.is_file()
    assert "clone" in _log_text(log)
    state = json.loads(
        (tmp_path / "cache" / "odg" / "llama.cpp" / "state.json").read_text(encoding="utf-8")
    )
    assert state["tag"] == "v0.9.0" and state["sha"] == new_sha


def test_converter_offline_falls_back_to_cached_tree(monkeypatch, tmp_path):
    script = _seed_cache(tmp_path, checked_at=0)
    _wire_git(monkeypatch, tmp_path, fail=True)
    assert freeze.find_convert_script() == script


def test_converter_offline_with_empty_cache_returns_none(monkeypatch, tmp_path):
    _wire_git(monkeypatch, tmp_path, fail=True)
    assert freeze.find_convert_script() is None


def test_local_checkout_beats_cache(monkeypatch, tmp_path):
    script = _seed_cache(tmp_path)
    local = tmp_path / "home" / "llama.cpp"
    local.mkdir(parents=True)
    (local / "convert_hf_to_gguf.py").write_text("# local\n", encoding="utf-8")
    log = _wire_git(monkeypatch, tmp_path, tags=("v0.8.0",))
    assert freeze.find_convert_script() == local / "convert_hf_to_gguf.py"
    assert _log_text(log) == ""


def test_llama_cpp_dir_beats_cache(monkeypatch, tmp_path):
    _seed_cache(tmp_path)
    other = tmp_path / "checkout"
    other.mkdir()
    (other / "convert_hf_to_gguf.py").write_text("# env\n", encoding="utf-8")
    log = _wire_git(monkeypatch, tmp_path, tags=("v0.8.0",))
    monkeypatch.setenv("LLAMA_CPP_DIR", str(other))
    assert freeze.find_convert_script() == other / "convert_hf_to_gguf.py"
    assert _log_text(log) == ""
