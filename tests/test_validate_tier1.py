"""Validate Tier-1: measured llama heldout gates + CLI wiring."""

import argparse
import subprocess

import pytest

import cli
from cli import _pipeline_expected_inputs

KL_LOG = """\
perplexity : 9.000
Mean PPL(Q) : 9.010
Mean KLD: 0.006330
Median KLD: 0.000005
99.0% KLD: 0.101000
99.9% KLD: 0.250000
Maximum KLD: 1.500000
Same top p: 98.300
"""

RECIPE = """\
size_bytes: 5118550976
predicted_mean_delta_kld: -0.00163
gguf_sha256: "{sha}"
target_size_bytes: 5122479400
"""


def _recipe(path, sha="a" * 64):
    path.write_text(RECIPE.format(sha=sha))
    return path


class _Proc:
    def __init__(self, out, code=0):
        self.stdout = out
        self.stderr = ""
        self.returncode = code


def _stub_run_factory(calls, out=KL_LOG, code=0):
    def _fake(cmd, **kw):
        calls.append(cmd)
        return _Proc(out, code)
    return _fake


def test_tier1_llama_pass_shape(tmp_path, monkeypatch):
    import llama_bins
    from validate import _tier1_llama

    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run_factory(calls))
    monkeypatch.setattr(
        llama_bins, "find_llama_binary", lambda name, hint=None: "/bin/llama-perplexity"
    )
    cand = tmp_path / "cand.gguf"
    cand.write_bytes(b"x")
    txt = tmp_path / "heldout.txt"
    txt.write_text("hi\n")
    binp = tmp_path / "logits-heldout.bin"
    binp.write_bytes(b"y")

    t1 = _tier1_llama(
        candidate=cand, heldout_txt=txt, heldout_bin=binp,
        out_dir=tmp_path, extra_args=["-ngl", "99"],
    )
    assert t1["method"] == "llama_heldout"
    assert t1["split"] == "heldout"
    assert t1["metrics"]["mean_kld"] == pytest.approx(0.00633)
    assert t1["metrics"]["p99_kld"] == pytest.approx(0.101)
    assert t1["metrics"]["top1_agree"] == pytest.approx(0.983)
    assert t1["pass"] is True
    assert (tmp_path / "llama-perplexity-tier1.log").is_file()
    cmd = calls[0]
    assert "--kl-divergence" in cmd and "--kl-divergence-base" in cmd
    assert cmd[-2:] == ["-ngl", "99"]


def test_tier1_llama_fail_and_failures(tmp_path, monkeypatch):
    import llama_bins
    from validate import _tier1_llama

    monkeypatch.setattr(
        llama_bins, "find_llama_binary", lambda name, hint=None: "/bin/llama-perplexity"
    )
    cand = tmp_path / "cand.gguf"
    cand.write_bytes(b"x")
    txt = tmp_path / "heldout.txt"
    txt.write_text("hi\n")
    binp = tmp_path / "logits-heldout.bin"
    binp.write_bytes(b"y")

    bad = KL_LOG.replace("Mean KLD: 0.006330", "Mean KLD: 0.500000")
    monkeypatch.setattr(subprocess, "run", _stub_run_factory([], out=bad))
    t1 = _tier1_llama(
        candidate=cand, heldout_txt=txt, heldout_bin=binp, out_dir=tmp_path,
    )
    assert t1["pass"] is False
    assert t1["pass_detail"]["mean_kld"] is False

    # nonzero exit raises
    monkeypatch.setattr(subprocess, "run", _stub_run_factory([], code=1))
    with pytest.raises(RuntimeError, match="exit 1"):
        _tier1_llama(
            candidate=cand, heldout_txt=txt, heldout_bin=binp,
            out_dir=tmp_path,
        )
    # missing KL lines raise via parser
    monkeypatch.setattr(
        subprocess, "run", _stub_run_factory([], out="no kl here\n"))
    with pytest.raises(ValueError, match="missing required KL lines"):
        _tier1_llama(
            candidate=cand, heldout_txt=txt, heldout_bin=binp,
            out_dir=tmp_path,
        )
    # missing binary / missing files raise
    monkeypatch.setattr(
        llama_bins, "find_llama_binary", lambda name, hint=None: None)
    with pytest.raises(RuntimeError, match="llama-perplexity"):
        _tier1_llama(
            candidate=cand, heldout_txt=txt, heldout_bin=binp,
            out_dir=tmp_path,
        )


def test_validate_mode_branches(tmp_path, monkeypatch):
    import llama_bins
    from validate import validate_and_release

    monkeypatch.setattr(
        llama_bins, "find_llama_binary", lambda name, hint=None: "/bin/llama-perplexity"
    )
    monkeypatch.setattr(
        subprocess, "run", _stub_run_factory([], out=KL_LOG))
    recipe = _recipe(tmp_path / "recipe.yaml")
    cand = tmp_path / "cand.gguf"
    cand.write_bytes(b"x" * 1024)
    export = {"gguf_out": str(cand), "method": "llama",
              "gguf_out_nbytes": 5118550976}
    txt = tmp_path / "heldout.txt"
    txt.write_text("hi\n")
    binp = tmp_path / "logits-heldout.bin"
    binp.write_bytes(b"y")
    base_kw = dict(
        model_ref="m", recipe_path=recipe,
    )
    no_cand = dict(export_manifest={"gguf_out": None})

    # llama mode without candidate → hard error
    with pytest.raises(RuntimeError, match="exported candidate GGUF"):
        validate_and_release(
            **base_kw, **no_cand, out_dir=tmp_path / "v1", mode="llama",
        )
    # llama mode without heldout assets → hard error naming step 11
    with pytest.raises(RuntimeError, match="reference-logits"):
        validate_and_release(
            **base_kw, export_manifest=export,
            out_dir=tmp_path / "v2", mode="llama",
        )
    # auto without assets → proxy fallback (existing behavior)
    res = validate_and_release(
        **base_kw, export_manifest=export,
        out_dir=tmp_path / "v3", mode="auto",
    )
    assert res.tier1["method"] == "proxy_from_recipe"
    # llama with assets → measured gates
    res = validate_and_release(
        **base_kw, export_manifest=export,
        out_dir=tmp_path / "v4", mode="llama",
        heldout_txt=txt, heldout_bin=binp,
        perplexity_args=["-ngl", "99"],
    )
    assert res.tier1["method"] == "llama_heldout"
    assert res.tier1["pass"] is True
    assert res.verdict == "RELEASE"


def test_validate_flags_and_stale_key(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(args: argparse.Namespace) -> int:
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "cmd_validate", _fake)
    code = cli.main([
        "--artifacts", str(tmp_path),
        "validate", "--model", "m",
        "--perplexity-args", "-ngl 99",
        "--llama-perplexity", "/bin/x",
    ])
    assert code == 0
    assert seen["args"].perplexity_args == "-ngl 99"
    assert str(seen["args"].llama_perplexity) == "/bin/x"

    base = dict(
        command="run", artifacts=None, model="m", quant=None, no_ask=True,
        new_run=False, run=None, force=False, prefer_hf=False,
        download_weights=False, until=None, from_step=None, quiet=True,
        no_explain=True, validate_mode="llama", strict=True,
        perplexity_args="-ngl 99",
    )
    exp = _pipeline_expected_inputs(
        "validate", argparse.Namespace(**base), fmt=None)
    assert exp["mode"] == "llama"
    assert exp["strict"] is True
    assert exp["perplexity_args"] == ["-ngl", "99"]
