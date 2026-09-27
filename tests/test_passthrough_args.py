"""Passthrough of extra llama.cpp args (--*-args) to the real binaries.

The repo never interprets these (e.g. ``-ngl 99`` for GPU offload on any
llama.cpp backend: CUDA, Vulkan, Metal, ROCm); it appends them verbatim.
"""

import os

import pytest

from cli import split_extra_args


def test_split_extra_args():
    assert split_extra_args("-ngl 99", "--perplexity-args") == ["-ngl", "99"]
    assert split_extra_args("--jinja --n-gpu-layers 99", "--x") == [
        "--jinja", "--n-gpu-layers", "99",
    ]
    assert split_extra_args(None, "--x") is None
    assert split_extra_args("   ", "--x") is None
    with pytest.raises(ValueError, match="--perplexity-args"):
        split_extra_args("'unterminated", "--perplexity-args")


def _touch(path):
    path.write_bytes(b"x")
    os.chmod(path, 0o755)
    return path


def test_build_imatrix_forwards_extra_args(tmp_path, monkeypatch):
    import imatrix

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    calib = tmp_path / "calib.txt"
    calib.write_text("hello world\n")
    fake = _touch(tmp_path / "llama-imatrix")
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        Path(kwargs["outfile"]).write_bytes(b"IMATRIX")
        return "ok"

    from pathlib import Path

    monkeypatch.setattr(imatrix, "run_llama_imatrix", fake_run)
    res = imatrix.build_imatrix(
        model_ref="t", out_dir=tmp_path / "out", gguf_path=gguf,
        calib_path=calib, catalog=None, mode="llama",
        llama_imatrix=fake, extra_args=["-ngl", "99"],
    )
    assert seen.get("extra_args") == ["-ngl", "99"]
    assert res.method == "llama_imatrix"


def test_cache_reference_logits_forwards_extra_args(tmp_path, monkeypatch):
    import logits

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    search = tmp_path / "search.txt"
    search.write_text("hello\n")
    heldout = tmp_path / "heldout.txt"
    heldout.write_text("world\n")
    fake = _touch(tmp_path / "llama-perplexity")
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        Path(kwargs["outfile"]).write_bytes(b"KL")
        return "Mean KLD: 0.1\n"

    from pathlib import Path

    monkeypatch.setattr(logits, "run_kl_divergence_base", fake_run)
    logits.cache_reference_logits(
        model_ref="t", out_dir=tmp_path / "out", gguf_path=gguf,
        search_path=search, heldout_path=heldout, mode="llama",
        llama_perplexity=fake, extra_args=["-ngl", "99"],
    )
    assert len(calls) == 2  # search + heldout
    assert all(c.get("extra_args") == ["-ngl", "99"] for c in calls)


STOCK_LOG = """\
Main:  pass 1 / 1
Final estimate: PPL = 12.157 ± 0.037
Mean KLD: 0.2432 ± 0.011
99.0% KLD: 3.937
Same top p: 89.76%
"""


def test_measure_column_appends_perplexity_args(tmp_path, monkeypatch):
    import llama_probe

    qbin = _touch(tmp_path / "llama-quantize")
    pbin = _touch(tmp_path / "llama-perplexity")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    search = tmp_path / "search.txt"
    search.write_text("hello\n")
    klbase = tmp_path / "kl.bin"
    klbase.write_bytes(b"KL")
    cmds = []

    def fake_run(cmd, *, what):
        cmds.append(cmd)
        if "trial-t0.gguf" in cmd[-2]:
            pass
        return STOCK_LOG

    # trial file must exist after the quantize call: create it lazily
    orig = fake_run

    def fake_run2(cmd, *, what):
        if cmd[0] == str(qbin):
            (tmp_path / "work" / "trial-t0.gguf").write_bytes(b"T")
        return orig(cmd, what=what)

    monkeypatch.setattr(llama_probe, "_run", fake_run2)
    out = llama_probe.measure_column(
        model_gguf=model, group_regex=None, probe_type="Q4_K",
        baseline_type="Q6_K", search_txt=search, kl_base_bin=klbase,
        work_dir=tmp_path / "work", tag="t0",
        llama_quantize=qbin, llama_perplexity=pbin,
        perplexity_args=["-ngl", "99"],
    )
    pcmd = [c for c in cmds if c[0] == str(pbin)]
    assert len(pcmd) == 1
    assert pcmd[0][-2:] == ["-ngl", "99"]
    assert out["kld_mean"] == pytest.approx(0.2432)
    assert out["kld_tail_1pct"] == pytest.approx(3.937)


def _tmap(entries):
    # entries: list of (name, dtype, shape)
    return {n: {"dtype": d, "shape": s} for n, d, s in entries}


def test_assert_probe_applied_ok_and_baseline_skip():
    from llama_probe import assert_probe_applied

    tmap = _tmap([
        ("blk.0.attn_q.weight", "Q2_K", [640, 640]),
        ("blk.0.norm.weight", "F32", [640]),
    ])
    # probe == baseline: no check at all
    assert assert_probe_applied(
        tmap, ["blk.0.attn_q.weight"], "Q6_K", "Q6_K", tag="t") == []
    # 2-D tensor at probe type passes
    assert assert_probe_applied(
        tmap, ["blk.0.attn_q.weight"], "Q2_K", "Q6_K", tag="t") == []
    # 1-D tensor exempt even though it stayed F32
    assert assert_probe_applied(
        tmap, ["blk.0.norm.weight"], "Q2_K", "Q6_K", tag="t") == [
            "blk.0.norm.weight"]


def test_assert_probe_applied_silent_no_match_raises():
    from llama_probe import assert_probe_applied

    tmap = _tmap([("blk.0.ffn_up.weight", "BF16", [640, 640])])
    with pytest.raises(RuntimeError, match="did not take probe"):
        assert_probe_applied(
            tmap, ["blk.0.ffn_up.weight"], "Q2_K", "Q6_K",
            tag="other@global-Q2_K",
        )


def test_assert_probe_applied_missing_tensor_raises():
    from llama_probe import assert_probe_applied

    with pytest.raises(RuntimeError, match="missing from trial"):
        assert_probe_applied({}, ["blk.0.missing.weight"], "Q2_K", "Q6_K")
