"""Splice-farm trial backend (splice_farm.py + step-12 wiring)."""
from __future__ import annotations

from pathlib import Path

import pytest

import sensitivity
import splice_farm
from splice_farm import (
    ASSEMBLY_PREFIX,
    MAP_FILENAME,
    SpliceFarm,
    assemble_trial,
    load_farm,
    measure_farm_column,
)

FARM_ROOT = Path("/home/lyra/splice-farm")
needs_farm = pytest.mark.skipif(
    not FARM_ROOT.is_dir(), reason="splice farm not on disk",
)

FAKE_KL_LOG = (
    "MMLU metric stub\n"
    "Mean KLD: 0.003369\n"
    "99.0% KLD: 0.050000\n"
    "99.9% KLD: 0.100000\n"
    "Maximum KLD: 0.200000\n"
    "Median KLD: 0.002000\n"
    "Mean PPL(Q) : 9.000000\n"
)


# --- synthetic mini-farm (no GPU, no real GGUF) -----------------------------

def _mini_farm(tmp_path: Path, *, n_shards: int = 3) -> Path:
    """Farm-shaped directory tree with empty shard files.

    Header reads are patched per-test (gguf_tensors.gguf_tensor_map), so
    the files only need the right names in the right rung dirs.
    """
    from splice_farm import RUNG_SUBDIR, _shard_name

    root = tmp_path / "farm"
    for sub in set(RUNG_SUBDIR.values()):
        d = root / sub
        d.mkdir(parents=True)
        for idx in range(1, n_shards + 1):
            (d / _shard_name(idx, n_shards)).write_bytes(b"")
    return root


def _patch_headers(monkeypatch, n_shards, *, probe_dtype="Q4_K"):
    """gguf_tensor_map: shard idx i holds tensor t{i} (2-D, N bytes)."""
    import gguf_tensors

    def fake(path):
        stem = Path(str(path)).name
        idx = int(stem.split("-")[1])
        name = f"t{idx}"
        dtype = probe_dtype if "rung720-q4_k" in str(path) else "Q8_0"
        nbytes = 1000 * idx
        return {"tensors": {name: {
            "name": name, "shape": [100, 10], "dtype": dtype,
            "ggml_type": 12, "offset": 0,
            "n_elements": 1000, "nbytes": nbytes,
        }}}

    monkeypatch.setattr(gguf_tensors, "gguf_tensor_map", fake)


def test_load_rejects_missing_dir(tmp_path):
    with pytest.raises(RuntimeError, match="[Nn]ot found"):
        load_farm(tmp_path / "nope")


def test_load_rejects_uneven_sets(tmp_path):
    root = _mini_farm(tmp_path)
    (root / "rung720-q4_k" / "q-00001-of-00003.gguf").unlink()
    with pytest.raises(RuntimeError, match="[Uu]neven"):
        load_farm(root)


def test_load_rejects_bad_basenames(tmp_path):
    root = _mini_farm(tmp_path)
    bad = root / "bg720-q8_0" / "q-00001-of-00003.gguf"
    bad.rename(root / "bg720-q8_0" / "shard1.gguf")
    with pytest.raises(RuntimeError, match="[Bb]asenames"):
        load_farm(root)


def test_map_build_scan_and_cache(tmp_path, monkeypatch):
    _patch_headers(monkeypatch, 3)
    root = _mini_farm(tmp_path, n_shards=3)
    farm = load_farm(root)
    assert farm.n_shards == 3
    assert farm.tensor_to_idx == {"t1": 1, "t2": 2, "t3": 3}
    assert (root / MAP_FILENAME).is_file()

    # Second load hits the cache: break the scanner, map still loads.
    import gguf_tensors

    def boom(path):
        raise AssertionError("scanner should not run on cache hit")

    monkeypatch.setattr(gguf_tensors, "gguf_tensor_map", boom)
    farm2 = load_farm(root)
    assert farm2.tensor_to_idx == {"t1": 1, "t2": 2, "t3": 3}


def test_assemble_trial_layout(tmp_path, monkeypatch):
    _patch_headers(monkeypatch, 3)
    farm = load_farm(_mini_farm(tmp_path, n_shards=3))
    first = assemble_trial(
        farm, group_tensors=["t2"], probe_type="Q4_K",
        baseline_type="Q8_0", dest_dir=tmp_path / "work", tag="g-Q4_K",
    )
    asm = tmp_path / "work" / f"{ASSEMBLY_PREFIX}g-Q4_K"
    assert first == asm / "q-00001-of-00003.gguf"
    links = sorted(p.name for p in asm.iterdir())
    assert links == [f"q-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    # Basenames preserved verbatim; group shard from the probe rung set,
    # everything else from the background set.
    assert (asm / "q-00002-of-00003.gguf").resolve().parent.name == "rung720-q4_k"
    assert (asm / "q-00001-of-00003.gguf").resolve().parent.name == "bg720-q8_0"
    assert (asm / "q-00003-of-00003.gguf").resolve().parent.name == "bg720-q8_0"


def test_assemble_rejects_unknown_tensor(tmp_path, monkeypatch):
    _patch_headers(monkeypatch, 3)
    farm = load_farm(_mini_farm(tmp_path, n_shards=3))
    with pytest.raises(RuntimeError, match="no.*shard"):
        assemble_trial(
            farm, group_tensors=["ghost.weight"], probe_type="Q4_K",
            baseline_type="Q8_0", dest_dir=tmp_path / "work", tag="x",
        )


def test_assemble_rejects_unknown_rung(tmp_path, monkeypatch):
    _patch_headers(monkeypatch, 3)
    farm = load_farm(_mini_farm(tmp_path, n_shards=3))
    with pytest.raises(RuntimeError, match="no rung set"):
        assemble_trial(
            farm, group_tensors=["t1"], probe_type="IQ4_XS",
            baseline_type="Q8_0", dest_dir=tmp_path / "work", tag="x",
        )


def test_measure_farm_column_shape(tmp_path, monkeypatch):
    """Same return shape as measure_column: parsed KL + exact byte math."""
    _patch_headers(monkeypatch, 3)
    monkeypatch.setattr(
        splice_farm, "_run", lambda cmd, what="": FAKE_KL_LOG)
    farm = load_farm(_mini_farm(tmp_path, n_shards=3))
    work = tmp_path / "work"
    m = measure_farm_column(
        farm=farm, probe_type="Q4_K", baseline_type="Q8_0",
        search_txt="s.txt", kl_base_bin="b.bin", work_dir=work,
        tag="g-Q4_K", group_tensors=["t2"],
    )
    assert m["kld_mean"] == pytest.approx(0.003369)
    assert m["kld_tail_1pct"] == pytest.approx(0.05)
    assert m["group_bytes_measured"] == 2000
    assert m["probe_exempt_tensors"] == []
    assert m["trial_tag"] == "g-Q4_K"
    assert (work / "farm-g-Q4_K.perplexity.log").is_file()
    # Symlink-only assembly dir removed after measuring.
    assert not (work / f"{ASSEMBLY_PREFIX}g-Q4_K").exists()


def test_measure_propagates_dtype_refusal(tmp_path, monkeypatch):
    """A group tensor the probe rung could not quantize raises (no zero row)."""
    _patch_headers(monkeypatch, 3, probe_dtype="F32")
    monkeypatch.setattr(
        splice_farm, "_run", lambda cmd, what="": FAKE_KL_LOG)
    farm = load_farm(_mini_farm(tmp_path, n_shards=3))
    with pytest.raises(RuntimeError, match="did not take probe type"):
        measure_farm_column(
            farm=farm, probe_type="Q4_K", baseline_type="Q8_0",
            search_txt="s.txt", kl_base_bin="b.bin",
            work_dir=tmp_path / "work", tag="g",
            group_tensors=["t2"],
        )


# --- live farm on disk (real headers; perplexity faked) ---------------------

@needs_farm
def test_live_map_matches_claimed_layout():
    farm = load_farm(FARM_ROOT)
    assert farm.n_shards == 720
    assert len(farm.tensor_to_idx) == 720
    assert farm.tensor_to_idx["token_embd.weight"] == 2
    assert farm.tensor_to_idx["rope_freqs.weight"] == 1


@needs_farm
def test_live_rung_coverage():
    """Every LADDER rung resolves to a full 720-shard set."""
    from optimizer import LADDER

    farm = load_farm(FARM_ROOT)
    assert set(LADDER) <= set(splice_farm.RUNG_SUBDIR)
    for rung in LADDER:
        d = farm.subdir_for(rung)
        assert len(list(d.glob("*.gguf"))) == 720, rung


@needs_farm
def test_live_assembly_and_byte_accounting(tmp_path, monkeypatch):
    """Real assembly + real header reads; only perplexity is faked."""
    monkeypatch.setattr(
        splice_farm, "_run", lambda cmd, what="": FAKE_KL_LOG)
    farm = load_farm(FARM_ROOT)
    group = ["token_embd.weight"]
    m = measure_farm_column(
        farm=farm, probe_type="Q4_K", baseline_type="Q8_0",
        search_txt="s.txt", kl_base_bin="b.bin",
        work_dir=tmp_path, tag="emb-Q4_K", group_tensors=group,
    )
    assert m["group_bytes_measured"] > 0
    assert m["probe_exempt_tensors"] == []
    assert m["kld_mean"] == pytest.approx(0.003369)


# --- step-12 wiring ----------------------------------------------------------

def _wiring_catalog():
    return {
        "groups": {
            "g1": {"role": "x", "depth": "early",
                   "tensor_names": ["t1"], "n_tensors": 1},
        },
        "tensors": {"t1": {"n_elements": 1_000_000}},
    }


def test_lazy_driver_uses_farm_when_set(tmp_path, monkeypatch):
    import llama_probe

    calls: list = []

    def fake_measure(*, farm, probe_type, tag, **kw):
        calls.append((probe_type, tag))
        return {
            "kld_mean": 0.01, "kld_tail_1pct": 0.02, "kld_p999": 0.03,
            "same_top_p": 0.99, "perplexity": 9.0,
            "group_bytes_measured": 100, "probe_exempt_tensors": [],
            "trial_tag": tag,
        }

    def fake_anchor(**kw):
        return {"kld_mean": 0.0, "kld_tail_1pct": 0.0,
                "trial_tag": "anchor"}

    monkeypatch.setattr(splice_farm, "load_farm", lambda root: "FARM")
    monkeypatch.setattr(splice_farm, "measure_farm_column", fake_measure)
    monkeypatch.setattr(llama_probe, "measure_source_anchor", fake_anchor)
    rows, absinfo = sensitivity.probe_groups_lazy(
        _wiring_catalog(),
        model_gguf="m.gguf", search_txt="s.txt", kl_base_bin="b.bin",
        probe_types=["Q2_K", "Q4_K"], baseline_type="Q8_0",
        work_dir=tmp_path, jobs=1,
        pricing_budget_bytes=10**9, splice_farm="/farm",
    )
    assert calls, "farm backend measured nothing"
    assert absinfo["splice_farm"] == "/farm"
    assert {r["probe"] for r in rows} <= {"Q2_K", "Q4_K"}


def test_lazy_driver_records_no_farm_by_default(tmp_path, monkeypatch):
    import llama_probe

    log: list = []

    def fake_measure(*, group_regex=None, probe_type, tag, **kw):
        log.append(tag)
        return {
            "kld_mean": 0.01, "kld_tail_1pct": 0.02, "kld_p999": 0.03,
            "same_top_p": 0.99, "perplexity": 9.0,
            "group_bytes_measured": 100, "probe_exempt_tensors": [],
            "trial_tag": tag,
        }

    monkeypatch.setattr(llama_probe, "measure_column", fake_measure)
    monkeypatch.setattr(
        llama_probe, "measure_source_anchor",
        lambda **kw: {"kld_mean": 0.0, "kld_tail_1pct": 0.0,
                      "trial_tag": "anchor"},
    )
    _rows, absinfo = sensitivity.probe_groups_lazy(
        _wiring_catalog(),
        model_gguf="m.gguf", search_txt="s.txt", kl_base_bin="b.bin",
        probe_types=["Q2_K"], baseline_type="Q8_0",
        work_dir=tmp_path, jobs=1, pricing_budget_bytes=10**9,
    )
    assert log, "pipeline backend measured nothing"
    assert absinfo["splice_farm"] is None


def _llama_inputs(tmp_path):
    model = tmp_path / "model.gguf"
    base = tmp_path / "kl.bin"
    search = tmp_path / "search.txt"
    for p in (model, base, search):
        p.write_bytes(b"")
    return model, base, search


def _lazy_abs():
    return {
        "kld_mean": 0.0, "kld_tail_1pct": 0.0,
        "pricing_rounds": 1, "pricing_lambda_history": [0.0],
        "pricing_excluded_columns": 0, "grid_skipped": 0,
        "fixed_skipped": [],
    }


def test_table_routes_farm_exhaustive_through_lazy(tmp_path, monkeypatch):
    """Farm + exhaustive: full universe via the priced driver (resume)."""
    seen: dict = {}

    def fake_lazy(**kw):
        seen.update(kw)
        return [], _lazy_abs()

    def boom(**kw):
        raise AssertionError("legacy no-resume path must not run with a farm")

    monkeypatch.setattr(sensitivity, "probe_groups_lazy", fake_lazy)
    monkeypatch.setattr(sensitivity, "probe_groups_llama", boom)
    model, base, search = _llama_inputs(tmp_path)
    result, rows = sensitivity.build_sensitivity_table(
        model_ref="m", out_dir=tmp_path / "out",
        catalog={"groups": {}, "tensors": {}},
        mode="llama", probe_types=["Q2_K"],
        model_gguf=model, kl_base_bin=base, search_path=search,
        trials_dir=tmp_path / "trials",
        certificate_mode="exhaustive", pricing_budget_bytes=10**18,
        splice_farm="/farm",
    )
    assert seen.get("splice_farm") == "/farm"
    assert seen.get("certificate_mode") == "exhaustive"
    assert rows == []


def test_table_keeps_legacy_exhaustive_without_farm(tmp_path, monkeypatch):
    """No farm + exhaustive: legacy direct path unchanged."""
    called: list = []

    def fake_legacy(**kw):
        called.append(True)
        return [], {
            "kld_mean": 0.0, "kld_tail_1pct": 0.0,
            "grid_skipped": 0, "fixed_skipped": [],
        }

    monkeypatch.setattr(sensitivity, "probe_groups_llama", fake_legacy)
    model, base, search = _llama_inputs(tmp_path)
    sensitivity.build_sensitivity_table(
        model_ref="m", out_dir=tmp_path / "out",
        catalog={"groups": {}, "tensors": {}},
        mode="llama", probe_types=["Q2_K"],
        model_gguf=model, kl_base_bin=base, search_path=search,
        trials_dir=tmp_path / "trials",
        certificate_mode="exhaustive",
    )
    assert called == [True]


# --- CLI flags -----------------------------------------------------------

def test_splice_farm_flags_parse_and_thread(monkeypatch, tmp_path):
    """--splice-farm exists on sensitivity/run/fit and reaches step 12."""
    import argparse

    import cli

    seen: dict[str, argparse.Namespace] = {}

    def fake(name):
        def _fn(args: argparse.Namespace) -> int:
            seen[name] = args
            return 0
        return _fn

    # Standalone sensitivity parser.
    monkeypatch.setattr(cli, "cmd_sensitivity", fake("cmd_sensitivity"))
    assert cli.main(["sensitivity", "--model", "m",
                     "--splice-farm", "/farm"]) == 0
    assert str(seen["cmd_sensitivity"].splice_farm) == "/farm"

    # Standalone fit parser.
    monkeypatch.setattr(cli, "cmd_fit", fake("cmd_fit"))
    assert cli.main(["fit", "--model", "m", "--device", "d",
                     "--splice-farm", "/farm"]) == 0
    assert str(seen["cmd_fit"].splice_farm) == "/farm"

    # odg run threads it into the shared step namespace.
    for step_fn in ("cmd_resolve", "cmd_load", "cmd_enumerate", "cmd_classify",
                    "cmd_catalog", "cmd_weight_features", "cmd_corpus",
                    "cmd_activation_features", "cmd_freeze_gguf", "cmd_imatrix",
                    "cmd_reband", "cmd_reference_logits", "cmd_sensitivity",
                    "cmd_optimize", "cmd_export", "cmd_validate"):
        monkeypatch.setattr(cli, step_fn, fake(step_fn))
    code = cli.main([
        "--artifacts", str(tmp_path),
        "run", "--model", "m", "--quant", "q4_k_m", "--no-ask",
        "--until", "validate", "--splice-farm", "/farm",
    ])
    assert code == 0
    assert str(seen["cmd_sensitivity"].splice_farm) == "/farm"


def test_splice_farm_in_stale_check():
    import argparse

    import cli

    args = argparse.Namespace(splice_farm="/farm")
    exp = cli._pipeline_expected_inputs("sensitivity", args, fmt=None)
    assert exp["splice_farm"] == "/farm"
    args = argparse.Namespace()
    exp = cli._pipeline_expected_inputs("sensitivity", args, fmt=None)
    assert exp["splice_farm"] is None
