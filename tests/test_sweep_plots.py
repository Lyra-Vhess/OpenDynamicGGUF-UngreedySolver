"""Sweep plot styling: log scales, reference lines, grid, bpw labels.

No winner marker, no DP-predicted overlay. Axes are captured by wrapping
plt.subplots so assertions run on the object model, not pixels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import pareto_sweep
from pareto_sweep import _bpw_label, parse_ppl_base, plot_frontier


def _measured():
    return [
        {"actual_bytes": 3_000_000_000,
         "predicted_mean_kld": 0.5,
         "tier1": {"metrics": {"mean_kld": 0.08, "p99_kld": 1.2,
                               "top1_agree": 0.94},
                   "perplexity": 2.69}},
        {"actual_bytes": 4_000_000_000,
         "predicted_mean_kld": 0.4,
         "tier1": {"metrics": {"mean_kld": 0.02, "p99_kld": 0.3,
                               "top1_agree": 0.97},
                   "perplexity": 2.71}},
    ]


def _xl():
    return {"actual_bytes": 5_000_000_000,
            "tier1": {"metrics": {"mean_kld": 0.017, "p99_kld": 0.25,
                                  "top1_agree": 0.97},
                      "perplexity": 2.75}}


@pytest.fixture
def capture_axes(monkeypatch):
    """Wrap plt.subplots: record every Axes created inside plot_frontier."""
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots
    axes = []

    def _wrapped(*a, **k):
        fig, ax = real_subplots(*a, **k)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _wrapped)
    return axes


def _legends(ax):
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def test_bpw_label_math():
    assert _bpw_label(3_362_197_824, 7_518_069_290) == "3.58"
    assert _bpw_label(100, None) is None
    assert _bpw_label(100, 0) is None


def test_parse_ppl_base():
    assert parse_ppl_base("Mean PPL(base)                :   2.722208 ± x") == 2.722208
    assert parse_ppl_base("no such line") is None


def test_plot_styling(tmp_path, capture_axes):
    plots = plot_frontier(
        _measured(), tmp_path, xl=_xl(),
        noise_floor_kld=0.001233, noise_floor_p99=0.0189,
        ppl_base=2.722208, n_params=7_518_069_290)
    assert [Path(p).name for p in plots] == [
        "pareto-mean.png", "pareto-p99.png",
        "pareto-top1.png", "pareto-ppl.png"]
    assert all(Path(p).is_file() for p in plots)
    assert len(capture_axes) == 4

    for ax in capture_axes:
        labs = _legends(ax)
        assert "pick" not in labs and "DP predicted (additive)" not in labs
        # faint grid at every interval (gridlines exist and are drawn)
        assert any(l.get_visible() for l in ax.get_xgridlines())
        assert any(l.get_visible() for l in ax.get_ygridlines())

    mean_ax, p99_ax, top1_ax, ppl_ax = capture_axes
    assert mean_ax.get_yscale() == "log"
    assert p99_ax.get_yscale() == "log"
    assert top1_ax.get_yscale() == "linear"
    assert ppl_ax.get_yscale() == "linear"
    # reference lines with labels, none on top-1
    assert "noise floor (all-Q8 background)" in _legends(mean_ax)
    assert "noise floor (all-Q8 background)" in _legends(p99_ax)
    # per-metric floors: the P99 hline sits at its own background value
    def _hlines(ax):
        return [l for l in ax.lines if l.get_linestyle() == "--"]
    mean_y = [l.get_ydata()[0] for l in _hlines(mean_ax)]
    p99_y = [l.get_ydata()[0] for l in _hlines(p99_ax)]
    assert mean_y and abs(mean_y[0] - 0.001233) < 1e-9
    assert p99_y and abs(p99_y[0] - 0.0189) < 1e-9
    assert "BF16 base" in _legends(ppl_ax)
    assert len(top1_ax.lines) == 1  # frontier only, no hline
    # per-dot bpw annotations: 2 frontier + XL star label each
    assert len(mean_ax.texts) >= 3
    assert any("bpw" in t.get_text() for t in mean_ax.texts)
    assert any(t.get_text().startswith("XL ") and "bpw" in t.get_text()
               for t in mean_ax.texts)
    assert mean_ax.texts[0].get_fontsize() < 7
    assert mean_ax.texts[0].get_fontweight() == "bold"
    assert any("Unsloth XL" in _legends(ax) for ax in capture_axes)
    assert any("measured frontier" in _legends(ax) for ax in capture_axes)


def _crowded(n=5):
    base = 4_500_000_000
    return [
        {"actual_bytes": 3_000_000_000,
         "tier1": {"metrics": {"mean_kld": 0.08, "p99_kld": 1.2,
                               "top1_agree": 0.94},
                   "perplexity": 2.69}},
        *({"actual_bytes": base + i * 1_000_000,
           "tier1": {"metrics": {"mean_kld": 0.01, "p99_kld": 0.2,
                                 "top1_agree": 0.985},
                     "perplexity": 2.73}}
          for i in range(n)),
        {"actual_bytes": 6_000_000_000,
         "tier1": {"metrics": {"mean_kld": 0.005, "p99_kld": 0.06,
                               "top1_agree": 0.986},
                   "perplexity": 2.75}},
    ]


def test_ppl_three_sig_figs(tmp_path, capture_axes):
    plot_frontier(_measured(), tmp_path, n_params=7_518_069_290)
    ppl_ax = capture_axes[3]
    texts = [t.get_text() for t in ppl_ax.texts]
    assert any("2.69," in t for t in texts)
    assert any("2.71," in t for t in texts)
    mean_ax = capture_axes[0]
    assert any(t.startswith("0.08,") for t in
               [t.get_text() for t in mean_ax.texts])


def _assert_placed(ax):
    """Real-bbox audit of adjustText output: labels pairwise disjoint
    and fully inside the axes."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.texts]
    assert boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j])
    frame = ax.get_window_extent(renderer)
    for b in boxes:
        # 1px tolerance: containment holds up to float dust.
        assert b.x0 >= frame.x0 - 1 and b.y0 >= frame.y0 - 1
        assert b.x1 <= frame.x1 + 1 and b.y1 <= frame.y1 + 1


def test_crowded_labels_placed(tmp_path, capture_axes, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    plot_frontier(_crowded(), tmp_path, n_params=7_518_069_290)
    top1_ax = capture_axes[2]
    assert len(top1_ax.texts) == 7  # every dot still labelled
    _assert_placed(top1_ax)
    assert sum(isinstance(p, FancyArrowPatch)
               for p in top1_ax.patches) >= 1  # displaced get leaders


def test_logscale_labels_placed(tmp_path, capture_axes, monkeypatch):
    """End-to-end: real text bboxes on the log-scale mean-KLD chart."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    klds = [0.0024, 0.0026, 0.0029, 0.0035, 0.0048, 0.0171]
    sizes = [3000.0, 3400.0, 3800.0, 4200.0, 4800.0, 5126.0]
    points = [
        {"actual_bytes": int(s * 1024 * 1024),
         "tier1": {"metrics": {"mean_kld": k, "tail_p99_kld": k * 8,
                               "top1_agree": 0.9, "perplexity": 3.0}}}
        for s, k in zip(sizes, klds)
    ]
    plot_frontier(points, tmp_path,
                  noise_floor_kld=0.0012, n_params=7518069290)
    ax = capture_axes[0]  # mean-KLD chart
    texts = [t for t in ax.texts if "bpw" in t.get_text()]
    assert len(texts) == len(klds)
    _assert_placed(ax)


def test_plot_without_baselines(tmp_path, capture_axes):
    plots = plot_frontier(_measured(), tmp_path)
    assert len(plots) == 4
    mean_ax = capture_axes[0]
    assert mean_ax.get_yscale() == "log"
    assert len(mean_ax.lines) == 1  # no reference line, no overlay
    assert len(mean_ax.texts) == 0  # no bpw without n_params


def test_run_sweep_records_plot_inputs(tmp_path):
    opt = tmp_path / "opt"
    (opt / "pareto").mkdir(parents=True)

    def fake_export(*, recipe_yaml, recipe_tt, gguf_in, imatrix,
                    out_path, base_type, llama_quantize):
        out_path = Path(out_path)
        out_path.write_bytes(b"G")
        return str(out_path), 3_000_000_000

    def fake_measure(candidate, *, heldout_txt, heldout_bin, out_dir,
                     tag, llama_perplexity, extra_args):
        return {"method": "fake",
                "metrics": {"mean_kld": 0.02, "p99_kld": 0.2,
                            "top1_agree": 0.97},
                "perplexity": 2.7}

    import json

    (opt / "pareto" / "pareto-00-1k.yaml").write_text(
        "schema: odg/recipe/v1\n  target_size_bytes: 100\n"
        "overrides:\n  \"a\": Q4_K\nestimate:\n  size_bytes: 90\n"
        "  predicted_mean_delta_kld: 0.020000\nvalidation: {}\n")
    (opt / "optimize_manifest.json").write_text(json.dumps({"pareto": []}))
    rec = pareto_sweep.run_sweep(
        optimize_dir=opt, gguf_in=tmp_path / "f.gguf", imatrix=None,
        heldout_txt=tmp_path / "h.txt", heldout_bin=tmp_path / "h.bin",
        out_dir=tmp_path / "sweep", size_cap=10**12,
        export_fn=fake_export, measure_fn=fake_measure,
        noise_floor_kld=0.001, n_params=1000)
    assert rec["noise_floor_kld"] == 0.001
    assert rec["n_params"] == 1000
    assert rec["ppl_base"] is None  # stub writes no Tier-1 logs
    assert len(rec["plots"]) == 4


def test_run_sweep_parses_ppl_base_from_logs(tmp_path):
    opt = tmp_path / "opt"
    (opt / "pareto").mkdir(parents=True)

    def fake_export(*, recipe_yaml, recipe_tt, gguf_in, imatrix,
                    out_path, base_type, llama_quantize):
        out_path = Path(out_path)
        tagdir = out_path.parent
        tagdir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"G")
        (tagdir / "llama-perplexity-tier1.log").write_text(
            "Mean PPL(Q)                   :   2.700000\n"
            "Mean PPL(base)                :   2.722208 ±   0.090947\n")
        return str(out_path), 3_000_000_000

    def fake_measure(candidate, *, heldout_txt, heldout_bin, out_dir,
                     tag, llama_perplexity, extra_args):
        # Mirror _tier1_llama: the log lives in the measurement out_dir.
        (Path(out_dir) / "llama-perplexity-tier1.log").write_text(
            "Mean PPL(Q)                   :   2.700000\n"
            "Mean PPL(base)                :   2.722208 ±   0.090947\n")
        return {"method": "fake",
                "metrics": {"mean_kld": 0.02, "p99_kld": 0.2,
                            "top1_agree": 0.97},
                "perplexity": 2.7}

    import json

    (opt / "pareto" / "pareto-00-1k.yaml").write_text(
        "schema: odg/recipe/v1\n  target_size_bytes: 100\n"
        "overrides:\n  \"a\": Q4_K\nestimate:\n  size_bytes: 90\n"
        "  predicted_mean_delta_kld: 0.020000\nvalidation: {}\n")
    (opt / "optimize_manifest.json").write_text(json.dumps({"pareto": []}))
    rec = pareto_sweep.run_sweep(
        optimize_dir=opt, gguf_in=tmp_path / "f.gguf", imatrix=None,
        heldout_txt=tmp_path / "h.txt", heldout_bin=tmp_path / "h.bin",
        out_dir=tmp_path / "sweep", size_cap=10**12,
        export_fn=fake_export, measure_fn=fake_measure)
    assert rec["ppl_base"] == pytest.approx(2.722208)
