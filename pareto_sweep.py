"""Pareto sweep: export + heldout Tier-1 every frontier recipe, plot, pick.

Reads a finished step-13 (optimize) directory: for each feasible Pareto
recipe (deduped by allocation hash), renders the tensor-type file from
the recipe's ``overrides:`` block, exports the candidate GGUF with the
same ``export_gguf`` path step 14 uses, and measures it with the same
``_tier1_llama`` held-out harness step 15 uses. Exports are transient:
each candidate is Tier-1'd, recorded, and deleted — only the winner
(lowest measured mean KLD at or under ``--size-cap``, default
5,127,000,000 bytes) is kept, with its recipe and provenance.

Outputs (``--out-dir``):
  frontier.json            per-point budgets/bytes/predicted/measured
  best-under-<cap>MB.gguf  the winner (+ .recipe.yaml/.tt/provenance.json)
  tier1-<hash>.log         perplexity log per measured point
  pareto-mean.png / pareto-p99.png / pareto-top1.png / pareto-ppl.png
                           measured metrics vs MiB, predicted-mean overlay,
                           optional XL reference marker

Usage:
  python3 pareto_sweep.py --optimize-dir <run>/steps/13_optimize \\
      --frozen <run>/steps/09_freeze_gguf/model-bf16.gguf \\
      --imatrix <run>/steps/10_imatrix/imatrix.gguf \\
      --heldout-txt <run>/steps/07_corpus/heldout.txt \\
      --heldout-bin <run>/steps/11_reference_logits/logits-heldout.bin \\
      --out-dir <run>/steps/16_pareto_sweep [--xl-gguf ...] [--dry-run]

  Step-15 frontier set (curated recipes, export deferred):
    --recipe-file <f.yaml> (repeat) --drop-winner
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_SIZE_CAP = 5_127_000_000  # bytes: at-or-below the XL reference size

_OVERRIDES_RE = re.compile(r'^\s*"((?:[^"\\]|\\.)+)":\s*([A-Za-z0-9_]+)\s*$')
_TARGET_RE = re.compile(r"^\s*target_size_bytes:\s*(\d+)\s*$")
_PRED_RE = re.compile(r"^\s*predicted_mean_delta_kld:\s*([-\d.eE+]+)\s*$")
_EST_RE = re.compile(r"^\s*size_bytes:\s*(\d+)\s*$")


def _parse_recipe_file(path: Path) -> dict[str, Any]:
    """Parse one recipe yaml: budget, predicted mean, overrides."""
    text = path.read_text(encoding="utf-8")
    budget = predicted = estimated = None
    overrides: dict[str, str] = {}
    in_overrides = False
    for line in text.splitlines():
        if line.startswith("overrides:"):
            in_overrides = True
            continue
        if in_overrides:
            m = _OVERRIDES_RE.match(line)
            if m:
                # Unescape the \" sequences the yaml renderer emits.
                regex = m.group(1).replace(r"\"", "\"").replace("\\\\", "\\")
                overrides[regex] = m.group(2)
                continue
            if line and not line.startswith(" "):
                in_overrides = False
        m = _TARGET_RE.match(line)
        if m:
            budget = int(m.group(1))
            continue
        m = _PRED_RE.match(line)
        if m:
            predicted = float(m.group(1))
            continue
        m = _EST_RE.match(line)
        if m:
            estimated = int(m.group(1))
    if budget is None or not overrides:
        raise RuntimeError(
            f"Could not parse recipe {path} "
            f"(budget={budget}, overrides={len(overrides)})."
        )
    return {
        "path": str(path),
        "budget_bytes": budget,
        "predicted_mean_kld": predicted,
        "estimated_bytes": estimated,
        "overrides": overrides,
    }


def parse_recipe_points(
    pareto_dir: Path,
    patterns: tuple[str, ...] = ("pareto-*.yaml",),
) -> list[dict[str, Any]]:
    """Parse every recipe matching patterns: budget, predicted, overrides.

    Line-based (render_recipe_yaml has a fixed layout); avoids a yaml
    dependency. Returns one dict per file: path, budget_bytes,
    predicted_mean_kld, estimated_bytes, overrides {regex: type}.
    """
    points = []
    for pat in patterns:
        for path in sorted(pareto_dir.glob(pat)):
            points.append(_parse_recipe_file(path))
    return points


def dedupe_points(
    points: list[dict[str, Any]],
    manifest_hashes: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Drop points whose allocation duplicates an earlier budget's.

    Identity key: the sorted overrides (exact recipe text). Manifest
    hashes preferred when the manifest maps budget → allocation_hash.
    Adjacent ratios often solve identically — dedupe before spending GPU.
    """
    seen: set[str] = set()
    kept = []
    for p in sorted(points, key=lambda p: p["budget_bytes"]):
        key = manifest_hashes.get(p["budget_bytes"]) if manifest_hashes else None
        if key is None:
            key = json.dumps(sorted(p["overrides"].items()),
                             separators=(",", ":"))
        if key in seen:
            p["deduped"] = True
            continue
        seen.add(key)
        p["deduped"] = False
        kept.append(p)
    return kept


def render_tt(overrides: dict[str, str], dest: Path) -> Path:
    """Write a llama-quantize --tensor-type-file from recipe overrides."""
    dest.write_text(
        "".join(f"{rx}={q.lower()}\n"
                for rx, q in sorted(overrides.items())),
        encoding="utf-8",
    )
    return dest


def measure_candidate(
    candidate: Path,
    *,
    heldout_txt: Path,
    heldout_bin: Path,
    out_dir: Path,
    tag: str,
    llama_perplexity=None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    from validate import _tier1_llama

    res = _tier1_llama(
        candidate=candidate, heldout_txt=heldout_txt,
        heldout_bin=heldout_bin, out_dir=out_dir,
        llama_perplexity=llama_perplexity, extra_args=extra_args,
    )
    return res


def export_candidate(
    *,
    recipe_yaml: Path,
    recipe_tt: Path,
    gguf_in: Path,
    imatrix: Path | None,
    out_path: Path,
    base_type: str,
    llama_quantize=None,
) -> tuple[str, int]:
    """Export one candidate GGUF via the step-14 path. Returns (path, bytes)."""
    from export import export_gguf

    res = export_gguf(
        model_ref=recipe_yaml.stem,
        out_dir=out_path.parent,
        gguf_in=gguf_in,
        recipe_path=recipe_yaml,
        recipe_tt=recipe_tt,
        imatrix_path=str(imatrix) if imatrix and imatrix.is_file() else None,
        mode="llama",
        llama_quantize=llama_quantize,
        base_type=base_type,
        out_name=out_path.name,
    )
    produced = getattr(res, "gguf_out", None) or str(out_path)
    nbytes = getattr(res, "gguf_out_nbytes", None) or Path(produced).stat().st_size
    return produced, int(nbytes)


def pick_winner(
    measured: list[dict[str, Any]], *, size_cap: int,
) -> dict[str, Any] | None:
    """Lowest measured mean KLD at or under the cap (hard limit)."""
    eligible = [m for m in measured
                if m.get("actual_bytes") is not None
                and int(m["actual_bytes"]) <= size_cap
                and m.get("tier1", {}).get("metrics", {}).get("mean_kld") is not None]
    if not eligible:
        return None
    return min(eligible,
               key=lambda m: float(m["tier1"]["metrics"]["mean_kld"]))


_PPL_BASE_RE = re.compile(r"Mean PPL\(base\)\s*:\s*([-\d.eE+]+)")


def parse_ppl_base(text: str) -> float | None:
    """Best-effort ``Mean PPL(base)`` from a llama-perplexity Tier-1 log."""
    m = _PPL_BASE_RE.search(text)
    return float(m.group(1)) if m else None


def _bpw_label(actual_bytes: int, n_params: int | None) -> str | None:
    """Real bits-per-weight: actual file bytes over total params."""
    if not n_params:
        return None
    return f"{int(actual_bytes) * 8 / int(n_params):.2f}"


def _place_labels(ax, texts, dot_xs, dot_ys, line_xs, line_ys):
    """Force-directed label placement (adjustText — the standard tool).

    Labels repel each other, the dots, and virtual points sampled along
    the frontier polyline; a weak pull keeps them near their dots and
    everything stays inside the axes. adjustText draws thin leaders for
    displaced labels itself.
    """
    from adjustText import adjust_text

    adjust_text(
        texts,
        x=list(dot_xs) + list(line_xs),
        y=list(dot_ys) + list(line_ys),
        arrowprops=dict(arrowstyle="-", color="#777777", linewidth=0.7),
        ax=ax,
    )


def frontier_figure(points: list[tuple[float, float, str | None]],
                    *, title: str | None, ylabel: str, log_y: bool = False,
                    ref_value: float | None = None,
                    ref_label: str | None = None,
                    xl: tuple[float, float, str | None] | None = None,
                    figsize: tuple[float, float] = (6.4, 4.8)):
    """One frontier chart (shared by the PNG and report-SVG backends).

    ``points``: ``(x, y, label)`` measured dots in x order (``label``
    may be None — the dot still plots); ``xl`` is an optional
    ``(x, y, label)`` star. Returns the placed Figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, "o-", label="measured frontier")
    if xl is not None:
        ax.scatter([xl[0]], [xl[1]], marker="*", s=200, label="Unsloth XL")
    if ref_value is not None:
        ax.axhline(float(ref_value), color="grey", linestyle="--",
                   linewidth=1.2, label=ref_label)
    use_log = bool(log_y) and all(float(y) > 0 for y in ys)
    if use_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Size (MiB)")
    ax.set_ylabel(f"{ylabel} (log scale)" if use_log else ylabel)
    if title is not None:
        ax.set_title(f"Pareto frontier — {title} vs size (held-out Tier-1)")
    ax.legend()
    fig.tight_layout()
    # Labels last: adjustText needs final display coords (tight_layout
    # moves the axes; anything earlier would be stale).
    fig.canvas.draw()
    all_pts = list(points) + ([xl] if xl is not None else [])
    texts = [ax.text(x, y, lab, fontsize=6, fontweight="bold",
                     color="#555555") for x, y, lab in all_pts
             if lab is not None]
    if texts:
        # Virtual points along the polyline repel labels from the line
        # itself (the adjustText-docs recipe); xs is monotonic in x.
        lx = np.linspace(min(xs), max(xs), 200).tolist()
        line = [(x, float(np.interp(x, xs, ys))) for x in lx]
        _place_labels(ax, texts,
                      [p[0] for p in all_pts], [p[1] for p in all_pts],
                      [p[0] for p in line], [p[1] for p in line])
    return fig


def frontier_svg(points: list[tuple[float, float, str]], **kwargs) -> str:
    """Inline-SVG twin of one :func:`frontier_figure` chart for report.html."""
    import io

    import matplotlib.pyplot as plt

    fig = frontier_figure(points, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    svg = buf.getvalue().decode("utf-8")
    svg = svg[svg.index("<svg"):]
    # Responsive width; keep the viewBox for aspect.
    import re as _re

    svg = _re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', 'width="100%"',
                  svg, count=1)
    return svg


def _chart_points(pts, key: str, tier1_level: bool, fmt: str,
                  n_params: int | None
                  ) -> list[tuple[float, float, str | None]]:
    """``(x-miB, y, "v, bpw bpw")`` tuples for one chart metric.

    The label is None when ``n_params`` is unknown — the dot still
    plots, it just goes unlabeled.
    """
    out = []
    for m in pts:
        y = (m["tier1"].get(key) if tier1_level
             else m["tier1"]["metrics"].get(key))
        if y is None:
            continue
        x = int(m["actual_bytes"]) / (1024 * 1024)
        lab = _bpw_label(int(m["actual_bytes"]), n_params)
        out.append((x, float(y),
                    f"{float(y):{fmt}}, {lab} bpw" if lab is not None else None))
    return out


def _chart_xl(xl: dict[str, Any] | None, key: str, tier1_level: bool,
              fmt: str, n_params: int | None
              ) -> tuple[float, float, str | None] | None:
    if not xl or xl.get("actual_bytes") is None:
        return None
    xval = ((xl.get("tier1") or {}).get(key) if tier1_level
            else ((xl.get("tier1") or {}).get("metrics") or {}).get(key))
    if xval is None:
        return None
    lab = _bpw_label(int(xl["actual_bytes"]), n_params)
    return (int(xl["actual_bytes"]) / (1024 * 1024), float(xval),
            f"XL {float(xval):{fmt}}, {lab} bpw" if lab is not None else None)


def plot_frontier(
    measured: list[dict[str, Any]],
    out_dir: Path,
    *,
    xl: dict[str, Any] | None = None,
    noise_floor_kld: float | None = None,
    noise_floor_p99: float | None = None,
    ppl_base: float | None = None,
    n_params: int | None = None,
) -> list[str]:
    """Four PNGs: mean / p99 / top1 / ppl vs MiB (measured Tier-1).

    KLD plots use a log y-scale; the KLD pair carries the background
    noise floor and the perplexity plot the BF16 base as a grey dashed
    reference. Every dot is labelled with its real bits-per-weight
    (``actual_bytes * 8 / n_params``) when ``n_params`` is known, placed
    by adjustText. No DP-predicted overlay, no winner marker — the
    frontier speaks.
    """
    import matplotlib.pyplot as plt

    pts = sorted(
        (m for m in measured
         if m.get("actual_bytes") and m.get("tier1", {}).get("metrics")),
        key=lambda m: int(m["actual_bytes"]),
    )
    if not pts:
        raise RuntimeError("Nothing measured — no plots to draw.")
    # (fname, title, key, tier1-level?, log_y, ref_value, ref_label, fmt)
    # Perplexity and top-1 need 3 sig figs (their action lives in the
    # 3rd digit); the KLD pair reads cleanly at 2.
    specs = [
        ("pareto-mean.png", "Mean KLD", "mean_kld", False, True,
         noise_floor_kld, "noise floor (all-Q8 background)", ".2g"),
        ("pareto-p99.png", "99% KLD", "p99_kld", False, True,
         noise_floor_p99
         if noise_floor_p99 is not None else noise_floor_kld,
         "noise floor (all-Q8 background)", ".2g"),
        ("pareto-top1.png", "Top-1 agreement", "top1_agree", False, False,
         None, None, ".3g"),
        ("pareto-ppl.png", "Perplexity", "perplexity", True, False,
         ppl_base, "BF16 base", ".3g"),
    ]
    written = []
    for fname, title, key, tier1_level, log_y, ref_val, ref_label, fmt in specs:
        items = _chart_points(pts, key, tier1_level, fmt, n_params)
        if not items:
            continue
        fig = frontier_figure(
            items, title=title, ylabel=title, log_y=log_y,
            ref_value=ref_val, ref_label=ref_label,
            xl=_chart_xl(xl, key, tier1_level, fmt, n_params))
        dest = out_dir / fname
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        written.append(str(dest))
    return written


def run_sweep(
    *,
    optimize_dir: Path,
    gguf_in: Path,
    imatrix: Path | None,
    heldout_txt: Path,
    heldout_bin: Path,
    out_dir: Path,
    base_type: str = "q4_k_m",
    size_cap: int = DEFAULT_SIZE_CAP,
    xl_gguf: Path | None = None,
    llama_quantize=None,
    llama_perplexity=None,
    perplexity_args: list[str] | None = None,
    dry_run: bool = False,
    export_fn=export_candidate,
    measure_fn=measure_candidate,
    recipe_files: list[Path] | None = None,
    keep_winner: bool = True,
    noise_floor_kld: float | None = None,
    noise_floor_p99: float | None = None,
    ppl_base: float | None = None,
    n_params: int | None = None,
) -> dict[str, Any]:
    """Full sweep. Returns the frontier record (also written to disk).

    ``recipe_files`` measures exactly those recipes (the step-15
    frontier set: curated BPW selections + the primary) instead of
    globbing the pareto dir; exact-duplicate allocations fold by
    overrides identity. ``keep_winner=False`` deletes every export
    after measuring (export deferred — plots + frontier.json stay).
    """
    optimize_dir = Path(optimize_dir)
    out_dir = Path(out_dir)
    (out_dir / "exports").mkdir(parents=True, exist_ok=True)

    manifest_hashes: dict[int, str] = {}
    manifest_path = optimize_dir / "optimize_manifest.json"
    if manifest_path.is_file():
        for entry in json.loads(manifest_path.read_text()).get("pareto", []):
            if entry.get("feasible") and entry.get("allocation_hash"):
                manifest_hashes[int(entry["budget_bytes"])] = str(
                    entry["allocation_hash"])

    if recipe_files is not None:
        points = [_parse_recipe_file(Path(p)) for p in recipe_files]
        kept = dedupe_points(points, None)
    else:
        points = parse_recipe_points(optimize_dir / "pareto")
        if manifest_hashes:
            feasible_budgets = set(manifest_hashes)
            points = [p for p in points if p["budget_bytes"] in feasible_budgets]
        kept = dedupe_points(points, manifest_hashes or None)

    measured: list[dict[str, Any]] = []
    for i, p in enumerate(kept):
        tag = f"{i:02d}-{p['budget_bytes'] // 1024}k"
        exp_dir = out_dir / "exports" / tag
        exp_dir.mkdir(parents=True, exist_ok=True)
        # Stage the .tt under a non-colliding name: export_gguf copies
        # recipe_tt → out_dir/recipe.tt for provenance (SameFileError if
        # we render directly to that name).
        tt = render_tt(p["overrides"], exp_dir / "candidate.tt")
        recipe_yaml = Path(p["path"])
        out_gguf = exp_dir / f"pareto-{tag}.gguf"
        rec: dict[str, Any] = {
            "hash": manifest_hashes.get(p["budget_bytes"]) or json.dumps(
                sorted(p["overrides"].items()), separators=(",", ":")),
            "budget_bytes": p["budget_bytes"],
            "predicted_mean_kld": p.get("predicted_mean_kld"),
            "estimated_bytes": p.get("estimated_bytes"),
            "recipe_yaml": str(recipe_yaml),
        }
        if dry_run:
            rec["tier1"] = None
            rec["actual_bytes"] = None
            measured.append(rec)
            continue
        produced, nbytes = export_fn(
            recipe_yaml=recipe_yaml, recipe_tt=tt, gguf_in=gguf_in,
            imatrix=imatrix, out_path=out_gguf, base_type=base_type,
            llama_quantize=llama_quantize,
        )
        rec["actual_bytes"] = nbytes
        rec["staged_tt"] = str(tt)
        tier1 = measure_fn(
            Path(produced), heldout_txt=heldout_txt,
            heldout_bin=heldout_bin, out_dir=exp_dir, tag=tag,
            llama_perplexity=llama_perplexity, extra_args=perplexity_args,
        )
        rec["tier1"] = tier1
        (exp_dir / "tier1.json").write_text(
            json.dumps(tier1, indent=2) + "\n", encoding="utf-8")
        measured.append(rec)

    xl_rec: dict[str, Any] | None = None
    if xl_gguf is not None and not dry_run:
        xl_dir = out_dir / "exports" / "xl"
        xl_dir.mkdir(parents=True, exist_ok=True)
        xl_tier1 = measure_fn(
            Path(xl_gguf), heldout_txt=heldout_txt,
            heldout_bin=heldout_bin, out_dir=xl_dir, tag="xl",
            llama_perplexity=llama_perplexity, extra_args=perplexity_args,
        )
        (xl_dir / "tier1.json").write_text(
            json.dumps(xl_tier1, indent=2) + "\n", encoding="utf-8")
        xl_rec = {
            "path": str(xl_gguf),
            "actual_bytes": Path(xl_gguf).stat().st_size,
            "tier1": xl_tier1,
        }

    winner = None if dry_run else pick_winner(measured, size_cap=size_cap)
    if winner is None and not dry_run:
        raise RuntimeError(
            f"No measured point at or under the {size_cap}-byte cap — "
            "nothing to crown. Frontier:\n" + "\n".join(
                f"  {m['budget_bytes']}B est / "
                f"{m.get('actual_bytes')}B actual" for m in measured)
        )
    if not dry_run and not keep_winner:
        # Export deferred (step-15 frontier mode): every candidate GGUF
        # is deleted after measuring — plots + frontier.json stay.
        assert winner is not None
        for m in measured:
            tag_dir = out_dir / "exports" / (
                f"{measured.index(m):02d}-{m['budget_bytes'] // 1024}k")
            for gguf in tag_dir.glob("*.gguf"):
                gguf.unlink(missing_ok=True)
        winner["kept_gguf"] = None
    # Delete losers: only the winner's GGUF stays on disk. The winner's
    # working export is removed too after copying to its final name —
    # one GGUF total, plus small recipe/provenance/logs.
    if not dry_run and keep_winner:
        assert winner is not None
        wi = measured.index(winner)
        wtag = f"{wi:02d}-{winner['budget_bytes'] // 1024}k"
        wdir = out_dir / "exports" / wtag
        ggufs = list(wdir.glob("*.gguf"))
        cap_mb = size_cap // 1_000_000
        final = out_dir / f"best-under-{cap_mb}MB.gguf"
        shutil.copy2(str(ggufs[0]), str(final))
        for gguf in ggufs:
            gguf.unlink()
        for m in measured:
            if m is winner:
                continue
            tag_dir = out_dir / "exports" / (
                f"{measured.index(m):02d}-{m['budget_bytes'] // 1024}k")
            for gguf in tag_dir.glob("*.gguf"):
                gguf.unlink(missing_ok=True)
        shutil.copy2(str(winner["recipe_yaml"]),
                     str(out_dir / f"best-under-{cap_mb}MB.recipe.yaml"))
        shutil.copy2(str(winner["staged_tt"]),
                     str(out_dir / f"best-under-{cap_mb}MB.recipe.tt"))
        (out_dir / f"best-under-{cap_mb}MB.provenance.json").write_text(
            json.dumps(winner, indent=2) + "\n", encoding="utf-8")
        winner["kept_gguf"] = str(final)

    if ppl_base is None and not dry_run:
        # Best-effort BF16 reference: the first Tier-1 log that prints it.
        # Stub measure functions (tests) write no logs — then there is
        # simply no PPL baseline line.
        for log in sorted(out_dir.glob("exports/*/llama-perplexity-tier1.log")):
            try:
                ppl_base = parse_ppl_base(log.read_text(encoding="utf-8"))
            except OSError:
                continue
            if ppl_base is not None:
                break
    plots = [] if dry_run else plot_frontier(
        measured, out_dir, xl=xl_rec,
        noise_floor_kld=noise_floor_kld,
        noise_floor_p99=noise_floor_p99,
        ppl_base=ppl_base,
        n_params=n_params,
    )
    record = {
        "optimize_dir": str(optimize_dir),
        "size_cap_bytes": size_cap,
        "n_recipes": len(points),
        "n_measured": len(measured),
        "points": measured,
        "xl": xl_rec,
        "winner": winner,
        "plots": plots,
        "noise_floor_kld": noise_floor_kld,
        "noise_floor_p99": noise_floor_p99,
        "ppl_base": ppl_base,
        "n_params": n_params,
    }
    (out_dir / "frontier.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pareto sweep (see module docstring).")
    ap.add_argument("--optimize-dir", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--imatrix", type=Path, default=None)
    ap.add_argument("--heldout-txt", type=Path, required=True)
    ap.add_argument("--heldout-bin", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-type", default="q4_k_m")
    ap.add_argument("--size-cap", type=int, default=DEFAULT_SIZE_CAP)
    ap.add_argument("--xl-gguf", type=Path, default=None)
    ap.add_argument("--llama-quantize", default=None)
    ap.add_argument("--llama-perplexity", default=None)
    ap.add_argument("--perplexity-args", default=None, metavar="ARGS",
                    help='Extra args for Tier-1 perplexity, e.g. "-ngl 99"')
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse + dedupe + pick-shape only (no GPU).")
    ap.add_argument("--recipe-file", type=Path, action="append", default=None,
                    help="Measure exactly these recipes (repeatable); "
                    "skips the pareto-dir glob. The step-15 frontier set.")
    ap.add_argument("--noise-floor-kld", type=float, default=None,
                    help="Background mean KLD for the grey dashed noise-floor "
                    "line on the KLD plots (default: no line).")

    ap.add_argument("--ppl-base", type=float, default=None,
                    help="BF16 base perplexity for the grey dashed line on "
                    "the PPL plot (default: parsed from Tier-1 logs).")
    ap.add_argument("--n-params", type=int, default=None,
                    help="Total model params for per-dot real-bpw labels "
                    "(default: no bpw labels).")
    ap.add_argument("--drop-winner", action="store_true",
                    help="Delete every export after measuring (export "
                    "deferred — keep plots + frontier.json only).")
    args = ap.parse_args(argv)

    from cli import split_extra_args

    record = run_sweep(
        optimize_dir=args.optimize_dir,
        gguf_in=args.frozen,
        imatrix=args.imatrix,
        heldout_txt=args.heldout_txt,
        heldout_bin=args.heldout_bin,
        out_dir=args.out_dir,
        base_type=args.base_type,
        size_cap=args.size_cap,
        xl_gguf=args.xl_gguf,
        llama_quantize=args.llama_quantize,
        llama_perplexity=args.llama_perplexity,
        perplexity_args=split_extra_args(args.perplexity_args,
                                         "--perplexity-args"),
        dry_run=args.dry_run,
        recipe_files=args.recipe_file,
        keep_winner=not args.drop_winner,
        noise_floor_kld=args.noise_floor_kld,
        ppl_base=args.ppl_base,
        n_params=args.n_params,
    )
    w = record["winner"] or {}
    print(json.dumps({
        "n_measured": record["n_measured"],
        "winner_bytes": (w.get("actual_bytes")),
        "winner_mean": ((w.get("tier1") or {}).get("metrics") or {}).get("mean_kld"),
        "winner_gguf": w.get("kept_gguf"),
        "plots": record["plots"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
