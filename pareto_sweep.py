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


def parse_recipe_points(pareto_dir: Path) -> list[dict[str, Any]]:
    """Parse every pareto-*.yaml: budget, predicted mean, overrides.

    Line-based (render_recipe_yaml has a fixed layout); avoids a yaml
    dependency. Returns one dict per file: path, budget_bytes,
    predicted_mean_kld, estimated_bytes, overrides {regex: type}.
    """
    points = []
    for path in sorted(pareto_dir.glob("pareto-*.yaml")):
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
                f"Could not parse pareto recipe {path} "
                f"(budget={budget}, overrides={len(overrides)})."
            )
        points.append({
            "path": str(path),
            "budget_bytes": budget,
            "predicted_mean_kld": predicted,
            "estimated_bytes": estimated,
            "overrides": overrides,
        })
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


def plot_frontier(
    measured: list[dict[str, Any]],
    out_dir: Path,
    *,
    xl: dict[str, Any] | None = None,
    winner_hash: str | None = None,
) -> list[str]:
    """Four PNGs: mean / p99 / top1 / ppl vs MiB + predicted overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted(
        (m for m in measured
         if m.get("actual_bytes") and m.get("tier1", {}).get("metrics")),
        key=lambda m: int(m["actual_bytes"]),
    )
    if not pts:
        raise RuntimeError("Nothing measured — no plots to draw.")
    mib = [int(m["actual_bytes"]) / (1024 * 1024) for m in pts]
    specs = [
        ("pareto-mean.png", "Mean KLD", "mean_kld", None),
        ("pareto-p99.png", "99% KLD", "p99_kld", None),
        ("pareto-top1.png", "Top-1 agreement", "top1_agree", None),
        ("pareto-ppl.png", "Perplexity", "perplexity", "tier1"),
    ]
    written = []
    for fname, title, key, where in specs:
        fig, ax = plt.subplots()
        if where == "tier1":
            ys = [m["tier1"].get(key) for m in pts]
        else:
            ys = [m["tier1"]["metrics"].get(key) for m in pts]
        ok = [(x, y) for x, y in zip(mib, ys) if y is not None]
        if not ok:
            plt.close(fig)
            continue
        xs, yv = zip(*ok)
        ax.plot(xs, yv, "o-", label="measured frontier")
        # Predicted-mean overlay on the mean plot (additivity audit).
        if key == "mean_kld":
            pred = [(int(m["actual_bytes"]) / (1024 * 1024),
                     m.get("predicted_mean_kld")) for m in pts
                    if m.get("predicted_mean_kld") is not None]
            if pred:
                px, py = zip(*pred)
                ax.plot(px, py, "x--", label="DP predicted (additive)")
        if xl and xl.get("actual_bytes") is not None:
            xval = None
            if where == "tier1":
                xval = (xl.get("tier1") or {}).get(key)
            else:
                xval = ((xl.get("tier1") or {}).get("metrics") or {}).get(key)
            if xval is not None:
                ax.scatter([int(xl["actual_bytes"]) / (1024 * 1024)], [xval],
                           marker="*", s=200, label="Unsloth XL")
        if winner_hash:
            for m in pts:
                if m.get("hash") == winner_hash:
                    y0 = (m["tier1"].get(key) if where == "tier1"
                          else m["tier1"]["metrics"].get(key))
                    if y0 is not None:
                        ax.scatter([int(m["actual_bytes"]) / (1024 * 1024)],
                                   [y0], marker="D", s=120, label="pick")
        ax.set_xlabel("Size (MiB)")
        ax.set_ylabel(title)
        ax.set_title(f"Pareto frontier — {title} vs size (held-out Tier-1)")
        ax.legend()
        fig.tight_layout()
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
) -> dict[str, Any]:
    """Full sweep. Returns the frontier record (also written to disk)."""
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
            "hash": manifest_hashes.get(p["budget_bytes"]),
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
    # Delete losers: only the winner's GGUF stays on disk. The winner's
    # working export is removed too after copying to its final name —
    # one GGUF total, plus small recipe/provenance/logs.
    if not dry_run:
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

    plots = [] if dry_run else plot_frontier(
        measured, out_dir, xl=xl_rec,
        winner_hash=(winner or {}).get("hash"),
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
