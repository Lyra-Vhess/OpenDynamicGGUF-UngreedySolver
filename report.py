"""HTML report generator — feature 03 of the platform (docs/platform/03-…).

Read-only over the run store: renders artifacts that steps already wrote into
one self-contained ``report.html`` (inline CSS + SVG, no external assets, works
offline from ``file://``). Missing artifacts render as "not run" — never an
error. Every section captions the artifact it came from.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sensitivity import estimate_group_nbytes

# Source of truth for role floors lives in optimizer._candidate_ladder
# (measured pin_high hints only); report reasons never invent pins.
_QUANT_COLORS = {
    "F32": "#e8e8e8",
    "F16": "#a5d6a7",
    "Q8_0": "#4dd0a5",
    "Q6_K": "#59b8e6",
    "Q5_K": "#7f96f0",
    "Q4_K": "#b58cf0",
    "Q3_K": "#e08cd0",
    "Q2_K": "#ef8080",
}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _mb(nbytes: float) -> str:
    return f"{nbytes / (1024**2):.1f} MB"


# ---------------------------------------------------------------------------
# Data extraction (one function per section; all pure reads)
# ---------------------------------------------------------------------------


def _steps_dir(run_root: Path) -> Path:
    return run_root / "steps"


def extract_summary(run_root: Path) -> dict[str, Any]:
    run_meta = _read_json(run_root / "run.json") or {}
    resolve = _read_json(_steps_dir(run_root) / "01_resolve" / "output.json")
    fit_plan = _read_json(run_root / "fit_plan.json")
    if not run_meta and not resolve:
        return {"available": False}
    desc = (resolve or {}).get("descriptor") or {}
    return {
        "available": True,
        "source": "run.json · steps/01_resolve/output.json",
        "model_ref": run_meta.get("model_ref") or (resolve or {}).get("user_ref"),
        "run_id": run_meta.get("run_id"),
        "status": run_meta.get("status"),
        "quant": run_meta.get("quant_label") or run_meta.get("quant_format"),
        "hf_repo_id": (resolve or {}).get("hf_repo_id"),
        "family": desc.get("family"),
        "layers": desc.get("layer_count"),
        "params": desc.get("parameter_count"),
        "context_length": desc.get("context_length"),
        "specialty_domain": desc.get("specialty_domain"),
        "fit_plan": fit_plan,
    }


def _load_catalog(run_root: Path) -> dict[str, Any] | None:
    for step in ("08_activation_features", "06_weight_features", "05_catalog"):
        cat = _read_json(_steps_dir(run_root) / step / "tensor_catalog.json")
        if cat:
            return cat
    return None


def _load_sensitivity(run_root: Path) -> dict[str, Any] | None:
    return _read_json(_steps_dir(run_root) / "12_sensitivity" / "sensitivity.json")


def extract_sensitivity(run_root: Path) -> dict[str, Any]:
    sens = _load_sensitivity(run_root)
    if not sens or not sens.get("rows"):
        return {"available": False}
    rows = sens["rows"]
    groups = sorted({r["group_id"] for r in rows})
    probes = list(sens.get("probe_types") or sorted({r["probe"] for r in rows}))
    cells: dict[str, dict[str, float]] = {}
    for r in rows:
        cells.setdefault(r["group_id"], {})[str(r["probe"]).upper()] = float(
            r.get("delta_kld") or 0.0
        )
    return {
        "available": True,
        "source": "steps/12_sensitivity/sensitivity.json",
        "method": sens.get("method"),
        "baseline": sens.get("baseline_type"),
        "groups": groups,
        "probes": [p.upper() for p in probes],
        "cells": cells,
    }


def extract_allocations(run_root: Path) -> dict[str, Any]:
    opt = _read_json(_steps_dir(run_root) / "13_optimize" / "output.json")
    if not opt or not opt.get("assignments"):
        return {"available": False}
    manifest = _read_json(_steps_dir(run_root) / "13_optimize" / "optimize_manifest.json") or {}
    catalog = _load_catalog(run_root) or {}
    sens = _load_sensitivity(run_root) or {}

    row_index = {
        (r["group_id"], str(r["probe"]).upper()): r for r in sens.get("rows") or []
    }
    baseline = str(sens.get("baseline_type") or "Q6_K").upper()
    history = (manifest.get("primary") or {}).get("history") or []
    last_move: dict[str, dict[str, Any]] = {}
    for h in history:
        last_move[h["group_id"]] = h

    groups_meta = catalog.get("groups") or {}
    tensors = catalog.get("tensors") or {}

    def group_elements(gid: str) -> int:
        g = groups_meta.get(gid) or {}
        return sum(
            int((tensors.get(n) or {}).get("n_elements") or 0)
            for n in g.get("tensor_names") or []
        )

    rows: list[dict[str, Any]] = []
    for gid, q in sorted(opt["assignments"].items()):
        q = str(q).upper()
        n_elem = group_elements(gid)
        try:
            nbytes = estimate_group_nbytes(n_elem, q) if n_elem else None
        except KeyError:
            nbytes = None
        srow = row_index.get((gid, q))
        dkld = float(srow["delta_kld"]) if srow else (0.0 if q == baseline else None)

        if q == baseline:
            reason = f"held at baseline {baseline} — no downgrade paid off"
        elif gid in last_move:
            mv = last_move[gid]
            reason = (
                f"greedy downgrade → {mv['to']}: saved {_mb(mv['delta_bytes'])} "
                f"for ΔKLD +{mv['delta_kld_inc']:.4f}"
            )
        else:
            reason = "assigned by optimizer"
        rows.append(
            {
                "group": gid,
                "quant": q,
                "bytes": nbytes,
                "delta_kld": dkld,
                "reason": reason,
            }
        )

    return {
        "available": True,
        "source": "steps/13_optimize/{output.json, optimize_manifest.json}",
        "method": opt.get("method"),
        "budget_bytes": opt.get("budget_bytes"),
        "estimated_bytes": opt.get("estimated_bytes"),
        "predicted_delta_kld": opt.get("predicted_delta_kld"),
        "baseline": baseline,
        "rows": rows,
        "n_decisions": len(history),
    }


def extract_pareto(run_root: Path) -> dict[str, Any]:
    manifest = _read_json(_steps_dir(run_root) / "13_optimize" / "optimize_manifest.json")
    if not manifest or not manifest.get("pareto"):
        return {"available": False}
    points = []
    for p in manifest["pareto"]:
        if p.get("estimated_bytes") is None:
            continue
        kld = p.get("predicted_delta_kld")
        if kld is None:
            kld = p.get("predicted_mean_kld")
        if kld is None:
            continue
        points.append({
            "size_mb": p["estimated_bytes"] / (1024**2),
            "kld": kld,
            "meets_budget": p.get("meets_budget"),
        })
    if not points:
        return {"available": False}
    return {
        "available": True,
        "source": "steps/13_optimize/optimize_manifest.json",
        "points": points,
    }


def extract_gates(run_root: Path) -> dict[str, Any]:
    val = _read_json(_steps_dir(run_root) / "15_validate" / "output.json")
    if not val:
        return {"available": False}
    tier1 = val.get("tier1") or {}
    metrics = tier1.get("metrics") or {}
    thresholds = tier1.get("gates") or {}
    gate_rows = []
    for key, thr_key, higher_better in (
        ("mean_kld", "mean_kld_max", False),
        ("p999_kld", "p999_kld_max", False),
        ("max_kld", "max_kld_max", False),
        ("top1_agree", "top1_agree_min", True),
    ):
        value = metrics.get(key)
        thr = thresholds.get(thr_key)
        if value is None or thr is None:
            continue
        ok = value >= thr if higher_better else value <= thr
        gate_rows.append(
            {
                "metric": key,
                "value": float(value),
                "threshold": float(thr),
                "higher_better": higher_better,
                "pass": bool(ok),
            }
        )
    return {
        "available": True,
        "source": "steps/15_validate/output.json",
        "verdict": val.get("verdict"),
        "method": val.get("method"),
        "tier1_pass": tier1.get("pass"),
        "tier1_note": tier1.get("note"),
        "gates": gate_rows,
    }


def extract_benchmarks(run_root: Path) -> dict[str, Any]:
    """Read the same-model experiment table if it was copied into this run."""
    path = Path(run_root) / "experiment" / "comparison.json"
    data = _read_json(path)
    if not data:
        return {"available": False}
    return {
        "available": True,
        "source": "experiment/comparison.json",
        "comparison": data,
    }


def extract_reproducibility(run_root: Path) -> dict[str, Any]:
    resolve = _read_json(_steps_dir(run_root) / "01_resolve" / "output.json") or {}
    freeze = _read_json(_steps_dir(run_root) / "09_freeze_gguf" / "output.json") or {}
    imatrix = _read_json(_steps_dir(run_root) / "10_imatrix" / "output.json") or {}
    corpus = _read_json(_steps_dir(run_root) / "07_corpus" / "output.json") or {}
    opt = _read_json(_steps_dir(run_root) / "13_optimize" / "output.json") or {}
    rows = [
        ("model source", resolve.get("hf_repo_id") or resolve.get("user_ref")),
        ("source sha256", resolve.get("source_sha256")),
        ("frozen gguf sha256", freeze.get("gguf_sha256")),
        ("imatrix sha256", imatrix.get("imatrix_sha256")),
        ("corpus id", corpus.get("corpus_id")),
        ("recipe", opt.get("recipe_path")),
    ]
    rows = [(k, v) for k, v in rows if v]
    if not rows:
        return {"available": False}
    return {
        "available": True,
        "source": "steps/{01_resolve,07_corpus,09_freeze_gguf,10_imatrix,13_optimize}/output.json",
        "rows": rows,
    }


def _find_sweep_record(run_root: Path) -> tuple[dict[str, Any], str, Path] | None:
    """Sweep frontier.json with measured points + its source caption."""
    rel = "15_validate/frontier/frontier.json"
    path = _steps_dir(Path(run_root)) / rel
    data = _read_json(path)
    if data and data.get("points"):
        return data, f"steps/{rel}", path
    return None


def _sweep_noise_floor(run_root: Path) -> tuple[float | None, float | None]:
    """Per-metric noise over probed background-rung cells (mean, P99)."""
    sens = _read_json(
        _steps_dir(Path(run_root)) / "12_sensitivity" / "sensitivity.json") or {}
    base = str(sens.get("baseline_type") or "").upper()
    if not base:
        return None, None
    bg = [r for r in sens.get("rows") or []
          if str(r.get("probe") or "").upper() == base]
    means = [float(r["kld_mean"]) for r in bg
             if r.get("kld_mean") is not None]
    tails = [float(r["kld_tail_1pct"]) for r in bg
             if r.get("kld_tail_1pct") is not None]
    return ((sum(means) / len(means)) if means else None,
            (sum(tails) / len(tails)) if tails else None)


def _sweep_ppl_base(sweep_file: Path) -> float | None:
    """Best-effort BF16 PPL(base) from sibling Tier-1 logs."""
    import re as _re

    for log in sorted(sweep_file.parent.glob("exports/*/llama-perplexity-tier1.log")):
        try:
            m = _re.search(r"Mean PPL\(base\)\s*:\s*([-\d.eE+]+)",
                           log.read_text(encoding="utf-8"))
        except OSError:
            continue
        if m:
            return float(m.group(1))
    return None


def extract_sweep(run_root: Path) -> dict[str, Any]:
    """Measured sweep frontier (all four Tier-1 metrics) for charting."""
    run_root = Path(run_root)
    found = _find_sweep_record(run_root)
    if not found:
        return {"available": False}
    data, source, sweep_file = found
    pts = sorted(
        (m for m in data["points"]
         if m.get("actual_bytes") and (m.get("tier1") or {}).get("metrics")),
        key=lambda m: int(m["actual_bytes"]),
    )
    if not pts:
        return {"available": False}
    n_params = data.get("n_params")
    if not n_params:
        catalog = _load_catalog(run_root) or {}
        total = sum(int(t.get("n_elements") or 0)
                    for t in (catalog.get("tensors") or {}).values())
        n_params = total or None

    def bpw(nbytes: int) -> str | None:
        if not n_params:
            return None
        return f"{int(nbytes) * 8 / int(n_params):.2f}"

    noise = data.get("noise_floor_kld")
    noise_p99 = data.get("noise_floor_p99")
    if noise is None or noise_p99 is None:
        fb_mean, fb_p99 = _sweep_noise_floor(run_root)
        noise = noise if noise is not None else fb_mean
        noise_p99 = noise_p99 if noise_p99 is not None else fb_p99
    ppl_base = data.get("ppl_base")
    if ppl_base is None:
        ppl_base = _sweep_ppl_base(sweep_file)
    xl_rec = data.get("xl") or {}

    def series(key: str, tier1_level: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        out = []
        for m in pts:
            v = (m["tier1"].get(key) if tier1_level
                 else m["tier1"]["metrics"].get(key))
            if v is None:
                continue
            out.append({"size_mb": int(m["actual_bytes"]) / (1024**2),
                        "kld": float(v),
                        "bpw": bpw(int(m["actual_bytes"]))})
        xlp = None
        if xl_rec.get("actual_bytes") and xl_rec.get("tier1"):
            xv = (xl_rec["tier1"].get(key) if tier1_level
                  else (xl_rec["tier1"].get("metrics") or {}).get(key))
            if xv is not None:
                xlp = {"size_mb": int(xl_rec["actual_bytes"]) / (1024**2),
                       "kld": float(xv),
                       "bpw": bpw(int(xl_rec["actual_bytes"])),
                       "xl": True}
        return out, xlp

    # Perplexity and top-1 need 3 sig figs (their action lives in the
    # 3rd digit); the KLD pair reads cleanly at 2.
    specs = [
        ("mean_kld", "Mean KLD", False, True,
         noise, "noise floor (all-Q8 background)", 2),
        ("p99_kld", "99% KLD", False, True,
         noise_p99 if noise_p99 is not None else noise,
         "noise floor (all-Q8 background)", 2),
        ("top1_agree", "Top-1 agreement", False, False, None, None, 3),
        ("perplexity", "Perplexity", True, False,
         ppl_base, "BF16 base", 3),
    ]
    charts = []
    for key, title, tier1_level, log_y, ref, ref_label, sig_figs in specs:
        spts, xlp = series(key, tier1_level)
        if not spts:
            continue
        charts.append({"key": key, "title": title, "log_y": log_y,
                       "ref": ref, "ref_label": ref_label,
                       "sig_figs": sig_figs,
                       "points": spts + ([xlp] if xlp else [])})
    if not charts:
        return {"available": False}
    return {"available": True, "source": source,
            "n_params": n_params, "charts": charts}


def build_report_data(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    return {
        "summary": extract_summary(run_root),
        "allocations": extract_allocations(run_root),
        "sensitivity": extract_sensitivity(run_root),
        "pareto": extract_pareto(run_root),
        "sweep": extract_sweep(run_root),
        "gates": extract_gates(run_root),
        "benchmarks": extract_benchmarks(run_root),
        "reproducibility": extract_reproducibility(run_root),
    }


# ---------------------------------------------------------------------------
# SVG / HTML rendering (no external assets)
# ---------------------------------------------------------------------------


def _svg_hbar(items: list[tuple[str, float, str, str]], *, width: int = 860) -> str:
    """items: (label, value, color, annotation). Bars scaled to max value."""
    if not items:
        return ""
    row_h, gap, label_w, annot_w = 22, 6, 190, 150
    bar_w = width - label_w - annot_w - 20
    vmax = max(v for _, v, _, _ in items) or 1.0
    h = len(items) * (row_h + gap)
    parts = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
    ]
    for i, (label, value, color, annot) in enumerate(items):
        y = i * (row_h + gap)
        w = max(2.0, bar_w * value / vmax)
        parts.append(
            f'<text x="{label_w - 8}" y="{y + row_h - 6}" text-anchor="end" '
            f'class="svgt">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 6}" '
            f'rx="3" fill="{color}"/>'
            f'<text x="{label_w + w + 8:.1f}" y="{y + row_h - 6}" class="svgt dim">'
            f"{_esc(annot)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _kld_color(v: float, vmax: float) -> str:
    """0 → deep green, vmax → red, log-ish ramp in between."""
    if vmax <= 0:
        t = 0.0
    else:
        t = min(1.0, max(0.0, v / vmax)) ** 0.5
    r = int(46 + t * (214 - 46))
    g = int(160 - t * (160 - 74))
    b = int(122 - t * (122 - 74))
    return f"rgb({r},{g},{b})"


def _heatmap_table(sens: dict[str, Any]) -> str:
    groups, probes, cells = sens["groups"], sens["probes"], sens["cells"]
    vmax = max(
        (v for g in cells.values() for v in g.values()), default=0.0
    )
    head = "".join(f"<th>{_esc(p)}</th>" for p in probes)
    body = []
    for gid in groups:
        tds = []
        for p in probes:
            v = cells.get(gid, {}).get(p)
            if v is None:
                tds.append('<td class="dim">·</td>')
            else:
                tds.append(
                    f'<td style="background:{_kld_color(v, vmax)};color:#0b1220">'
                    f"{v:.4f}</td>"
                )
        body.append(f"<tr><th>{_esc(gid)}</th>{''.join(tds)}</tr>")
    return (
        '<table class="heat"><thead><tr><th>group ↓ / probe →</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        f'<p class="cap">ΔKLD vs {_esc(sens.get("baseline") or "baseline")} — '
        "green: cheap to compress · red: expensive.</p>"
    )


def _svg_scatter(
    points: list[dict[str, Any]],
    *,
    width: int = 640,
    height: int = 320,
    log_y: bool = False,
    ref_value: float | None = None,
    ref_label: str | None = None,
    ylabel: str = "predicted ΔKLD",
    sig_figs: int = 2,
) -> str:
    """Frontier scatter as inline SVG — rendered by the shared
    matplotlib figure builder, so the report charts use the same engine
    and adjustText label placement as the PNGs.

    ``points``: {size_mb, kld, bpw? (real bits-per-weight label), xl?}.
    No winner highlight anywhere — the frontier speaks.
    """
    from validate import frontier_svg

    def _fmt(v: float) -> str:
        return f"{v:.{sig_figs}g}"

    items: list[tuple[float, float, str | None]] = []
    xlp: tuple[float, float, str | None] | None = None
    for p in sorted(points, key=lambda p: p["size_mb"]):
        lab = (f"{_fmt(p['kld'])}, {p['bpw']} bpw"
               if p.get("bpw") else None)
        if p.get("xl"):
            xlp = (p["size_mb"], p["kld"],
                   f"XL {lab}" if lab is not None else None)
        else:
            items.append((p["size_mb"], p["kld"], lab))
    if not items:
        return ""
    return frontier_svg(items, title=None, ylabel=ylabel, log_y=log_y,
                        ref_value=ref_value, ref_label=ref_label, xl=xlp,
                        figsize=(6.4, 3.2))


def _section(title: str, source: str | None, inner: str) -> str:
    cap = f'<p class="cap src">source: {_esc(source)}</p>' if source else ""
    return f'<section><h2>{_esc(title)}</h2>{inner}{cap}</section>'


def _not_run(title: str, hint: str) -> str:
    return (
        f"<section><h2>{_esc(title)}</h2>"
        f'<p class="dim">not run — {_esc(hint)}</p></section>'
    )


_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 20px 80px; background: #0b1220; color: #dbe4f0;
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h1 span { color: #59d6e6; }
h2 { font-size: 18px; margin: 0 0 12px; color: #9fd8e8;
  border-bottom: 1px solid #22314a; padding-bottom: 6px; }
section { background: #111a2c; border: 1px solid #22314a; border-radius: 12px;
  padding: 20px 22px; margin: 18px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 5px 9px; text-align: left; }
thead th { color: #8aa2c0; font-weight: 600; border-bottom: 1px solid #2a3a55; }
tbody tr:nth-child(odd) { background: #0e1626; }
tbody th { font-weight: 500; color: #c6d4e6; white-space: nowrap; }
.heat td { text-align: center; font-variant-numeric: tabular-nums; border-radius: 3px; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 999px;
  font-weight: 700; font-size: 13px; }
.badge.RELEASE { background: #1d7a4f; color: #fff; }
.badge.PROVISIONAL { background: #b98a1c; color: #14100a; }
.badge.FAIL { background: #a33; color: #fff; }
.pass { color: #4dd0a5; font-weight: 600; } .fail { color: #ef8080; font-weight: 600; }
.dim, .cap { color: #71829c; } .cap { font-size: 12px; margin: 10px 0 0; }
.cap.src { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.svgt { font: 12px -apple-system, sans-serif; fill: #c6d4e6; }
.svgt.dim { fill: #71829c; }

.qtag { display: inline-block; min-width: 46px; text-align: center; padding: 1px 7px;
  border-radius: 5px; font-weight: 700; font-size: 12px; color: #0b1220; }
.kv { display: grid; grid-template-columns: 180px 1fr; gap: 3px 14px; font-size: 14px; }
.kv b { color: #8aa2c0; font-weight: 600; }
code { background: #0e1626; border: 1px solid #22314a; border-radius: 5px;
  padding: 1px 6px; font-size: 12.5px; }
"""


def render_html(data: dict[str, Any]) -> str:
    parts: list[str] = []
    s = data["summary"]

    # --- header + summary ---
    title_model = s.get("model_ref") or "unknown model" if s.get("available") else "unknown model"
    if s.get("available"):
        kv_rows = [
            ("model", s.get("model_ref")),
            ("run", s.get("run_id")),
            ("status", s.get("status")),
            ("quant target", s.get("quant")),
            ("hf source", s.get("hf_repo_id")),
            ("family", s.get("family")),
            ("layers", s.get("layers")),
            ("parameters", f"{s['params']:,}" if s.get("params") else None),
            ("trained context", s.get("context_length")),
            ("specialty domain", s.get("specialty_domain")),
        ]
        kv_html = "".join(
            f"<b>{_esc(k)}</b><span>{_esc(v)}</span>" for k, v in kv_rows if v is not None
        )
        inner = f'<div class="kv">{kv_html}</div>'
        fit = s.get("fit_plan")
        if fit:
            frows = [
                ("hardware", (fit.get("profile") or {}).get("id")),
                ("context sized for", fit.get("ctx")),
                ("usable memory", f"{fit.get('usable_gb', 0):.2f} GB"),
                ("kv cache", f"−{fit.get('kv_cache_gb', 0):.2f} GB"),
                (
                    "weight budget",
                    f"{fit.get('weight_budget_bytes', 0) / (1024**3):.2f} GB",
                ),
            ]
            fkv = "".join(
                f"<b>{_esc(k)}</b><span>{_esc(v)}</span>" for k, v in frows if v is not None
            )
            inner += (
                '<h2 style="margin-top:18px">Hardware fit</h2>'
                f'<div class="kv">{fkv}</div>'
            )
        parts.append(_section("Model", s.get("source"), inner))
    else:
        parts.append(_not_run("Model", "no run.json / resolve output found"))

    # --- allocations ---
    a = data["allocations"]
    if a.get("available"):
        head = (
            "<thead><tr><th>group</th><th>assigned</th><th>size</th>"
            "<th>ΔKLD</th><th>why</th></tr></thead>"
        )
        rows_html = []
        for r in a["rows"]:
            color = _QUANT_COLORS.get(r["quant"], "#8aa2c0")
            dkld = f"+{r['delta_kld']:.4f}" if r["delta_kld"] else ("0" if r["delta_kld"] == 0.0 else "—")
            size = _mb(r["bytes"]) if r["bytes"] else "—"
            rows_html.append(
                f"<tr><th>{_esc(r['group'])}</th>"
                f'<td><span class="qtag" style="background:{color}">{_esc(r["quant"])}</span></td>'
                f"<td>{size}</td><td>{dkld}</td>"
                f'<td class="dim">{_esc(r["reason"])}</td></tr>'
            )
        meta = (
            f'<p>budget {_mb(a["budget_bytes"])} · estimated {_mb(a["estimated_bytes"])} · '
            f'predicted ΔKLD +{a["predicted_delta_kld"]:.4f} · '
            f'{a["n_decisions"]} greedy decisions ({_esc(a["method"])})</p>'
        )
        table = f"<table>{head}<tbody>{''.join(rows_html)}</tbody></table>"

        bar_items = [
            (r["group"], float(r["bytes"]), _QUANT_COLORS.get(r["quant"], "#8aa2c0"), _mb(r["bytes"]))
            for r in sorted(a["rows"], key=lambda x: -(x["bytes"] or 0))
            if r["bytes"]
        ]
        bytes_chart = _svg_hbar(bar_items) if bar_items else ""
        parts.append(_section("Bit allocation", a.get("source"), meta + table))
        if bytes_chart:
            parts.append(
                _section(
                    "Where the bytes live",
                    a.get("source"),
                    bytes_chart
                    + '<p class="cap">Estimated group size at its assigned quant type.</p>',
                )
            )
    else:
        parts.append(_not_run("Bit allocation", "step 13 (optimize) has not produced a recipe"))

    # --- sensitivity heatmap ---
    sens = data["sensitivity"]
    if sens.get("available"):
        note = ""
        if "proxy" in str(sens.get("method") or ""):
            note = (
                '<p class="cap">⚠ proxy estimates from features — not measured '
                "probes. Run step 12 with llama.cpp tools for measured ΔKLD.</p>"
            )
        parts.append(
            _section("Sensitivity heatmap", sens.get("source"), _heatmap_table(sens) + note)
        )
    else:
        parts.append(_not_run("Sensitivity heatmap", "step 12 (sensitivity) has not run"))

    # --- sweep frontier (measured) over predicted pareto fallback ---
    sw = data.get("sweep") or {}
    if sw.get("available"):
        for chart in sw["charts"]:
            parts.append(
                _section(
                    f"Frontier — {chart['title']} vs size",
                    sw.get("source"),
                    _svg_scatter(chart["points"], log_y=chart["log_y"],
                                 ref_value=chart["ref"],
                                 ref_label=chart["ref_label"],
                                 ylabel=f"measured {chart['title']}",
                                 sig_figs=chart.get("sig_figs", 2))
                    + '<p class="cap">Each point is a full recipe measured '
                    "on held-out Tier-1; ★ is the Unsloth XL reference. "
                    "Labels are real bits-per-weight.</p>",
                )
            )
        p = data["pareto"]
        if p.get("available"):
            parts.append(
                _section(
                    "Size ↔ quality frontier (predicted)",
                    p.get("source"),
                    _svg_scatter(p["points"])
                    + '<p class="cap">DP-predicted allocations at each '
                    "budget (estimates, not measurements).</p>",
                )
            )
    else:
        p = data["pareto"]
        if p.get("available"):
            parts.append(
                _section(
                    "Size ↔ quality frontier",
                    p.get("source"),
                    _svg_scatter(p["points"])
                    + '<p class="cap">Each point is a full recipe optimized at a '
                    "different budget (DP-predicted).</p>",
                )
            )
        else:
            parts.append(_not_run("Size ↔ quality frontier", "no Pareto set from step 13"))

    # --- gates ---
    g = data["gates"]
    if g.get("available"):
        verdict = str(g.get("verdict") or "?")
        inner = f'<p>verdict <span class="badge {_esc(verdict)}">{_esc(verdict)}</span></p>'
        if g["gates"]:
            head = (
                "<thead><tr><th>metric</th><th>value</th><th>gate</th>"
                "<th>result</th></tr></thead>"
            )
            rws = []
            for row in g["gates"]:
                op = "≥" if row["higher_better"] else "≤"
                cls = "pass" if row["pass"] else "fail"
                word = "pass" if row["pass"] else "fail"
                rws.append(
                    f"<tr><th>{_esc(row['metric'])}</th><td>{row['value']:.6g}</td>"
                    f"<td>{op} {row['threshold']:.6g}</td>"
                    f'<td class="{cls}">{word}</td></tr>'
                )
            inner += f"<table>{head}<tbody>{''.join(rws)}</tbody></table>"
        if g.get("tier1_note"):
            inner += f'<p class="cap">{_esc(g["tier1_note"])}</p>'
        parts.append(_section("Validation gates (Tier 1)", g.get("source"), inner))
    else:
        parts.append(_not_run("Validation gates", "step 15 (validate) has not run"))

    # --- experiment (same model, same harness) ---
    b = data["benchmarks"]
    if b.get("available"):
        comp = b.get("comparison") or {}
        variants = ["bf16", "q4_k_m", "q5_k_m", "q6_k", "odg"]
        labels = {
            "bf16": "BF16",
            "q4_k_m": "Q4_K_M",
            "q5_k_m": "Q5_K_M",
            "q6_k": "Q6_K",
            "odg": "ODG",
        }
        head = (
            "<thead><tr><th>benchmark</th>"
            + "".join(f"<th>{labels[v]}</th>" for v in variants)
            + "</tr></thead>"
        )
        rws = []
        for row in comp.get("quality") or []:
            cells = []
            for v in variants:
                val = (row.get("scores") or {}).get(v)
                if val is None:
                    cells.append("—")
                elif 0 <= float(val) <= 1:
                    cells.append(f"{float(val) * 100:.1f}")
                else:
                    cells.append(f"{float(val):.3f}")
            rws.append(
                "<tr><th>"
                + _esc(row.get("label") or row.get("task"))
                + "</th>"
                + "".join(f"<td>{c}</td>" for c in cells)
                + "</tr>"
            )
        size_rws = []
        for row in comp.get("compression") or []:
            mb = row.get("bytes")
            size = f"{float(mb) / (1024**2):.1f} MB" if mb else "—"
            size_rws.append(f"<tr><th>{_esc(row.get('label'))}</th><td>{size}</td></tr>")
        inner = f"<p class='cap'>{_esc(comp.get('model') or '')}</p>"
        if rws:
            inner += f"<table>{head}<tbody>{''.join(rws)}</tbody></table>"
        if size_rws:
            inner += (
                "<table><thead><tr><th>model</th><th>size</th></tr></thead>"
                f"<tbody>{''.join(size_rws)}</tbody></table>"
            )
        if comp.get("size_note"):
            inner += f'<p class="cap">{_esc(comp["size_note"])}</p>'
        parts.append(_section("Experiment", b.get("source"), inner))
    else:
        parts.append(
            _not_run(
                "Experiment",
                "no experiment/comparison.json — run: odg experiment prepare "
                "&& odg experiment run --all && odg experiment compare",
            )
        )

    # --- reproducibility ---
    r = data["reproducibility"]
    if r.get("available"):
        kvh = "".join(
            f"<b>{_esc(k)}</b><span><code>{_esc(v)}</code></span>" for k, v in r["rows"]
        )
        parts.append(
            _section(
                "Reproducibility",
                r.get("source"),
                f'<div class="kv">{kvh}</div>'
                '<p class="cap">The recipe + these hashes rebuild the GGUF '
                "bit-for-bit: <code>odg export</code> from the same inputs.</p>",
            )
        )
    else:
        parts.append(_not_run("Reproducibility", "no provenance artifacts found"))

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>OpenDynamicGGUF report — {_esc(title_model)}</title>"
        f"<style>{_CSS}</style></head><body><main>"
        f"<h1><span>OpenDynamicGGUF</span> optimization report</h1>"
        f'<p class="dim">{_esc(title_model)} — every number below links back to a '
        "run artifact; nothing is recomputed at render time.</p>"
        f"{''.join(parts)}"
        "</main></body></html>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class ReportResult:
    run_root: str
    report_path: str
    sections_rendered: int
    sections_missing: list[str] = field(default_factory=list)
    steps_log: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def generate_report(run_root: Path, out_path: Path | None = None) -> ReportResult:
    run_root = Path(run_root)
    if not (run_root / "run.json").is_file():
        raise FileNotFoundError(f"Not a run directory (no run.json): {run_root}")

    data = build_report_data(run_root)
    html_text = render_html(data)
    out = Path(out_path) if out_path else run_root / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")

    missing = [k for k, v in data.items() if not v.get("available")]
    rendered = len(data) - len(missing)
    log = [
        f"1. Extracted {len(data)} sections from {run_root}",
        f"2. Rendered {rendered} sections; missing: {missing or 'none'}",
        f"3. Wrote {out} ({out.stat().st_size / 1024:.0f} KB, self-contained)",
    ]
    return ReportResult(
        run_root=str(run_root),
        report_path=str(out),
        sections_rendered=rendered,
        sections_missing=missing,
        steps_log=log,
    )
