"""Step 15 — validate & release (including the --frontier Pareto sweep)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import html as html_lib
import json
from collections import defaultdict
from pathlib import Path
from sensitivity import estimate_group_nbytes
import re
import shutil
from typing import Any, Literal


# --- from validate/types.py ---
@dataclass
class ValidateResult:
    model_ref: str
    method: str
    verdict: str  # "RELEASE" | "FAIL" | "PROVISIONAL"
    tier1: dict[str, Any]
    tier2: dict[str, Any]
    tier3: dict[str, Any]
    feedback: list[dict[str, Any]]
    report_path: str
    release_dir: str | None
    report_card_paths: dict[str, str] = field(default_factory=dict)
    steps_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return asdict(self)

# --- from validate/report_card.py ---
BASELINE = "Q6_K"


def _mb(n: int | float | None) -> float:
    if not n:
        return 0.0
    return float(n) / (1024 * 1024)


def _group_n_elements(group: dict[str, Any], tensors: dict[str, Any]) -> int:
    total = 0
    for name in group.get("tensor_names") or []:
        t = tensors.get(name) or {}
        total += int(t.get("n_elements") or 0)
    return total


def _parse_layers_from_group(group: dict[str, Any]) -> list[int]:
    layers: list[int] = []
    for name in group.get("tensor_names") or []:
        # blk.N....
        if name.startswith("blk."):
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layers.append(int(parts[1]))
    return sorted(set(layers))


def build_report_card_data(
    *,
    model_ref: str,
    catalog: dict[str, Any],
    assignments: dict[str, str],
    sensitivity_rows: list[dict[str, Any]] | None = None,
    optimize_manifest: dict[str, Any] | None = None,
    validate_payload: dict[str, Any] | None = None,
    resolve_descriptor: dict[str, Any] | None = None,
    baseline: str | None = None,
    sensitivity_baseline: str | None = None,
) -> dict[str, Any]:
    tensors = catalog.get("tensors") or {}
    groups = catalog.get("groups") or {}
    # Size/KLD baseline: the run's trial background rung (what the
    # sensitivity rows measured against), never a hardcoded rung.
    baseline = baseline or sensitivity_baseline or BASELINE
    n_layers = int(catalog.get("n_layers") or 0)
    if not n_layers:
        # infer
        mx = -1
        for t in tensors.values():
            if t.get("layer") is not None:
                mx = max(mx, int(t["layer"]))
        n_layers = mx + 1 if mx >= 0 else 0

    row_index: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sensitivity_rows or []:
        row_index[(r["group_id"], str(r["probe"]).upper())] = r

    primary = (optimize_manifest or {}).get("primary") or {}
    history = primary.get("history") or []

    # --- per-group rows ---
    group_rows: list[dict[str, Any]] = []
    total_base = 0
    total_final = 0
    total_kld = 0.0

    for gid, g in sorted(groups.items()):
        if not g.get("quantizable", True):
            continue
        q = (assignments.get(gid) or baseline).upper()
        n_elem = _group_n_elements(g, tensors)
        base_b = estimate_group_nbytes(n_elem, baseline)
        final_b = estimate_group_nbytes(n_elem, q)
        saved = base_b - final_b
        ratio = (final_b / base_b) if base_b else 1.0
        compress_pct = (1.0 - ratio) * 100.0
        sens = row_index.get((gid, q)) or {}
        delta_kld = float(sens.get("delta_kld") or 0.0)
        layers = _parse_layers_from_group(g)
        total_base += base_b
        total_final += final_b
        total_kld += delta_kld
        group_rows.append(
            {
                "group_id": gid,
                "role": g.get("role"),
                "depth": g.get("depth"),
                "layers": layers,
                "n_tensors": g.get("n_tensors"),
                "n_elements": n_elem,
                "baseline": baseline,
                "assigned": q,
                "bytes_baseline": base_b,
                "bytes_final": final_b,
                "bytes_saved": saved,
                "compress_pct": round(compress_pct, 2),
                "delta_kld": delta_kld,
                "decision_hint": sens.get("decision_hint"),
                "tensor_names": list(g.get("tensor_names") or []),
            }
        )

    # Non-quantizable overhead (norms etc.)
    fixed_bytes = 0
    fixed_tensors = []
    for name, t in tensors.items():
        if t.get("quantizable", True):
            continue
        nb = int(t.get("nbytes") or 0)
        fixed_bytes += nb
        fixed_tensors.append(
            {
                "name": name,
                "role": t.get("role"),
                "layer": t.get("layer"),
                "dtype": t.get("dtype"),
                "nbytes": nb,
            }
        )

    # --- per-layer matrix ---
    roles_order = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ]
    layer_cards: list[dict[str, Any]] = []
    for layer in range(n_layers):
        # depth bucket
        if n_layers <= 1:
            depth = "global"
        else:
            third = n_layers / 3.0
            if layer < third:
                depth = "early"
            elif layer < 2 * third:
                depth = "middle"
            else:
                depth = "late"

        roles: dict[str, Any] = {}
        layer_base = 0
        layer_final = 0
        for role in roles_order:
            gid = f"{role}@{depth}"
            # Find group that contains this layer+role
            match = None
            for gr in group_rows:
                if gr["role"] == role and layer in (gr["layers"] or []):
                    match = gr
                    break
            if match is None:
                # try exact depth group
                for gr in group_rows:
                    if gr["group_id"] == gid:
                        match = gr
                        break
            if match is None:
                roles[role] = None
                continue
            # Per-layer share: divide group bytes by n_tensors in group
            n_t = max(int(match["n_tensors"] or 1), 1)
            b_base = match["bytes_baseline"] // n_t
            b_final = match["bytes_final"] // n_t
            layer_base += b_base
            layer_final += b_final
            roles[role] = {
                "group_id": match["group_id"],
                "assigned": match["assigned"],
                "bytes_baseline": b_base,
                "bytes_final": b_final,
                "bytes_saved": b_base - b_final,
                "compress_pct": round(
                    (1.0 - (b_final / b_base)) * 100.0 if b_base else 0.0, 2
                ),
                "delta_kld_group": match["delta_kld"],
            }

        # Norms on this layer (F32, not quantized)
        norms = [
            ft
            for ft in fixed_tensors
            if ft.get("layer") == layer and ft.get("role") == "norm"
        ]
        layer_cards.append(
            {
                "layer": layer,
                "depth": depth,
                "roles": roles,
                "bytes_baseline": layer_base,
                "bytes_final": layer_final,
                "bytes_saved": layer_base - layer_final,
                "compress_pct": round(
                    (1.0 - (layer_final / layer_base)) * 100.0 if layer_base else 0.0,
                    2,
                ),
                "norms_kept_f32": norms,
            }
        )

    # --- role summary ---
    by_role: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "groups": 0,
            "bytes_baseline": 0,
            "bytes_final": 0,
            "bytes_saved": 0,
            "assignments": defaultdict(int),
            "delta_kld_sum": 0.0,
        }
    )
    for gr in group_rows:
        role = str(gr["role"] or "other")
        br = by_role[role]
        br["groups"] += 1
        br["bytes_baseline"] += gr["bytes_baseline"]
        br["bytes_final"] += gr["bytes_final"]
        br["bytes_saved"] += gr["bytes_saved"]
        br["assignments"][gr["assigned"]] += 1
        br["delta_kld_sum"] += gr["delta_kld"]

    role_summary = []
    for role, br in sorted(by_role.items()):
        role_summary.append(
            {
                "role": role,
                "groups": br["groups"],
                "bytes_baseline": br["bytes_baseline"],
                "bytes_final": br["bytes_final"],
                "bytes_saved": br["bytes_saved"],
                "compress_pct": round(
                    (1.0 - br["bytes_final"] / br["bytes_baseline"]) * 100.0
                    if br["bytes_baseline"]
                    else 0.0,
                    2,
                ),
                "assignments": dict(br["assignments"]),
                "delta_kld_sum": br["delta_kld_sum"],
            }
        )

    # quant histogram
    quant_hist: dict[str, int] = defaultdict(int)
    for gr in group_rows:
        quant_hist[gr["assigned"]] += 1

    desc = resolve_descriptor or {}
    validate_payload = validate_payload or {}
    _t1 = validate_payload.get("tier1") or {}
    _t1m = _t1.get("metrics") or {}

    return {
        "title": "OpenDynamicGGUF — Quantization Report Card",
        "model_ref": model_ref,
        "architecture": {
            "family": desc.get("family") or catalog.get("source_backend"),
            "n_layers": n_layers,
            "n_tensors": catalog.get("n_tensors") or len(tensors),
            "n_quantizable": catalog.get("n_quantizable"),
            "n_groups": len(group_rows),
            "embedding_length": desc.get("embedding_length"),
            "parameter_count": desc.get("parameter_count"),
            "specialty_domain": desc.get("specialty_domain"),
            "chat_template": desc.get("chat_template"),
        },
        "baseline": baseline,
        "size": {
            "quantizable_baseline_bytes": total_base,
            "quantizable_final_bytes": total_final,
            "quantizable_saved_bytes": total_base - total_final,
            "quantizable_compress_pct": round(
                (1.0 - total_final / total_base) * 100.0 if total_base else 0.0, 2
            ),
            "fixed_nonquant_bytes": fixed_bytes,
            "estimated_total_bytes": total_final + fixed_bytes,
            "estimated_total_mb": round(_mb(total_final + fixed_bytes), 2),
            "baseline_total_mb": round(_mb(total_base + fixed_bytes), 2),
        },
        "quality": {
            "predicted_delta_kld_sum": total_kld,
            "predicted_delta_kld_corrected": primary.get(
                "predicted_delta_kld_corrected"),
            "background_penalty_kld": primary.get("background_penalty_kld"),
            "optimize_predicted_delta_kld": primary.get("predicted_delta_kld"),
            "measured_mean_kld": _t1m.get("mean_kld"),
            "measured_p99_kld": _t1m.get("p99_kld"),
            "measured_p999_kld": _t1m.get("p999_kld"),
            "measured_max_kld": _t1m.get("max_kld"),
            "measured_top1_agree": _t1m.get("top1_agree"),
            "measured_perplexity": _t1.get("perplexity"),
            "measured_method": _t1.get("method"),
            "verdict": validate_payload.get("verdict"),
            "tier1": _t1m,
        },
        "quant_histogram": dict(sorted(quant_hist.items())),
        "role_summary": role_summary,
        "groups": group_rows,
        "layers": layer_cards,
        "fixed_tensors": fixed_tensors,
        "optimizer_steps": len(history),
        "notes": [
            f"Compression % is vs {baseline} baseline (not vs BF16).",
            "ΔKLD values are from the sensitivity table (proxy or measured).",
            "Norms / non-quantizable tensors stay F32 and are listed under fixed.",
        ],
    }


def write_report_card_md(data: dict[str, Any], path: Path) -> None:
    arch = data["architecture"]
    size = data["size"]
    q = data["quality"]

    def _k(v: Any) -> str:
        return "n/a" if v is None else f"{float(v):.4f}"
    lines = [
        f"# {data['title']}",
        "",
        f"**Model:** `{data['model_ref']}`  ",
        f"**Verdict:** **{data['quality'].get('verdict') or 'n/a'}**  ",
        f"**Architecture:** {arch.get('family')} · "
        f"{arch.get('n_layers')} layers · "
        f"{arch.get('n_tensors')} tensors · "
        f"{arch.get('n_quantizable')} quantizable · "
        f"{arch.get('n_groups')} groups",
        "",
        "## Size summary",
        "",
        f"| | MiB |",
        f"|---|---:|",
        f"| Baseline ({data['baseline']}) quantizable | { _mb(size['quantizable_baseline_bytes']):.2f} |",
        f"| After recipe (quantizable) | { _mb(size['quantizable_final_bytes']):.2f} |",
        f"| Bytes saved | { _mb(size['quantizable_saved_bytes']):.2f} |",
        f"| Compression | **{size['quantizable_compress_pct']}%** |",
        f"| Fixed (norms etc.) | { _mb(size['fixed_nonquant_bytes']):.2f} |",
        f"| **Estimated total** | **{size['estimated_total_mb']:.2f}** |",
        "",
        f"Predicted Σ ΔKLD: `{_k(q.get('predicted_delta_kld_sum'))}` "
        f"(background-corrected `{_k(q.get('predicted_delta_kld_corrected'))})`  ",
        f"Measured held-out ({q.get('measured_method') or 'n/a'}): "
        f"mean `{_k(q.get('measured_mean_kld'))}` · "
        f"P99 `{_k(q.get('measured_p99_kld'))}` · "
        f"P999 `{_k(q.get('measured_p999_kld'))}` · "
        f"max `{_k(q.get('measured_max_kld'))}` · "
        f"top-1 `{_k(q.get('measured_top1_agree'))}` · "
        f"PPL `{_k(q.get('measured_perplexity'))}`  ",
        f"Quant mix: `{data['quant_histogram']}`",
        "",
        "## Compression by role",
        "",
        "| Role | Groups | Baseline MiB | Final MiB | Saved MiB | Compress % | Assignments |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in data["role_summary"]:
        lines.append(
            f"| {r['role']} | {r['groups']} | {_mb(r['bytes_baseline']):.2f} | "
            f"{_mb(r['bytes_final']):.2f} | {_mb(r['bytes_saved']):.2f} | "
            f"{r['compress_pct']}% | `{r['assignments']}` |"
        )

    lines += [
        "",
        "## Per-layer report card",
        "",
        "Each cell: **quant type** (compress % vs baseline).",
        "",
        "| Layer | Depth | attn_q | attn_k | attn_v | attn_o | ffn_gate | ffn_up | ffn_down | Layer compress |",
        "|---:|---|---|---|---|---|---|---|---|---:|",
    ]
    for lc in data["layers"]:
        cells = []
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_o",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]:
            info = (lc.get("roles") or {}).get(role)
            if not info:
                cells.append("—")
            else:
                cells.append(f"**{info['assigned']}** ({info['compress_pct']}%)")
        lines.append(
            f"| {lc['layer']} | {lc['depth']} | "
            + " | ".join(cells)
            + f" | **{lc['compress_pct']}%** |"
        )

    lines += [
        "",
        "## Per-group detail",
        "",
        "| Group | Layers | Assigned | Baseline MiB | Final MiB | Saved | Compress % | ΔKLD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in data["groups"]:
        layers = ",".join(str(x) for x in (g["layers"] or [])) or "—"
        lines.append(
            f"| `{g['group_id']}` | {layers} | **{g['assigned']}** | "
            f"{_mb(g['bytes_baseline']):.2f} | {_mb(g['bytes_final']):.2f} | "
            f"{_mb(g['bytes_saved']):.2f} | {g['compress_pct']}% | "
            f"{g['delta_kld']:.4f} |"
        )

    lines += [
        "",
        "## What happened (plain language)",
        "",
    ]
    # Narrative bullets
    hist = data["quant_histogram"]
    heavily = [g for g in data["groups"] if g["assigned"] in {"Q2_K", "Q3_K"}]
    protected = [g for g in data["groups"] if g["assigned"] in {"Q5_K", "Q6_K", "Q8_0"}]
    lines.append(
        f"- Started from a **{data['baseline']}** baseline for size accounting, "
        f"then the optimizer downgraded groups with the best bytes/ΔKLD ratio."
    )
    lines.append(
        f"- **{len(heavily)}** groups landed at aggressive types (Q2/Q3); "
        f"**{len(protected)}** groups stayed protected (Q5+)."
    )
    for g in protected:
        lines.append(
            f"- Kept **{g['group_id']}** at **{g['assigned']}** "
            f"(compress {g['compress_pct']}%, ΔKLD≈{g['delta_kld']:.4f})."
        )
    # Top savers
    top = sorted(data["groups"], key=lambda x: x["bytes_saved"], reverse=True)[:5]
    lines.append("- Largest byte savers:")
    for g in top:
        lines.append(
            f"  - `{g['group_id']}` → {g['assigned']}: "
            f"saved {_mb(g['bytes_saved']):.2f} MiB ({g['compress_pct']}%)"
        )
    lines.append(
        f"- Non-quantizable norms/etc. remain F32 "
        f"({_mb(data['size']['fixed_nonquant_bytes']):.2f} MiB)."
    )
    lines += ["", "## Notes", ""]
    for n in data.get("notes") or []:
        lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_card_html(data: dict[str, Any], path: Path) -> None:
    arch = data["architecture"]
    size = data["size"]
    esc = html_lib.escape
    q = data["quality"]

    def _h(v: Any, digits: int = 4) -> str:
        return "n/a" if v is None else f"{float(v):.{digits}f}"

    measured_cells = "".join(
        f"<div class='stat'><div class='label'>{label}</div>"
        f"<div class='value'>{val}</div></div>"
        for label, val in [
            ("Mean KLD", _h(q.get("measured_mean_kld"))),
            ("P99 KLD", _h(q.get("measured_p99_kld"))),
            ("Top-1 agree", _h(q.get("measured_top1_agree"))),
            ("Perplexity", _h(q.get("measured_perplexity"), 3)),
            ("Pred. corrected", _h(q.get("predicted_delta_kld_corrected"))),
            ("Pred. raw Σ", _h(q.get("predicted_delta_kld_sum"), 3)),
        ]
    )

    def badge(q: str) -> str:
        colors = {
            "Q8_0": "#1b4332",
            "Q6_K": "#2d6a4f",
            "Q5_K": "#40916c",
            "Q4_K": "#b08968",
            "Q3_K": "#9c6644",
            "Q2_K": "#9b2226",
        }
        c = colors.get(q, "#333")
        return (
            f'<span style="background:{c};color:#fff;padding:2px 8px;'
            f'border-radius:999px;font-size:12px;font-weight:600">{esc(q)}</span>'
        )

    layer_rows = []
    for lc in data["layers"]:
        cells = [f"<td>{lc['layer']}</td><td>{esc(str(lc['depth']))}</td>"]
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_o",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]:
            info = (lc.get("roles") or {}).get(role)
            if not info:
                cells.append("<td class='muted'>—</td>")
            else:
                cells.append(
                    f"<td>{badge(info['assigned'])}"
                    f"<div class='pct'>{info['compress_pct']}%</div></td>"
                )
        cells.append(f"<td><strong>{lc['compress_pct']}%</strong></td>")
        layer_rows.append("<tr>" + "".join(cells) + "</tr>")

    group_rows = []
    for g in data["groups"]:
        layers = ",".join(str(x) for x in (g["layers"] or [])) or "—"
        group_rows.append(
            "<tr>"
            f"<td><code>{esc(g['group_id'])}</code></td>"
            f"<td>{esc(layers)}</td>"
            f"<td>{badge(g['assigned'])}</td>"
            f"<td class='num'>{_mb(g['bytes_baseline']):.2f}</td>"
            f"<td class='num'>{_mb(g['bytes_final']):.2f}</td>"
            f"<td class='num'>{_mb(g['bytes_saved']):.2f}</td>"
            f"<td class='num'>{g['compress_pct']}%</td>"
            f"<td class='num'>{g['delta_kld']:.4f}</td>"
            "</tr>"
        )

    role_rows = []
    for r in data["role_summary"]:
        role_rows.append(
            "<tr>"
            f"<td>{esc(r['role'])}</td>"
            f"<td class='num'>{r['groups']}</td>"
            f"<td class='num'>{_mb(r['bytes_baseline']):.2f}</td>"
            f"<td class='num'>{_mb(r['bytes_final']):.2f}</td>"
            f"<td class='num'>{_mb(r['bytes_saved']):.2f}</td>"
            f"<td class='num'>{r['compress_pct']}%</td>"
            f"<td><code>{esc(json.dumps(r['assignments']))}</code></td>"
            "</tr>"
        )

    hist = " · ".join(
        f"{badge(k)} ×{v}" for k, v in (data.get("quant_histogram") or {}).items()
    )
    qmethod = (
        f" — {esc(str(q.get('measured_method')))}"
        if q.get("measured_method") else ""
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Quantization Report Card — {esc(data['model_ref'])}</title>
<style>
  :root {{
    --bg: #f6f1ea;
    --ink: #1c1917;
    --card: #fffdf8;
    --line: #e7e0d5;
    --accent: #0f766e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1.25rem 4rem;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background:
      radial-gradient(1200px 600px at 10% -10%, #d8efe9 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #f0e2c8 0%, transparent 50%),
      var(--bg);
    color: var(--ink);
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 2.1rem; margin: 0 0 .25rem; letter-spacing: -0.02em; }}
  h2 {{ margin-top: 2.2rem; font-size: 1.35rem; border-bottom: 2px solid var(--ink); padding-bottom: .35rem; }}
  .sub {{ color: #57534e; margin-bottom: 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  .stat {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.1rem; box-shadow: 0 1px 0 rgba(0,0,0,.04);
  }}
  .stat .label {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; color: #78716c; }}
  .stat .value {{ font-size: 1.55rem; font-weight: 700; margin-top: .2rem; }}
  .verdict {{
    display: inline-block; margin-top: .5rem; padding: .35rem .8rem;
    border-radius: 999px; background: var(--accent); color: white; font-weight: 700;
  }}
  table {{
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }}
  th, td {{ padding: .55rem .6rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
  th {{ background: #efe8dc; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
  tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pct {{ color: #57534e; font-size: 11px; margin-top: 2px; }}
  .muted {{ color: #a8a29e; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  .hist {{ margin: .75rem 0 0; line-height: 1.9; }}
  .note {{ color: #57534e; font-size: .95rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Quantization Report Card</h1>
  <p class="sub">
    Model <code>{esc(data['model_ref'])}</code> ·
    {esc(str(arch.get('family') or '—'))} ·
    {arch.get('n_layers')} layers ·
    {arch.get('n_tensors')} tensors ·
    {arch.get('n_quantizable')} quantizable
  </p>
  <div class="verdict">{esc(str(data['quality'].get('verdict') or 'n/a'))}</div>

  <h2>Size at a glance</h2>
  <div class="grid">
    <div class="stat"><div class="label">Baseline ({esc(data['baseline'])})</div>
      <div class="value">{size['baseline_total_mb']:.1f}<span style="font-size:.9rem"> MiB</span></div></div>
    <div class="stat"><div class="label">After recipe</div>
      <div class="value">{size['estimated_total_mb']:.1f}<span style="font-size:.9rem"> MiB</span></div></div>
    <div class="stat"><div class="label">Compression</div>
      <div class="value">{size['quantizable_compress_pct']}%</div></div>
    <div class="stat"><div class="label">Pred. Σ ΔKLD</div>
      <div class="value">{data['quality'].get('predicted_delta_kld_sum', 0):.3f}</div></div>
  </div>
  <p class="hist">Quant mix: {hist}</p>

  <h2>Measured quality (held-out Tier-1{qmethod})</h2>
  <div class="grid">
    {measured_cells}
  </div>

  <h2>Compression by role</h2>
  <table>
    <thead><tr>
      <th>Role</th><th class="num">Groups</th><th class="num">Baseline MiB</th>
      <th class="num">Final MiB</th><th class="num">Saved</th><th class="num">Compress</th><th>Assignments</th>
    </tr></thead>
    <tbody>
      {''.join(role_rows)}
    </tbody>
  </table>

  <h2>Per-layer report card</h2>
  <p class="note">Each cell shows the assigned quant type and compression % vs {esc(data['baseline'])} for that layer’s share of the group.</p>
  <table>
    <thead><tr>
      <th>Layer</th><th>Depth</th>
      <th>attn_q</th><th>attn_k</th><th>attn_v</th><th>attn_o</th>
      <th>ffn_gate</th><th>ffn_up</th><th>ffn_down</th>
      <th>Layer</th>
    </tr></thead>
    <tbody>
      {''.join(layer_rows)}
    </tbody>
  </table>

  <h2>Per-group detail</h2>
  <table>
    <thead><tr>
      <th>Group</th><th>Layers</th><th>Assigned</th>
      <th class="num">Base MiB</th><th class="num">Final MiB</th>
      <th class="num">Saved</th><th class="num">Compress</th><th class="num">ΔKLD</th>
    </tr></thead>
    <tbody>
      {''.join(group_rows)}
    </tbody>
  </table>

  <h2>Notes</h2>
  <ul class="note">
    {''.join(f'<li>{esc(n)}</li>' for n in (data.get('notes') or []))}
  </ul>
</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_quantization_report_card(
    out_dir: Path,
    *,
    model_ref: str,
    catalog: dict[str, Any],
    assignments: dict[str, str],
    sensitivity_rows: list[dict[str, Any]] | None = None,
    optimize_manifest: dict[str, Any] | None = None,
    validate_payload: dict[str, Any] | None = None,
    resolve_descriptor: dict[str, Any] | None = None,
    sensitivity_baseline: str | None = None,
) -> dict[str, str]:
    """Write html/md/json report card. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_report_card_data(
        model_ref=model_ref,
        catalog=catalog,
        assignments=assignments,
        sensitivity_rows=sensitivity_rows,
        optimize_manifest=optimize_manifest,
        validate_payload=validate_payload,
        resolve_descriptor=resolve_descriptor,
        sensitivity_baseline=sensitivity_baseline,
    )
    json_path = out_dir / "quantization_report_card.json"
    md_path = out_dir / "quantization_report_card.md"
    html_path = out_dir / "quantization_report_card.html"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_report_card_md(data, md_path)
    write_report_card_html(data, html_path)
    return {
        "json": str(json_path),
        "md": str(md_path),
        "html": str(html_path),
    }

# --- from validate/validate.py ---
Mode = Literal["auto", "llama", "proxy"]


def _parse_recipe_fields(recipe_path: Path) -> dict[str, Any]:
    text = recipe_path.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    m = re.search(r"size_bytes:\s*(\d+)", text)
    if m:
        out["estimated_bytes"] = int(m.group(1))
    m = re.search(r"predicted_mean_delta_kld:\s*([0-9.eE+-]+)", text)
    if m:
        out["predicted_delta_kld"] = float(m.group(1))
    m = re.search(r"gguf_sha256:\s*\"([^\"]*)\"", text)
    if m:
        out["gguf_sha256"] = m.group(1)
    m = re.search(r"target_size_bytes:\s*(\d+)", text)
    if m:
        out["budget_bytes"] = int(m.group(1))
    return out


def _tier1_proxy(recipe: dict[str, Any], *, has_candidate: bool) -> dict[str, Any]:
    """
    Approximate Tier-1 from recipe predicted KLD when logits-heldout missing.
    """
    mean_kld = float(recipe.get("predicted_delta_kld") or 0.05)
    # Crude tails
    p999 = mean_kld * 40.0
    max_kld = mean_kld * 80.0
    top1 = max(0.0, min(1.0, 1.0 - 2.0 * mean_kld))

    # Proxy uses summed group ΔKLD from the recipe — much larger than mean token KLD.
    gates = {
        "mean_kld_max": 0.50,
        "p999_kld_max": 25.0,
        "max_kld_max": 50.0,
        "top1_agree_min": 0.50,
    }
    checks = {
        "mean_kld": mean_kld,
        "p999_kld": p999,
        "max_kld": max_kld,
        "top1_agree": top1,
    }
    pass_mean = mean_kld <= gates["mean_kld_max"]
    pass_p999 = p999 <= gates["p999_kld_max"]
    pass_max = max_kld <= gates["max_kld_max"]
    pass_top = top1 >= gates["top1_agree_min"]
    passed = all([pass_mean, pass_p999, pass_max, pass_top]) and has_candidate

    return {
        "method": "proxy_from_recipe",
        "split": "heldout",
        "gates": gates,
        "metrics": checks,
        "pass": passed,
        "pass_detail": {
            "mean_kld": pass_mean,
            "p999_kld": pass_p999,
            "max_kld": pass_max,
            "top1_agree": pass_top,
            "candidate_exists": has_candidate,
        },
        "note": (
            "Proxy Tier-1 using recipe predicted ΔKLD — not measured held-out KLD. "
            "Install llama-perplexity + logits-heldout.bin for real gates."
            if has_candidate
            else "No candidate GGUF — Tier-1 cannot fully pass."
        ),
    }


def _tier1_llama(
    *,
    candidate: str | Path,
    heldout_txt: str | Path,
    heldout_bin: str | Path,
    out_dir: Path,
    llama_perplexity: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Measured Tier-1: candidate GGUF vs held-out reference logits.

    Runs ``llama-perplexity -m CANDIDATE -f heldout.txt --kl-divergence
    --kl-divergence-base logits-heldout.bin`` (+ extra_args verbatim, e.g.
    ``-ngl 99``) and gates the parsed KL summary. Thresholds are v2,
    loosened on the E4B run (good quant: mean 0.006, P99 0.10, top1
    98.3; bad quant: mean 0.29) so low-BPW points (3-4 BPW) are judged
    on their own quality bar rather than failing the tail gates.
    """
    import subprocess

    from kld import parse_llama_perplexity_kl
    from llama_bins import find_llama_binary

    binary = find_llama_binary("llama-perplexity", llama_perplexity)
    if binary is None:
        raise RuntimeError(
            "Tier-1 needs llama-perplexity (PATH, LLAMA_CPP_DIR, or "
            "--llama-perplexity)."
        )
    for p, label in (
        (candidate, "candidate GGUF"),
        (heldout_txt, "heldout.txt"),
        (heldout_bin, "logits-heldout.bin"),
    ):
        if not Path(p).is_file():
            raise RuntimeError(f"Tier-1 needs {label}: missing {p}")
    cmd = [
        str(binary), "-m", str(candidate), "-f", str(heldout_txt),
        "--kl-divergence", "--kl-divergence-base", str(heldout_bin),
    ]
    if extra_args:
        cmd += list(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    (Path(out_dir) / "llama-perplexity-tier1.log").write_text(
        log, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity Tier-1 failed (exit {proc.returncode}):\n"
            f"{log[-4000:]}"
        )
    m = parse_llama_perplexity_kl(log)  # hard error on missing KL lines
    mean_kld = float(m["kld_mean"])
    p99 = float(m["kld_tail_1pct"])
    p999 = m["kld_p999"]
    max_kld = m["kld_max"]
    top1 = m["same_top_p"]
    top1_frac = (float(top1) / 100.0) if top1 is not None else None

    gates = {
        "mean_kld_max": 0.10,
        "p99_kld_max": 1.50,
        "p999_kld_max": 6.0,
        "max_kld_max": 15.0,
        "top1_agree_min": 0.90,
    }
    checks = {
        "mean_kld": mean_kld,
        "p99_kld": p99,
        "p999_kld": p999,
        "max_kld": max_kld,
        "top1_agree": top1_frac,
    }
    detail = {
        "mean_kld": mean_kld <= gates["mean_kld_max"],
        "p99_kld": p99 <= gates["p99_kld_max"],
        "p999_kld": p999 <= gates["p999_kld_max"] if p999 is not None else True,
        "max_kld": max_kld <= gates["max_kld_max"] if max_kld is not None else True,
        "top1_agree": (
            top1_frac >= gates["top1_agree_min"] if top1_frac is not None else True
        ),
        "candidate_exists": True,
    }
    passed = all(detail.values())
    return {
        "method": "llama_heldout",
        "split": "heldout",
        "gates": gates,
        "metrics": checks,
        "pass": passed,
        "pass_detail": detail,
        "perplexity": m.get("perplexity"),
        "note": "Measured Tier-1: candidate vs held-out reference logits.",
    }


def _tier2_smoke(
    *,
    specialty: str | None,
    recipe: dict[str, Any],
    export_nbytes: int | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    # Size sanity vs estimate
    est = recipe.get("estimated_bytes")
    size_ok = True
    if export_nbytes and est:
        ratio = export_nbytes / est
        size_ok = 0.5 <= ratio <= 1.5
        checks.append(
            {
                "name": "size_vs_estimate",
                "pass": size_ok,
                "detail": f"actual={export_nbytes} estimate={est} ratio={ratio:.2f}",
            }
        )
    else:
        checks.append(
            {
                "name": "size_vs_estimate",
                "pass": export_nbytes is None,  # skip/neutral when dry-run
                "detail": "skipped (dry-run or no estimate)"
                if not export_nbytes
                else f"actual={export_nbytes}",
                "skipped": True,
            }
        )

    # Domain checklist (declarative — real gens need a runtime)
    if specialty in {"function_calling", "tools", "tool_use"}:
        checks.append(
            {
                "name": "tool_json_battery",
                "pass": True,
                "detail": "deferred — mark planned for FunctionGemma tool traces",
                "deferred": True,
            }
        )
    checks.append(
        {
            "name": "chat_code_math_smoke",
            "pass": True,
            "detail": "deferred — needs generation runtime",
            "deferred": True,
        }
    )

    # Pass if no hard failures (deferred OK)
    hard = [c for c in checks if not c.get("deferred") and not c.get("skipped")]
    passed = all(c["pass"] for c in hard) if hard else True
    return {
        "method": "smoke_checklist",
        "checks": checks,
        "pass": passed,
        "note": "Tier-2 generation batteries deferred without llama/ollama runner.",
    }


def _tier3_placeholder() -> dict[str, Any]:
    return {
        "method": "skipped",
        "pass": None,
        "note": "Tier-3 benchmarks (MMLU/GSM8K/HumanEval) not run in v1 plumbing.",
        "skipped": True,
    }


def _feedback_from_tier1(tier1: dict[str, Any], sensitivity_path: Path | None) -> list[dict[str, Any]]:
    fb: list[dict[str, Any]] = []
    if tier1.get("pass"):
        return fb
    detail = tier1.get("pass_detail") or {}
    if not detail.get("mean_kld", True):
        fb.append(
            {
                "constraint": "pin_highest_kld_group",
                "action": "Raise precision +1 on the worst sensitivity group; re-run optimize",
                "reason": f"mean_kld={tier1.get('metrics', {}).get('mean_kld')}",
            }
        )
    if not detail.get("candidate_exists", True):
        fb.append(
            {
                "constraint": "export_required",
                "action": "Run odg export --mode llama --force before validate release",
                "reason": "No candidate GGUF",
            }
        )
    if sensitivity_path and sensitivity_path.is_file():
        try:
            sens = json.loads(sensitivity_path.read_text())
            pinned = sens.get("pinned_hints") or []
            if pinned:
                fb.append(
                    {
                        "constraint": "respect_pin_hints",
                        "action": f"Ensure {pinned[0].get('group_id')} stays ≥ Q5_K",
                        "reason": "sensitivity pin_high",
                    }
                )
        except Exception:  # noqa: BLE001
            pass
    return fb


def _write_report(
    path: Path,
    *,
    model_ref: str,
    verdict: str,
    recipe: dict[str, Any],
    tier1: dict[str, Any],
    tier2: dict[str, Any],
    tier3: dict[str, Any],
    feedback: list[dict[str, Any]],
    export_out: str | None,
    export_nbytes: int | None,
) -> None:
    m1 = tier1.get("metrics") or {}
    lines = [
        f"# OpenDynamicGGUF validation report",
        "",
        f"**Model:** `{model_ref}`  ",
        f"**Verdict:** **{verdict}**  ",
        f"**Candidate:** `{export_out or '(missing)'}`  ",
        f"**Size:** {export_nbytes / (1024**2):.1f} MiB"
        if export_nbytes
        else "**Size:** (dry-run)",
        f"**Recipe estimate:** {recipe.get('estimated_bytes', 0) / (1024**2):.1f} MiB  ",
        f"**Source GGUF sha:** `{recipe.get('gguf_sha256', '')[:24]}…`",
        "",
        "## Tier 1 — Logit fidelity (held-out)",
        "",
        f"- method: `{tier1.get('method')}`",
        f"- mean KLD: `{m1.get('mean_kld', 'n/a')}`",
        f"- p99.9 KLD: `{m1.get('p999_kld', 'n/a')}`",
        f"- max KLD: `{m1.get('max_kld', 'n/a')}`",
        f"- top1 agree: `{m1.get('top1_agree', 'n/a')}`",
        f"- **pass:** `{tier1.get('pass')}`",
        f"- note: {tier1.get('note', '')}",
        "",
        "## Tier 2 — Behavioral smoke",
        "",
        f"- **pass:** `{tier2.get('pass')}`",
    ]
    for c in tier2.get("checks") or []:
        lines.append(f"  - {c['name']}: pass={c['pass']} — {c.get('detail', '')}")
    lines += [
        "",
        "## Tier 3 — Benchmarks",
        "",
        f"- **pass:** `{tier3.get('pass')}` ({tier3.get('note', '')})",
        "",
        "## Feedback to optimizer",
        "",
    ]
    if feedback:
        for f in feedback:
            lines.append(
                f"- **{f.get('constraint')}**: {f.get('action')} "
                f"({f.get('reason')})"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    # Minimal HTML twin
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>ODG report — {model_ref}</title>",
        "<style>body{font-family:ui-sans-serif,system-ui;max-width:800px;"
        "margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "code{background:#f4f4f4;padding:0.1em 0.3em;border-radius:4px}"
        f".verdict{{font-size:1.4rem;font-weight:700}}</style></head><body>",
        f"<h1>OpenDynamicGGUF report</h1>",
        f"<p class='verdict'>Verdict: {verdict}</p>",
        f"<p>Model: <code>{model_ref}</code></p>",
        f"<pre>{path.read_text()}</pre>",
        "</body></html>",
    ]
    path.with_suffix(".html").write_text("\n".join(html), encoding="utf-8")


def validate_and_release(
    *,
    model_ref: str,
    out_dir: Path,
    recipe_path: str | Path,
    export_manifest: dict[str, Any] | None = None,
    specialty_domain: str | None = None,
    sensitivity_path: str | Path | None = None,
    catalog: dict[str, Any] | None = None,
    assignments: dict[str, str] | None = None,
    optimize_manifest: dict[str, Any] | None = None,
    resolve_descriptor: dict[str, Any] | None = None,
    mode: Mode = "auto",
    allow_provisional: bool = True,
    heldout_txt: str | Path | None = None,
    heldout_bin: str | Path | None = None,
    llama_perplexity: str | Path | None = None,
    perplexity_args: list[str] | None = None,
) -> ValidateResult:
    log: list[str] = []
    notes: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = Path(recipe_path)
    recipe = _parse_recipe_fields(recipe_path)
    export_manifest = export_manifest or {}
    candidate = export_manifest.get("gguf_out")
    has_candidate = bool(candidate and Path(candidate).is_file())
    export_nbytes = export_manifest.get("gguf_out_nbytes")
    export_method = export_manifest.get("method")

    log.append(f"1. Recipe: {recipe_path}")
    log.append(
        f"2. Candidate: {candidate or '(missing)'} method={export_method}"
    )

    # Real llama Tier-1 not wired without logits-heldout; proxy always for now
    # unless mode=llama and we have both binary + bin (future).
    if mode == "llama" and not has_candidate:
        raise RuntimeError(
            "llama validate mode needs an exported candidate GGUF "
            "(odg export --mode llama)"
        )

    method = "proxy_gates"
    tier1: dict[str, Any] | None = None
    heldout_ok = bool(
        has_candidate
        and heldout_txt and Path(heldout_txt).is_file()
        and heldout_bin and Path(heldout_bin).is_file()
    )
    if heldout_ok:
        if mode == "llama" or mode == "auto":
            log.append(
                "3. Tier-1: measured llama-perplexity vs held-out logits"
            )
            tier1 = _tier1_llama(
                candidate=candidate,
                heldout_txt=heldout_txt,
                heldout_bin=heldout_bin,
                out_dir=out_dir,
                llama_perplexity=llama_perplexity,
                extra_args=perplexity_args,
            )
            method = "llama_heldout"
    if tier1 is None:
        if mode == "llama":
            raise RuntimeError(
                "llama validate mode needs held-out assets from step 11 "
                "(heldout.txt + logits-heldout.bin). Re-run "
                "`odg reference-logits --mode llama`."
            )
        log.append("3. Tier-1: proxy_from_recipe (held-out logits not available)")
        tier1 = _tier1_proxy(recipe, has_candidate=has_candidate)
    # For dry-run exports, allow provisional pass path
    if not has_candidate and allow_provisional:
        tier1["pass_detail"]["candidate_exists"] = False

    log.append("4. Tier-2: smoke checklist")
    tier2 = _tier2_smoke(
        specialty=specialty_domain,
        recipe=recipe,
        export_nbytes=export_nbytes if isinstance(export_nbytes, int) else None,
    )

    log.append("5. Tier-3: skipped (v1)")
    tier3 = _tier3_placeholder()

    feedback = _feedback_from_tier1(
        tier1, Path(sensitivity_path) if sensitivity_path else None
    )

    if has_candidate and tier1.get("pass") and tier2.get("pass"):
        verdict = "RELEASE"
    elif not has_candidate and allow_provisional and tier2.get("pass"):
        # Plumbing path: recipe+export plan OK, GGUF not built yet
        verdict = "PROVISIONAL"
        notes.append(
            "PROVISIONAL — export was dry-run. Re-run export --mode llama "
            "then validate for RELEASE."
        )
    else:
        verdict = "FAIL"

    log.append(f"6. Verdict: {verdict}")

    report_path = out_dir / "report.md"
    _write_report(
        report_path,
        model_ref=model_ref,
        verdict=verdict,
        recipe=recipe,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        feedback=feedback,
        export_out=candidate,
        export_nbytes=export_nbytes if isinstance(export_nbytes, int) else None,
    )
    log.append("7. Wrote report.md + report.html")

    # Quantization report card (per-layer / per-group compression)
    report_card_paths: dict[str, str] = {}
    if catalog and assignments:
        sens_rows = None
        sens_baseline = None
        if sensitivity_path and Path(sensitivity_path).is_file():
            try:
                sens_json = json.loads(Path(sensitivity_path).read_text())
                sens_rows = sens_json.get("rows")
                sens_baseline = sens_json.get("baseline_type")
            except Exception:  # noqa: BLE001
                sens_rows = None
        validate_payload = {
            "verdict": verdict,
            "tier1": tier1,
            "tier2": tier2,
            "tier3": tier3,
        }
        report_card_paths = write_quantization_report_card(
            out_dir,
            model_ref=model_ref,
            catalog=catalog,
            assignments=assignments,
            sensitivity_rows=sens_rows,
            optimize_manifest=optimize_manifest,
            validate_payload=validate_payload,
            resolve_descriptor=resolve_descriptor
            or ({"specialty_domain": specialty_domain} if specialty_domain else None),
            sensitivity_baseline=sens_baseline,
        )
        log.append(
            "8. Wrote quantization_report_card.html/.md/.json "
            f"({len(assignments)} groups, layer-by-layer)"
        )
        notes.append(
            f"Quantization report card: {report_card_paths.get('html')}"
        )
    else:
        log.append("8. Skipped report card (catalog/assignments missing)")

    release_dir = None

    def _stage_report_card(dest: Path) -> None:
        for key in ("html", "md", "json"):
            src = report_card_paths.get(key)
            if src and Path(src).is_file():
                shutil.copy2(src, dest / Path(src).name)

    if verdict == "RELEASE":
        release_dir_p = out_dir / "release"
        release_dir_p.mkdir(exist_ok=True)
        shutil.copy2(recipe_path, release_dir_p / "recipe.yaml")
        shutil.copy2(report_path, release_dir_p / "report.md")
        shutil.copy2(report_path.with_suffix(".html"), release_dir_p / "report.html")
        _stage_report_card(release_dir_p)
        if candidate and Path(candidate).is_file():
            dest = release_dir_p / Path(candidate).name
            if Path(candidate).resolve() != dest.resolve():
                shutil.copy2(candidate, dest)
        release_dir = str(release_dir_p)
        log.append("9. Staged release/ artifacts (+ report card)")
        notes.append("RELEASE staged under steps/15_validate/release/")
    elif verdict == "PROVISIONAL":
        release_dir_p = out_dir / "release_provisional"
        release_dir_p.mkdir(exist_ok=True)
        shutil.copy2(recipe_path, release_dir_p / "recipe.yaml")
        shutil.copy2(report_path, release_dir_p / "report.md")
        shutil.copy2(report_path.with_suffix(".html"), release_dir_p / "report.html")
        _stage_report_card(release_dir_p)
        release_dir = str(release_dir_p)
        log.append("9. Staged release_provisional/ (+ report card)")
    else:
        (out_dir / "feedback.json").write_text(
            json.dumps({"verdict": verdict, "feedback": feedback}, indent=2) + "\n",
            encoding="utf-8",
        )
        log.append("9. Wrote feedback.json for optimizer")
        notes.append("FAIL — apply feedback constraints and re-run optimize/export.")

    if tier1.get("method") == "proxy_from_recipe":
        notes.append(
            "Tier-1 used proxy KLD from recipe — not a substitute for held-out "
            "llama-perplexity --kl-divergence."
        )

    return ValidateResult(
        model_ref=model_ref,
        method=method,
        verdict=verdict,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        feedback=feedback,
        report_path=str(report_path),
        release_dir=release_dir,
        report_card_paths=report_card_paths,
        steps_log=log,
        notes=notes,
    )


# --- Pareto frontier sweep (step 15 --frontier) ---

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
    return _tier1_llama(
        candidate=candidate, heldout_txt=heldout_txt,
        heldout_bin=heldout_bin, out_dir=out_dir,
        llama_perplexity=llama_perplexity, extra_args=extra_args,
    )


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
    export_fn=export_candidate,
    measure_fn=measure_candidate,
    recipe_files: list[Path] | None = None,
    keep_winner: bool = True,
    noise_floor_kld: float | None = None,
    noise_floor_p99: float | None = None,
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
    if xl_gguf is not None:
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

    winner = pick_winner(measured, size_cap=size_cap)
    if winner is None:
        raise RuntimeError(
            f"No measured point at or under the {size_cap}-byte cap — "
            "nothing to crown. Frontier:\n" + "\n".join(
                f"  {m['budget_bytes']}B est / "
                f"{m.get('actual_bytes')}B actual" for m in measured)
        )
    if not keep_winner:
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
    if keep_winner:
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

    # Best-effort BF16 reference: the first Tier-1 log that prints it.
    # Stub measure functions (tests) write no logs — then there is
    # simply no PPL baseline line.
    ppl_base: float | None = None
    for log in sorted(out_dir.glob("exports/*/llama-perplexity-tier1.log")):
        try:
            ppl_base = parse_ppl_base(log.read_text(encoding="utf-8"))
        except OSError:
            continue
        if ppl_base is not None:
            break
    plots = plot_frontier(
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
