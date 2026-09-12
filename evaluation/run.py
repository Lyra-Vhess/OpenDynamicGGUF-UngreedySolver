"""HF-free local eval: original test splits + llama.cpp.

    python benchmark.py --model original.gguf --model opendynamic.gguf
    python benchmark.py --suite release --model original.gguf --model opendynamic.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fetch import fetch_all, link_models
from evaluation.server import LlamaServer, open_backend
from evaluation.suites import load_config, resolve_limits
from evaluation.tasks import PROTOCOLS, TaskResult, run_task

RESULTS = ROOT / "evaluation" / "results"
DEFAULT_MODELS = (
    "models/original.gguf",
    "models/q4_k_m.gguf",
    "models/opendynamic.gguf",
)
DEFAULT_TASKS = tuple(load_config()["tasks"])
LABELS = {
    "original": "Source",
    "original.gguf": "Source",
    "functiongemma-270m-bf16.gguf": "Source",
    "q4_k_m": "Q4_K_M",
    "q4_k_m.gguf": "Q4_K_M",
    "functiongemma-270m-q4_k_m.gguf": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q5_k_m.gguf": "Q5_K_M",
    "functiongemma-270m-q5_k_m.gguf": "Q5_K_M",
    "q6_k": "Q6_K",
    "q6_k.gguf": "Q6_K",
    "functiongemma-270m-q6_k.gguf": "Q6_K",
    "opendynamic": "OpenDynamicGGUF",
    "opendynamic.gguf": "OpenDynamicGGUF",
    "functiongemma-270m-odg.gguf": "OpenDynamicGGUF",
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _label(path: Path) -> str:
    return LABELS.get(path.name) or LABELS.get(path.stem) or path.stem


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def eval_one(
    gguf: Path,
    *,
    tasks: list[str],
    limits: dict[str, int | None],
    suite: str,
    out_dir: Path,
) -> dict[str, Any]:
    gguf = gguf.resolve()
    out: dict[str, Any] = {
        "schema": "odg/eval/v1",
        "gguf": str(gguf),
        "label": _label(gguf),
        "bytes": gguf.stat().st_size if gguf.is_file() else None,
        "suite": suite,
        "limits": limits,
        "created_at": _utc(),
        "tasks": {},
        "notes": [],
    }
    server = open_backend(gguf)
    if getattr(server, "_fallback_note", None):
        out["notes"].append(server._fallback_note)
        print(f"  note: {server._fallback_note}")
    try:
        for name in tasks:
            cap = limits.get(name)
            shown = "full" if cap is None else str(cap)
            print(f"  {out['label']}: {name} (n≤{shown})")
            try:
                result: TaskResult = run_task(server, name, limit=cap)
            except RuntimeError as exc:
                out["tasks"][name] = {"skipped": True, "reason": str(exc), "n": 0, "limit": cap}
                print(f"    skipped: {exc}")
                continue
            out["tasks"][name] = {
                "score": result.score,
                "metric": result.metric,
                "n": result.n,
                "limit": cap,
                "protocol": result.protocol,
                "extra": result.extra,
            }
            print(f"    {result.metric}={result.score:.4f}  n={result.n}")
    finally:
        server.close()
    _write(out_dir / f"{gguf.stem}.json", out)
    return out


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = list(DEFAULT_TASKS)
    suites = {r.get("suite") for r in results if r.get("suite")}
    notes: list[str] = []
    if len(suites) > 1:
        notes.append(
            "Mixed suites in these results — scores are not comparable. "
            "Re-run every GGUF with the same --suite."
        )
    rows = []
    for task in tasks:
        scores = {}
        for r in results:
            t = (r.get("tasks") or {}).get(task) or {}
            scores[r["label"]] = t.get("score")
        quantized = {k: v for k, v in scores.items() if k != "Source" and isinstance(v, float)}
        best = max(quantized, key=quantized.get) if quantized else None
        rows.append(
            {
                "task": task,
                "label": {
                    "mmlu": "MMLU",
                    "gsm8k": "GSM8K",
                    "hellaswag": "HellaSwag",
                    "arc_challenge": "ARC-C",
                }.get(task, task),
                "scores": scores,
                "best_quantized": best,
                "protocol": PROTOCOLS[task],
            }
        )
    sizes = [
        {"label": r["label"], "bytes": r.get("bytes"), "gguf": r.get("gguf")}
        for r in results
    ]
    suite_out: Any
    if len(suites) == 1:
        suite_out = next(iter(suites))
    elif suites:
        suite_out = sorted(suites)
    else:
        suite_out = None
    return {
        "schema": "odg/eval/v1",
        "created_at": _utc(),
        "suite": suite_out,
        "quality": rows,
        "compression": sizes,
        "protocols": PROTOCOLS,
        "notes": notes,
    }


def render_markdown(comp: dict[str, Any]) -> str:
    labels: list[str] = []
    for row in comp.get("quality") or []:
        for k in row.get("scores") or {}:
            if k not in labels:
                labels.append(k)
    if not labels:
        labels = [c["label"] for c in comp.get("compression") or []]
    header = "| Benchmark | " + " | ".join(labels) + " |"
    sep = "| --- |" + " --- |" * len(labels)
    lines = [
        "# Local evaluation (original datasets, llama.cpp)",
        "",
        "Original files kept as-is. Scoring protocols pinned below. No Hugging Face.",
        "",
    ]
    suite = comp.get("suite")
    if suite:
        lines += [f"Suite: `{suite}`.", ""]
    for note in comp.get("notes") or []:
        lines += [f"> {note}", ""]
    lines += ["## Quality", "", header, sep]
    for row in comp.get("quality") or []:
        cells = []
        for lab in labels:
            v = (row.get("scores") or {}).get(lab)
            if v is None:
                text = "—"
            else:
                text = f"{v * 100:.1f}"
                if lab == row.get("best_quantized"):
                    text = f"**{text}**"
            cells.append(text)
        lines.append("| " + row["label"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## Compression", "", "| Model | Size |", "| --- | ---: |"]
    for row in comp.get("compression") or []:
        b = row.get("bytes")
        size = "—" if not b else f"{b / (1024**2):.1f} MB"
        lines.append(f"| {row['label']} | {size} |")
    lines += ["", "## Protocols"]
    for name, proto in (comp.get("protocols") or PROTOCOLS).items():
        lines.append(
            f"- **{name}**: {proto['shots']}-shot {proto['split']} · "
            f"{proto['metric']} · {proto['scoring']}"
        )
        lines.append(f"  source: {proto['source']}")
    lines.append("")
    return "\n".join(lines)


def _resolve_models(values: list[str]) -> list[Path]:
    out: list[Path] = []
    for v in values:
        given = Path(v)
        candidates = [given]
        if not given.is_absolute():
            candidates += [
                ROOT / given,
                ROOT / "models" / given,
                ROOT / "models" / given.name,
            ]
        found = next((p.resolve() for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"GGUF not found: {v}")
        out.append(found)
    return out


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["eval"]
    if argv[0] in {"fetch", "eval", "compare", "-h", "--help"}:
        return argv
    return ["eval", *argv]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Original-dataset / llama.cpp evaluator")
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fetch", help="Download original MMLU/GSM8K/HellaSwag/ARC files")
    f.add_argument("--tasks", default="mmlu,gsm8k,hellaswag,arc")
    e = sub.add_parser("eval", help="Score one or more local GGUFs")
    e.add_argument(
        "--model",
        action="append",
        dest="models",
        help="GGUF path (repeatable). Default: original, q4_k_m, opendynamic",
    )
    e.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    e.add_argument(
        "--suite",
        default="dev",
        choices=("dev", "release"),
        help="dev = 500/200/500/200; release = full official test splits (~26K)",
    )
    e.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override every task to this many items (0 = full). Default: the suite caps",
    )
    e.add_argument("--out", type=Path, default=RESULTS)
    c = sub.add_parser("compare", help="Write comparison.md from evaluation/results")
    c.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args(_normalize_argv(list(argv) if argv is not None else sys.argv[1:]))

    if args.action == "fetch":
        fetch_all([t.strip() for t in args.tasks.split(",") if t.strip()])
        return 0

    if args.action == "eval":
        link_models(ROOT)
        models = args.models or [m for m in DEFAULT_MODELS if (ROOT / m).is_file()]
        if not models:
            print(
                "ERROR: no GGUFs. Place them under models/ "
                "(original.gguf, q4_k_m.gguf, opendynamic.gguf) or pass --model.",
                file=sys.stderr,
            )
            return 1
        paths = _resolve_models(models)
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        limits = resolve_limits(args.suite, limit=args.limit, tasks=tasks)
        print(f"suite={args.suite}  limits={limits}")
        payloads = []
        for gguf in paths:
            print(f"Evaluating {gguf}")
            payloads.append(
                eval_one(
                    gguf,
                    tasks=tasks,
                    limits=limits,
                    suite=args.suite,
                    out_dir=args.out,
                )
            )
        comp = compare(payloads)
        md = render_markdown(comp)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "comparison.json").write_text(json.dumps(comp, indent=2) + "\n")
        (args.out / "comparison.md").write_text(md + "\n")
        print(md)
        print(f"wrote {args.out / 'comparison.md'}")
        return 0

    files = sorted(args.out.glob("*.json"))
    payloads = []
    for fpath in files:
        if fpath.name == "comparison.json":
            continue
        payloads.append(json.loads(fpath.read_text()))
    if not payloads:
        print("ERROR: no evaluation/results/*.json — run eval first", file=sys.stderr)
        return 1
    comp = compare(payloads)
    md = render_markdown(comp)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(comp, indent=2) + "\n")
    (args.out / "comparison.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
