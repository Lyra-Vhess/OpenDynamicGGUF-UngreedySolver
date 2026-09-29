"""CLI for OpenDynamicGGUF."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import ui


def split_extra_args(raw: str | None, flag: str) -> list[str] | None:
    """Parse a `--*-args \"EXTRA ARGS\"` string into an argv list.

    Passthrough to llama.cpp binaries (e.g. ``-ngl 99`` for GPU offload).
    This repo never interprets the contents — they are appended verbatim.
    """
    if raw is None or not raw.strip():
        return None
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        raise ValueError(f"{flag}: cannot parse {raw!r}: {exc}")
    return parts or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="odg",
        description="OpenDynamicGGUF — automatic dynamic GGUF quantization",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts root for checkpoints (default: ./artifacts)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- resolve (step 01) ---
    p_resolve = sub.add_parser(
        "resolve",
        help="Step 01: resolve model ref + checkpoint to filesystem store",
    )
    p_resolve.add_argument("--model", "-m", required=True)
    p_resolve.add_argument("--download-weights", action="store_true")
    p_resolve.add_argument(
        "--prefer-hf",
        action="store_true",
        help="For Ollama tags: use Hugging Face BF16 instead of local Ollama GGUF",
    )
    p_resolve.add_argument(
        "--run",
        default=None,
        help="Existing run_id to resume into (default: model's CURRENT run or new)",
    )
    p_resolve.add_argument(
        "--new-run",
        action="store_true",
        help="Always create a fresh run (do not resume CURRENT)",
    )
    p_resolve.add_argument(
        "--force",
        action="store_true",
        help="Re-run resolve even if already checkpointed as done",
    )
    p_resolve.add_argument(
        "--quant",
        "-q",
        default=None,
        metavar="FORMAT",
        help=(
            "Target framework (dynamic mix under the hood): "
            "q4_k_m, q5_k_m, q3_k_m, q4_k_s, q5_k_s, q6_k, q8_0, q2_k, "
            "iq4_xs, iq4_nl, iq3_m, iq2_xxs, q4_0. "
            "If omitted in a TTY, you will be asked interactively."
        ),
    )
    p_resolve.add_argument(
        "--no-ask",
        action="store_true",
        help="Do not prompt for quant format; use --quant or default q4_k_m",
    )
    p_resolve.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional extra copy of output JSON",
    )
    p_resolve.add_argument("--cache-dir", type=Path, default=None)
    p_resolve.add_argument("--no-explain", action="store_true")

    # --- load (step 02) ---
    p_load = sub.add_parser(
        "load",
        help="Step 02: open resolved model (GGUF/HF) and checkpoint tensor index",
    )
    p_load.add_argument("--model", "-m", default=None, help="Model ref (uses CURRENT run)")
    p_load.add_argument("--run", default=None, help="run_id (default: CURRENT for --model)")
    p_load.add_argument("--force", action="store_true", help="Re-run even if done")
    p_load.add_argument("--no-explain", action="store_true")

    # --- enumerate (step 03) ---
    p_enum = sub.add_parser(
        "enumerate",
        help="Step 03: list every tensor (name/shape/dtype/nbytes) from Step 02 index",
    )
    p_enum.add_argument("--model", "-m", default=None)
    p_enum.add_argument("--run", default=None)
    p_enum.add_argument("--force", action="store_true")
    p_enum.add_argument("--no-explain", action="store_true")

    # --- classify (step 04) ---
    p_cls = sub.add_parser(
        "classify",
        help="Step 04: assign role / depth / quantizable to every tensor",
    )
    p_cls.add_argument("--model", "-m", default=None)
    p_cls.add_argument("--run", default=None)
    p_cls.add_argument("--force", action="store_true")
    p_cls.add_argument("--no-explain", action="store_true")

    # --- catalog (step 05) ---
    p_cat = sub.add_parser(
        "catalog",
        help="Step 05: build tensor_catalog.json (source of truth for probes)",
    )
    p_cat.add_argument("--model", "-m", default=None)
    p_cat.add_argument("--run", default=None)
    p_cat.add_argument("--force", action="store_true")
    p_cat.add_argument("--no-explain", action="store_true")

    # --- weight-features (step 06) ---
    p_wf = sub.add_parser(
        "weight-features",
        help="Step 06: compute weight stats (mean/var/sparsity/outliers/norms)",
    )
    p_wf.add_argument("--model", "-m", default=None)
    p_wf.add_argument("--run", default=None)
    p_wf.add_argument("--force", action="store_true")
    p_wf.add_argument(
        "--only-quantizable",
        action="store_true",
        help="Skip non-quantizable tensors (norms, etc.)",
    )
    p_wf.add_argument("--no-explain", action="store_true")

    # --- corpus (step 07) ---
    p_corp = sub.add_parser(
        "corpus",
        help="Step 07: build calib/search/heldout text corpus (3-way split)",
    )
    p_corp.add_argument("--model", "-m", default=None)
    p_corp.add_argument("--run", default=None)
    p_corp.add_argument("--force", action="store_true")
    p_corp.add_argument(
        "--target-tokens",
        type=int,
        default=50_000,
        help="Approximate token budget (chars/4). Default 50000; use 300000+ for production",
    )
    p_corp.add_argument("--seed", type=int, default=42)
    p_corp.add_argument("--no-explain", action="store_true")

    # --- activation-features (step 08) ---
    p_af = sub.add_parser(
        "activation-features",
        help="Step 08: activation stats from calib (forward hooks or proxy)",
    )
    p_af.add_argument("--model", "-m", default=None)
    p_af.add_argument("--run", default=None)
    p_af.add_argument("--force", action="store_true")
    p_af.add_argument(
        "--mode",
        choices=("auto", "forward", "proxy"),
        default="auto",
        help="auto: forward if BF16+torch available, else proxy (default)",
    )
    p_af.add_argument(
        "--max-docs",
        type=int,
        default=32,
        help="Max calib docs for forward pass (default 32)",
    )
    p_af.add_argument("--no-explain", action="store_true")

    # --- freeze-gguf (step 09) ---
    p_fz = sub.add_parser(
        "freeze-gguf",
        help="Step 09: freeze hashed GGUF reference (BF16 convert or promote source)",
    )
    p_fz.add_argument("--model", "-m", default=None)
    p_fz.add_argument("--run", default=None)
    p_fz.add_argument("--force", action="store_true")
    p_fz.add_argument(
        "--mode",
        choices=("auto", "hf-convert", "promote"),
        default="auto",
        help="auto: HF→BF16 if possible, else promote working GGUF",
    )
    p_fz.add_argument(
        "--convert-script",
        type=Path,
        default=None,
        help="Path to llama.cpp convert_hf_to_gguf.py (or set LLAMA_CPP_DIR)",
    )
    p_fz.add_argument(
        "--require-bf16",
        action="store_true",
        help="Fail unless the frozen file is BF16/F16 (not Q8)",
    )
    p_fz.add_argument("--no-explain", action="store_true")

    # --- imatrix (step 10) ---
    p_im = sub.add_parser(
        "imatrix",
        help="Step 10: build importance matrix from calib.txt (llama-imatrix or proxy)",
    )
    p_im.add_argument("--model", "-m", default=None)
    p_im.add_argument("--run", default=None)
    p_im.add_argument("--force", action="store_true")
    p_im.add_argument(
        "--mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
        help="auto: llama-imatrix if found, else proxy scores",
    )
    p_im.add_argument(
        "--llama-imatrix",
        type=Path,
        default=None,
        help="Path to llama-imatrix binary (or set LLAMA_CPP_DIR)",
    )
    p_im.add_argument(
        "--chunks",
        type=int,
        default=64,
        help="llama-imatrix --chunks (default 64; 0 = omit flag)",
    )
    p_im.add_argument(
        "--imatrix-args",
        default=None,
        metavar="ARGS",
        help='Extra args appended verbatim to llama-imatrix, e.g. --imatrix-args "-ngl 99"',
    )
    p_im.add_argument("--no-explain", action="store_true")

    # --- reference-logits (step 11) ---
    p_lg = sub.add_parser(
        "reference-logits",
        help="Step 11: cache search/heldout reference logits for KL divergence",
    )
    p_lg.add_argument("--model", "-m", default=None)
    p_lg.add_argument("--run", default=None)
    p_lg.add_argument("--force", action="store_true")
    p_lg.add_argument(
        "--mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
        help="auto: llama-perplexity if found, else proxy manifest",
    )
    p_lg.add_argument(
        "--llama-perplexity",
        type=Path,
        default=None,
        help="Path to llama-perplexity (or set LLAMA_CPP_DIR)",
    )
    p_lg.add_argument(
        "--perplexity-args",
        default=None,
        metavar="ARGS",
        help='Extra args appended verbatim to llama-perplexity, e.g. --perplexity-args "-ngl 99"',
    )
    p_lg.add_argument("--no-explain", action="store_true")

    # --- reband (step 11b) ---
    p_reband = sub.add_parser(
        "reband",
        help="Step 11b: re-cut depth bands by imatrix importance (Fisher-Jenks)",
    )
    p_reband.add_argument("--model", "-m", default=None)
    p_reband.add_argument("--run", default=None)
    p_reband.add_argument("--force", action="store_true")
    p_reband.add_argument(
        "--bands-per-role",
        type=int,
        default=3,
        help="Contiguous bands per role (default 3, same count as thirds)",
    )
    p_reband.add_argument("--no-explain", action="store_true")

    # --- sensitivity (step 12) ---
    p_sens = sub.add_parser(
        "sensitivity",
        help="Step 12: probe groups → Δbytes/ΔKLD sensitivity table",
    )
    p_sens.add_argument("--model", "-m", default=None)
    p_sens.add_argument("--run", default=None)
    p_sens.add_argument("--force", action="store_true")
    p_sens.add_argument(
        "--mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
        help="auto/proxy: feature-estimated table; llama: real trial quant (needs tools)",
    )
    p_sens.add_argument(
        "--baseline",
        default=None,
        help="Baseline type for Δbytes (default: from run --quant profile)",
    )
    p_sens.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel trial probes (default: 1; llama mode only)",
    )
    p_sens.add_argument(
        "--llama-quantize",
        type=Path,
        default=None,
        help="Path to llama-quantize for trial quants (or set LLAMA_CPP_DIR)",
    )
    p_sens.add_argument(
        "--llama-perplexity",
        type=Path,
        default=None,
        help="Path to llama-perplexity for KL probes (or set LLAMA_CPP_DIR)",
    )
    p_sens.add_argument(
        "--perplexity-args",
        default=None,
        metavar="ARGS",
        help='Extra args appended verbatim to probe llama-perplexity runs, e.g. --perplexity-args "-ngl 99"',
    )
    p_sens.add_argument(
        "--quant",
        "-q",
        default=None,
        metavar="FORMAT",
        help="Override run quant target for this step",
    )
    p_sens.add_argument(
        "--probe-types",
        default=None,
        metavar="TYPES",
        help="Comma-separated probe grid override (default: from --quant profile)",
    )
    p_sens.add_argument(
        "--fixed-groups",
        default=None,
        metavar="GIDS",
        help="Comma-separated group ids kept at source precision (e.g. tensors "
        "llama-quantize cannot quantize). Skipped at probe time; accounted "
        "toward every budget at optimize time (pass the same flag there).",
    )
    p_sens.add_argument(
        "--kld-objective",
        choices=("tail_1pct", "mean"),
        default="mean",
        help="Pricing objective for lazy probing (default: mean)",
    )
    p_sens.add_argument(
        "--certificate",
        choices=("bounded", "exhaustive"),
        default="bounded",
        help="bounded: lazy price-gated probing (default); exhaustive: measure all",
    )
    p_sens.add_argument(
        "--lipschitz",
        type=float,
        default=None,
        help="Lipschitz bound L for pricing (default: auto-calibrated x2 margin)",
    )
    p_sens.add_argument(
        "--splice-farm",
        type=Path,
        default=None,
        metavar="DIR",
        help="Precomputed per-tensor rung-shard farm: assemble probe trials "
        "by symlink instead of running llama-quantize per probe (llama mode "
        "lazy driver only). Farm trials reuse the farm-build imatrix.",
    )
    p_sens.add_argument("--no-explain", action="store_true")

    # --- optimize (step 13) ---
    p_opt = sub.add_parser(
        "optimize",
        help="Step 13: greedy recipe under size budget → recipe.yaml",
    )
    p_opt.add_argument("--model", "-m", default=None)
    p_opt.add_argument("--run", default=None)
    p_opt.add_argument("--force", action="store_true")
    p_opt.add_argument(
        "--budget-mb",
        type=float,
        default=None,
        help="Target size in MiB (default: from run quant profile ratio)",
    )
    p_opt.add_argument(
        "--budget-ratio",
        type=float,
        default=None,
        help="If --budget-mb omitted, fraction of Q6_K baseline (default: from --quant)",
    )
    p_opt.add_argument(
        "--quant",
        "-q",
        default=None,
        metavar="FORMAT",
        help="Override run quant target for this step",
    )
    p_opt.add_argument(
        "--optimizer",
        choices=("greedy", "dp_mckp"),
        default="dp_mckp",
        help="Recipe optimizer (default: dp_mckp; greedy kept for A/B comparison)",
    )
    p_opt.add_argument(
        "--kld-objective",
        choices=("tail_1pct", "mean"),
        default="mean",
        help="Optimizer objective: mean KLD (default, with automatic P99 guardrail) or tail_1pct",
    )
    p_opt.add_argument(
        "--certificate",
        choices=("bounded", "exhaustive"),
        default="bounded",
        help="bounded: prune via Lipschitz bound (default); exhaustive: probe all columns",
    )
    p_opt.add_argument(
        "--lipschitz",
        type=float,
        default=None,
        help="Lipschitz bound L (default: auto-calibrated x2 margin from probed data)",
    )
    p_opt.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Process-level probe parallelism (default: 1)",
    )
    p_opt.add_argument(
        "--pareto-ratios",
        default=None,
        metavar="RATIOS",
        help="Comma-separated budget ratios (default: 0.55,0.65,0.72,0.80,0.90,1.0); "
        "points above the solve budget are dropped, the budget is a hard limit",
    )
    p_opt.add_argument(
        "--fixed-groups",
        default=None,
        metavar="GIDS",
        help="Comma-separated group ids kept at source precision (e.g. tensors "
        "llama-quantize cannot quantize). Excluded from candidates and recipe; "
        "their measured bytes count toward every budget.",
    )
    p_opt.add_argument("--no-explain", action="store_true")

    # --- export (step 14) ---
    p_ex = sub.add_parser(
        "export",
        help="Step 14: export candidate GGUF from recipe (llama-quantize or dry-run)",
    )
    p_ex.add_argument("--model", "-m", default=None)
    p_ex.add_argument("--run", default=None)
    p_ex.add_argument("--force", action="store_true")
    p_ex.add_argument(
        "--mode",
        choices=("auto", "llama", "dry-run"),
        default="auto",
    )
    p_ex.add_argument(
        "--llama-quantize",
        type=Path,
        default=None,
        help="Path to llama-quantize (or set LLAMA_CPP_DIR)",
    )
    p_ex.add_argument(
        "--base-type",
        default=None,
        help="Fallback/base type for llama-quantize (default: from run --quant)",
    )
    p_ex.add_argument(
        "--quant",
        "-q",
        default=None,
        metavar="FORMAT",
        help="Override run quant target for this step",
    )
    p_ex.add_argument("--no-explain", action="store_true")

    # --- validate (step 15) ---
    p_val = sub.add_parser(
        "validate",
        help="Step 15: validate candidate on held-out gates; stage release",
    )
    p_val.add_argument("--model", "-m", default=None)
    p_val.add_argument("--run", default=None)
    p_val.add_argument("--force", action="store_true")
    p_val.add_argument(
        "--mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
    )
    p_val.add_argument(
        "--strict",
        action="store_true",
        help="Do not allow PROVISIONAL verdict without a real GGUF",
    )
    p_val.add_argument(
        "--llama-perplexity",
        type=Path,
        default=None,
        help="Path to llama-perplexity for measured Tier-1 (or set LLAMA_CPP_DIR)",
    )
    p_val.add_argument(
        "--perplexity-args",
        default=None,
        metavar="ARGS",
        help='Extra args appended verbatim to Tier-1 llama-perplexity runs, e.g. --perplexity-args "-ngl 99"',
    )
    p_val.add_argument("--no-explain", action="store_true")

    # --- run (full pipeline in one command) ---
    p_run = sub.add_parser(
        "run",
        help="Run the full pipeline (steps 01–15) in one command",
    )
    p_run.add_argument("--model", "-m", required=True, help="Model ref (e.g. functiongemma:latest)")
    p_run.add_argument(
        "--quant",
        "-q",
        default=None,
        metavar="FORMAT",
        help="Target quant framework (omit in a TTY to pick interactively)",
    )
    p_run.add_argument(
        "--no-ask",
        action="store_true",
        help="Do not prompt for quant format; use --quant or default q4_k_m",
    )
    p_run.add_argument("--new-run", action="store_true", help="Start a fresh run")
    p_run.add_argument("--run", default=None, help="Resume a specific run_id")
    p_run.add_argument(
        "--force",
        action="store_true",
        help="Re-run every step even if already checkpointed",
    )
    p_run.add_argument("--prefer-hf", action="store_true")
    p_run.add_argument("--download-weights", action="store_true")
    p_run.add_argument(
        "--until",
        default=None,
        metavar="STEP",
        help="Stop after this step id (e.g. catalog, optimize, validate)",
    )
    p_run.add_argument(
        "--from-step",
        default=None,
        metavar="STEP",
        help="Start from this step id (skip earlier ones; they must already be done)",
    )
    p_run.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose per-step panels (still shows pipeline progress)",
    )
    p_run.add_argument("--no-explain", action="store_true")
    # --- threaded step flags (full parity with the step commands) ---
    # Global probe mode for the llama-capable steps (imatrix,
    # reference-logits, sensitivity). Freeze/export/validate have their own
    # mode flags below because their mode vocabularies differ.
    p_run.add_argument(
        "--mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
        help="Probe mode for imatrix / reference-logits / sensitivity "
        "(default: auto)",
    )
    p_run.add_argument(
        "--target-tokens",
        type=int,
        default=50_000,
        help="Corpus token budget (default 50000; use 300000+ for production)",
    )
    p_run.add_argument("--seed", type=int, default=42, help="Corpus split seed")
    p_run.add_argument(
        "--max-docs",
        type=int,
        default=32,
        help="Max calib docs for forward activation pass (default 32)",
    )
    p_run.add_argument(
        "--freeze-mode",
        choices=("auto", "hf-convert", "promote"),
        default="auto",
        help="freeze-gguf mode (default: auto)",
    )
    p_run.add_argument(
        "--convert-script",
        type=Path,
        default=None,
        help="Path to llama.cpp convert_hf_to_gguf.py (or set LLAMA_CPP_DIR)",
    )
    p_run.add_argument(
        "--require-bf16",
        action="store_true",
        help="Fail unless the frozen file is BF16/F16 (not Q8)",
    )
    p_run.add_argument(
        "--chunks",
        type=int,
        default=64,
        help="llama-imatrix --chunks (default 64; 0 = omit flag)",
    )
    p_run.add_argument(
        "--imatrix-args",
        default=None,
        metavar="ARGS",
        help='Extra args appended verbatim to llama-imatrix, e.g. --imatrix-args "-ngl 99"',
    )
    p_run.add_argument(
        "--perplexity-args",
        default=None,
        metavar="ARGS",
        help='Extra args appended verbatim to probe llama-perplexity runs, e.g. --perplexity-args "-ngl 99"',
    )
    p_run.add_argument(
        "--bands-per-role",
        type=int,
        default=3,
        help="Reband contiguous bands per role (default 3)",
    )
    p_run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel trial probes for sensitivity + optimize (default: 1)",
    )
    p_run.add_argument(
        "--probe-types",
        default=None,
        metavar="TYPES",
        help="Comma-separated sensitivity probe grid override",
    )
    p_run.add_argument(
        "--fixed-groups",
        default=None,
        metavar="GIDS",
        help="Comma-separated group ids kept at source precision (skipped at "
        "probe time; counted toward every budget at optimize time)",
    )
    p_run.add_argument(
        "--optimizer",
        choices=("greedy", "dp_mckp"),
        default="dp_mckp",
        help="Recipe optimizer (default: dp_mckp; greedy kept for A/B)",
    )
    p_run.add_argument(
        "--kld-objective",
        choices=("tail_1pct", "mean"),
        default="mean",
        help="Optimizer objective (default: mean, with automatic P99 guardrail)",
    )
    p_run.add_argument(
        "--certificate",
        choices=("bounded", "exhaustive"),
        default="bounded",
        help="bounded: prune via Lipschitz bound (default); exhaustive: probe all",
    )
    p_run.add_argument(
        "--lipschitz",
        type=float,
        default=None,
        help="Lipschitz bound L (default: auto-calibrated x2 margin)",
    )
    p_run.add_argument(
        "--splice-farm",
        type=Path,
        default=None,
        metavar="DIR",
        help="Precomputed per-tensor rung-shard farm for step-12 trials "
        "(symlink assembly instead of per-probe llama-quantize)",
    )
    p_run.add_argument(
        "--pareto-ratios",
        default=None,
        metavar="RATIOS",
        help="Comma-separated budget ratios (default: 0.55,0.65,0.72,0.80,0.90,1.0); "
        "points above the solve budget are dropped, the budget is a hard limit",
    )
    p_run.add_argument(
        "--budget-mb",
        type=float,
        default=None,
        help="Target size in MiB (default: from --quant profile ratio)",
    )
    p_run.add_argument(
        "--budget-ratio",
        type=float,
        default=None,
        help="Fraction of Q6_K baseline if --budget-mb omitted",
    )
    p_run.add_argument(
        "--export-mode",
        choices=("auto", "llama", "dry-run"),
        default="auto",
        help="export mode (default: auto)",
    )
    p_run.add_argument(
        "--base-type",
        default=None,
        help="Fallback/base type for llama-quantize (default: from --quant)",
    )
    p_run.add_argument(
        "--validate-mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
        help="validate mode (default: auto)",
    )
    p_run.add_argument(
        "--strict",
        action="store_true",
        help="Do not allow PROVISIONAL verdict without a real GGUF",
    )
    p_run.add_argument(
        "--only-quantizable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip non-quantizable tensors in weight features "
        "(default: True; --no-only-quantizable to include)",
    )

    # --- status / runs ---
    p_status = sub.add_parser("status", help="Show checkpoint status for a run")
    p_status.add_argument("--run", default=None, help="run_id (default: latest for --model)")
    p_status.add_argument("--model", "-m", default=None)

    p_runs = sub.add_parser("runs", help="List all checkpointed runs")

    p_formats = sub.add_parser(
        "formats",
        help="List supported target quant formats (q4_k_m, q5_k_m, …)",
    )
    p_formats.add_argument("--no-explain", action="store_true")

    # --- fit (hardware-aware pipeline: hardware in, budget out) ---
    p_fit = sub.add_parser(
        "fit",
        help="Run the pipeline with a budget derived from your hardware "
        "(--gpu 12GB / --ram 32GB / --device rtx-3060-12gb)",
    )
    p_fit.add_argument("--model", "-m", required=True)
    p_fit.add_argument("--gpu", default=None, metavar="SIZE", help="VRAM, e.g. 24GB")
    p_fit.add_argument("--ram", default=None, metavar="SIZE", help="System RAM, e.g. 32GB")
    p_fit.add_argument(
        "--device",
        default=None,
        metavar="ID",
        help="Named device profile (see: odg devices)",
    )
    p_fit.add_argument(
        "--cpu-only",
        action="store_true",
        help="Treat --ram as a plain CPU pool (no GPU/Metal)",
    )
    p_fit.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="Context length to size the KV cache for (default: min(4096, model ctx))",
    )
    p_fit.add_argument(
        "--kv-dtype",
        default="f16",
        choices=("f16", "f32", "q8_0"),
        help="KV cache dtype for sizing (default f16)",
    )
    p_fit.add_argument(
        "--quant",
        "-q",
        default=None,
        metavar="FORMAT",
        help="Quant family for probe types / base type (default q4_k_m); "
        "the size budget itself comes from your hardware",
    )
    p_fit.add_argument("--new-run", action="store_true")
    p_fit.add_argument("--run", default=None)
    p_fit.add_argument("--force", action="store_true")
    p_fit.add_argument("--prefer-hf", action="store_true")
    p_fit.add_argument("--download-weights", action="store_true")
    p_fit.add_argument(
        "--until",
        default=None,
        metavar="STEP",
        help="Stop after this step id (default: run through validate)",
    )
    p_fit.add_argument("--quiet", action="store_true")
    p_fit.add_argument("--no-explain", action="store_true")
    # --- threaded step flags (mirrors `odg run`; budget comes from hardware) ---
    p_fit.add_argument(
        "--mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
        help="Probe mode for imatrix / reference-logits / sensitivity",
    )
    p_fit.add_argument("--target-tokens", type=int, default=50_000)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--max-docs", type=int, default=32)
    p_fit.add_argument(
        "--freeze-mode",
        choices=("auto", "hf-convert", "promote"),
        default="auto",
    )
    p_fit.add_argument("--convert-script", type=Path, default=None)
    p_fit.add_argument("--require-bf16", action="store_true")
    p_fit.add_argument("--chunks", type=int, default=64)
    p_fit.add_argument("--imatrix-args", default=None, metavar="ARGS")
    p_fit.add_argument("--perplexity-args", default=None, metavar="ARGS")
    p_fit.add_argument("--bands-per-role", type=int, default=3)
    p_fit.add_argument("--jobs", type=int, default=1)
    p_fit.add_argument("--probe-types", default=None, metavar="TYPES")
    p_fit.add_argument("--fixed-groups", default=None, metavar="GIDS")
    p_fit.add_argument(
        "--optimizer", choices=("greedy", "dp_mckp"), default="dp_mckp"
    )
    p_fit.add_argument(
        "--kld-objective", choices=("tail_1pct", "mean"), default="mean"
    )
    p_fit.add_argument(
        "--certificate", choices=("bounded", "exhaustive"), default="bounded"
    )
    p_fit.add_argument("--lipschitz", type=float, default=None)
    p_fit.add_argument("--splice-farm", type=Path, default=None, metavar="DIR")
    p_fit.add_argument("--pareto-ratios", default=None, metavar="RATIOS")
    p_fit.add_argument(
        "--export-mode",
        choices=("auto", "llama", "dry-run"),
        default="auto",
    )
    p_fit.add_argument("--base-type", default=None)
    p_fit.add_argument(
        "--validate-mode",
        choices=("auto", "llama", "proxy"),
        default="auto",
    )
    p_fit.add_argument("--strict", action="store_true")
    p_fit.add_argument(
        "--only-quantizable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # --- devices (named hardware profiles for odg fit) ---
    p_dev = sub.add_parser(
        "devices",
        help="List named hardware profiles usable with odg fit --device",
    )
    p_dev.add_argument("--no-explain", action="store_true")

    # --- experiment (same model, same harness; only quantization changes) ---
    p_exp = sub.add_parser(
        "experiment",
        help="Same-model experiment: FunctionGemma 270M BF16 vs Q4_K_M / "
        "Q5_K_M / Q6_K / OpenDynamicGGUF on a pinned lm-eval suite",
    )
    p_exp.add_argument(
        "action",
        choices=("prepare", "run", "compare"),
        help="prepare GGUFs, run one/all variants, or write the comparison table",
    )
    p_exp.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Experiment directory (default: ./benchmark)",
    )
    p_exp.add_argument(
        "--model",
        "-m",
        default=None,
        help="HF repo (default: google/functiongemma-270m-it)",
    )
    p_exp.add_argument(
        "--variant",
        "-v",
        default=None,
        help="bf16 | q4_k_m | q5_k_m | q6_k | odg",
    )
    p_exp.add_argument("--all", action="store_true", help="Run every prepared variant")
    p_exp.add_argument("--suite", default="dev", help="dev (default, 32 samples) or paper")
    p_exp.add_argument("--limit", type=int, default=None, help="Override per-task sample cap")
    p_exp.add_argument(
        "--gguf-backend",
        choices=("gguf", "hf"),
        default=None,
        help="gguf (llama-cpp-python, default) or hf (gguf_file=)",
    )
    p_exp.add_argument("--skip-odg", action="store_true")
    p_exp.add_argument("--skip-quality", action="store_true")
    p_exp.add_argument("--skip-behavior", action="store_true")
    p_exp.add_argument("--force", action="store_true")
    p_exp.add_argument("--no-explain", action="store_true")

    # --- report (self-contained report.html from run artifacts) ---
    p_report = sub.add_parser(
        "report",
        help="Render report.html (allocations, sensitivity heatmap, Pareto, "
        "gates, experiment table) from run artifacts",
    )
    p_report.add_argument("--model", "-m", default=None)
    p_report.add_argument("--run", default=None)
    p_report.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <run>/report.html)",
    )
    p_report.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the report in a browser after writing it",
    )
    p_report.add_argument("--no-explain", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    if args.command == "resolve":
        return cmd_resolve(args)
    if args.command == "load":
        return cmd_load(args)
    if args.command == "enumerate":
        return cmd_enumerate(args)
    if args.command == "classify":
        return cmd_classify(args)
    if args.command == "catalog":
        return cmd_catalog(args)
    if args.command == "weight-features":
        return cmd_weight_features(args)
    if args.command == "corpus":
        return cmd_corpus(args)
    if args.command == "activation-features":
        return cmd_activation_features(args)
    if args.command == "freeze-gguf":
        return cmd_freeze_gguf(args)
    if args.command == "imatrix":
        return cmd_imatrix(args)
    if args.command == "reference-logits":
        return cmd_reference_logits(args)
    if args.command == "reband":
        return cmd_reband(args)
    if args.command == "sensitivity":
        return cmd_sensitivity(args)
    if args.command == "optimize":
        return cmd_optimize(args)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "runs":
        return cmd_runs(args)
    if args.command == "formats":
        return cmd_formats(args)
    if args.command == "fit":
        return cmd_fit(args)
    if args.command == "devices":
        return cmd_devices(args)
    if args.command == "experiment":
        return cmd_experiment(args)
    if args.command == "report":
        return cmd_report(args)

    parser.error(f"Unknown command {args.command}")
    return 2


def _store(args: argparse.Namespace):
    from store import RunStore

    return RunStore(args.artifacts)


def _apply_quant_format(store, meta, quant: str | None, *, explain: bool = True):
    """Resolve QuantFormat for a run; optionally override via CLI --quant."""
    from quant_formats import get_format

    if quant:
        fmt = get_format(quant)
        store.set_quant_format(
            meta.run_id, quant_format=fmt.id, quant_label=fmt.label
        )
        meta = store.load_run(meta.run_id)
    elif meta.quant_format:
        fmt = get_format(meta.quant_format)
    else:
        fmt = get_format(None)
        store.set_quant_format(
            meta.run_id, quant_format=fmt.id, quant_label=fmt.label
        )
        meta = store.load_run(meta.run_id)
    if explain:
        ui.show_quant_choice(fmt, explain=explain)
    return meta, fmt


def cmd_formats(args: argparse.Namespace) -> int:
    from quant_formats import list_formats_rows

    print_explain = not args.no_explain
    ui.formats_table(list_formats_rows(), explain=print_explain)
    if print_explain:
        ui.info(
            "One command:     odg run --model <ref> --quant q5_k_m --no-ask\n"
            "Or step-by-step: odg resolve --model <ref>   (asks for format in a TTY)"
        )
    return 0


def _run_flag(args: argparse.Namespace, name: str, default=None):
    """getattr with default — pipeline Namespaces built by hand may lack new flags."""
    return getattr(args, name, default)


def _pipeline_fmt(store, args):
    """Best-effort quant format for pipeline input comparison.

    Returns None when no run exists yet (fresh pipeline) or the id is invalid;
    callers then simply omit format-derived keys from the comparison.
    """
    try:
        from quant_formats import get_format
    except ImportError:
        return None
    if _run_flag(args, "quant", None):
        try:
            return get_format(args.quant)
        except ValueError:
            return None
    run_id = _run_flag(args, "run", None)
    try:
        if run_id:
            meta = store.load_run(run_id)
        else:
            meta = store.latest_run_for_model(args.model)
    except (ValueError, FileNotFoundError, OSError):
        return None
    if meta is None or not meta.quant_format:
        return None
    try:
        return get_format(meta.quant_format)
    except ValueError:
        return None


def _pipeline_expected_inputs(step_id: str, args: argparse.Namespace, fmt):
    """Result-affecting pipeline inputs for *step_id*, keyed like input.json.

    Only keys the pipeline (`odg run` / `odg fit`) controls are included;
    `jobs` is deliberately excluded (parallelism changes no numbers).
    Keys requiring a resolved quant format are omitted when *fmt* is None.
    """
    exp: dict = {}
    if step_id == "corpus":
        exp["target_tokens"] = int(_run_flag(args, "target_tokens", 50_000) or 50_000)
        seed = _run_flag(args, "seed", 42)
        exp["seed"] = int(42 if seed is None else seed)
    elif step_id == "activation_features":
        exp["mode"] = _run_flag(args, "mode", "auto") or "auto"
        exp["max_docs"] = int(_run_flag(args, "max_docs", 32) or 32)
    elif step_id == "freeze_gguf":
        exp["mode"] = _run_flag(args, "freeze_mode", "auto") or "auto"
        exp["require_bf16"] = bool(_run_flag(args, "require_bf16", False))
        cs = _run_flag(args, "convert_script", None)
        exp["convert_script"] = str(cs) if cs else None
    elif step_id == "imatrix":
        exp["mode"] = _run_flag(args, "mode", "auto") or "auto"
        chunks = int(_run_flag(args, "chunks", 64) or 0)
        exp["chunks"] = None if chunks <= 0 else chunks
        exp["imatrix_args"] = split_extra_args(
            _run_flag(args, "imatrix_args", None), "--imatrix-args"
        )
    elif step_id == "reband":
        exp["bands_per_role"] = int(_run_flag(args, "bands_per_role", 3) or 3)
    elif step_id == "reference_logits":
        exp["mode"] = _run_flag(args, "mode", "auto") or "auto"
        exp["perplexity_args"] = split_extra_args(
            _run_flag(args, "perplexity_args", None), "--perplexity-args"
        )
    elif step_id == "sensitivity":
        exp["mode"] = _run_flag(args, "mode", "auto") or "auto"
        pt = _run_flag(args, "probe_types", None)
        if pt:
            normed = [t.strip().upper() for t in str(pt).split(",") if t.strip()]
            if normed:
                exp["probe_types"] = normed
        exp["perplexity_args"] = split_extra_args(
            _run_flag(args, "perplexity_args", None), "--perplexity-args"
        )
        exp["fixed_groups"] = sorted(
            g.strip()
            for g in str(_run_flag(args, "fixed_groups", None) or "").split(",")
            if g.strip()
        )
        exp["kld_objective"] = _run_flag(args, "kld_objective", "mean")
        exp["certificate"] = _run_flag(args, "certificate", "bounded")
        exp["lipschitz"] = _run_flag(args, "lipschitz", None)
        _sf = _run_flag(args, "splice_farm", None)
        exp["splice_farm"] = str(_sf) if _sf is not None else None
        _sbm = _run_flag(args, "budget_mb", None)
        exp["budget_mb"] = float(_sbm) if _sbm is not None else None
        _sbr = _run_flag(args, "budget_ratio", None)
        if _sbr is not None:
            exp["budget_ratio"] = float(_sbr)
        elif fmt is not None:
            exp["budget_ratio"] = float(fmt.budget_ratio)
        if fmt is not None:
            exp["baseline"] = fmt.baseline_type
            exp["quant_format"] = fmt.id
    elif step_id == "optimize":
        bm = _run_flag(args, "budget_mb", None)
        exp["budget_mb"] = float(bm) if bm is not None else None
        br = _run_flag(args, "budget_ratio", None)
        if br is not None:
            exp["budget_ratio"] = float(br)
        elif fmt is not None:
            exp["budget_ratio"] = float(fmt.budget_ratio)
        if fmt is not None:
            exp["quant_format"] = fmt.id
        exp["optimizer"] = _run_flag(args, "optimizer", "dp_mckp")
        exp["kld_objective"] = _run_flag(args, "kld_objective", "mean")
        exp["certificate"] = _run_flag(args, "certificate", "bounded")
        exp["lipschitz"] = _run_flag(args, "lipschitz", None)
        exp["pareto_ratios"] = _run_flag(args, "pareto_ratios", None)
        exp["fixed_groups"] = sorted(
            g.strip()
            for g in str(_run_flag(args, "fixed_groups", None) or "").split(",")
            if g.strip()
        )
    elif step_id == "export":
        exp["mode"] = _run_flag(args, "export_mode", "auto") or "auto"
        bt = _run_flag(args, "base_type", None)
        if bt:
            exp["base_type"] = bt
        elif fmt is not None:
            exp["base_type"] = fmt.base_type
        if fmt is not None:
            exp["quant_format"] = fmt.id
    elif step_id == "validate":
        exp["mode"] = _run_flag(args, "validate_mode", "auto") or "auto"
        exp["strict"] = bool(_run_flag(args, "strict", False))
        exp["perplexity_args"] = split_extra_args(
            _run_flag(args, "perplexity_args", None), "--perplexity-args"
        )
    return exp


def _check_stale_step(store, run_id: str, step_id: str, expected: dict,
                       *, interactive: bool) -> bool:
    """Warn-and-confirm when pipeline flags disagree with a done step.

    Compares only keys present in the recorded input.json (older checkpoints
    predate some keys and must not false-positive). Returns True when the
    caller should force re-run this step; False means proceed normally (the
    step fn will skip itself when done). Non-interactive sessions warn and
    keep the checkpoint.
    """
    if not expected:
        return False
    try:
        if not store.is_step_done(run_id, step_id):
            return False
    except (OSError, ValueError, FileNotFoundError):
        return False
    inp = store.step_path(run_id, step_id) / "input.json"
    try:
        recorded = json.loads(inp.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(recorded, dict):
        return False
    diffs = [
        (k, recorded[k], v)
        for k, v in expected.items()
        if k in recorded and recorded[k] != v
    ]
    if not diffs:
        return False
    lines = "\n".join(
        f"  {k}: checkpoint has {old!r}, pipeline wants {new!r}"
        for k, old, new in diffs
    )
    ui.warn(
        f"Step '{step_id}' is already done but pipeline flags differ:\n{lines}"
    )
    if not interactive:
        ui.warn(
            f"Keeping the checkpointed '{step_id}' output. "
            "Pass --force to re-run it with the new flags."
        )
        return False
    try:
        answer = input(f"Re-run '{step_id}' with the new flags? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def cmd_run(args: argparse.Namespace) -> int:
    """Run steps 01–15 in order with one CLI invocation."""
    from steps import STEPS, STEPS_BY_ID

    print_explain = not args.no_explain and not args.quiet
    quiet_panels = bool(args.quiet) or bool(args.no_explain)

    pipeline = [
        ("resolve", cmd_resolve),
        ("load", cmd_load),
        ("enumerate", cmd_enumerate),
        ("classify", cmd_classify),
        ("catalog", cmd_catalog),
        ("weight_features", cmd_weight_features),
        ("corpus", cmd_corpus),
        ("activation_features", cmd_activation_features),
        ("freeze_gguf", cmd_freeze_gguf),
        ("imatrix", cmd_imatrix),
        ("reband", cmd_reband),
        ("reference_logits", cmd_reference_logits),
        ("sensitivity", cmd_sensitivity),
        ("optimize", cmd_optimize),
        ("export", cmd_export),
        ("validate", cmd_validate),
    ]

    # CLI command names differ slightly from step ids
    step_cli = {
        "weight_features": "weight-features",
        "activation_features": "activation-features",
        "freeze_gguf": "freeze-gguf",
        "reference_logits": "reference-logits",
    }

    start_id = args.from_step.replace("-", "_") if args.from_step else pipeline[0][0]
    until_id = args.until.replace("-", "_") if args.until else pipeline[-1][0]
    if start_id not in STEPS_BY_ID:
        print(f"ERROR: unknown --from-step {args.from_step!r}", file=sys.stderr)
        return 1
    if until_id not in STEPS_BY_ID:
        print(f"ERROR: unknown --until {args.until!r}", file=sys.stderr)
        return 1

    ids = [s for s, _ in pipeline]
    try:
        i0 = ids.index(start_id)
        i1 = ids.index(until_id)
    except ValueError:
        print("ERROR: step not in pipeline", file=sys.stderr)
        return 1
    if i0 > i1:
        print("ERROR: --from-step is after --until", file=sys.stderr)
        return 1
    selected = pipeline[i0 : i1 + 1]

    ui.step_banner(
        0,
        "Full pipeline",
        model=args.model,
        run_id=args.run or "(auto)",
        root=str(getattr(args, "artifacts", None) or Path.cwd() / "artifacts"),
        goal="Run selected steps end-to-end (checkpoints resume automatically).",
        bullets=[
            f"Steps: {selected[0][0]} → {selected[-1][0]} ({len(selected)} steps)",
            f"Quant: {args.quant or '(ask / default q4_k_m)'}",
            "Already-done steps are skipped unless --force",
        ],
        explain=not args.no_explain,
    )

    results: list[tuple[str, str]] = []  # (step_id, status)

    # Stale-checkpoint support: compare pipeline flags against each done
    # step's recorded input.json (warn-and-confirm on mismatch).
    store = _store(args)
    fmt = _pipeline_fmt(store, args)
    interactive = sys.stdin.isatty() and not bool(_run_flag(args, "no_ask", False))

    for idx, (step_id, fn) in enumerate(selected, 1):
        title = STEPS_BY_ID[step_id].title
        ui.info(
            f"[{idx}/{len(selected)}] {step_cli.get(step_id, step_id)} — {title}",
            explain=not args.no_explain,
        )

        # Per-step mode: freeze/export/validate have their own vocabularies.
        if step_id == "freeze_gguf":
            step_mode = _run_flag(args, "freeze_mode", "auto") or "auto"
        elif step_id == "export":
            step_mode = _run_flag(args, "export_mode", "auto") or "auto"
        elif step_id == "validate":
            step_mode = _run_flag(args, "validate_mode", "auto") or "auto"
        else:
            step_mode = _run_flag(args, "mode", "auto") or "auto"

        step_args = argparse.Namespace(
            command=step_cli.get(step_id, step_id.replace("_", "-")),
            artifacts=args.artifacts,
            model=args.model,
            run=args.run,
            force=bool(args.force),
            no_explain=quiet_panels,
            # resolve
            quant=args.quant,
            no_ask=bool(args.no_ask),
            new_run=bool(args.new_run) if step_id == "resolve" else False,
            prefer_hf=bool(args.prefer_hf),
            download_weights=bool(args.download_weights),
            cache_dir=None,
            out=None,
            # weight features
            only_quantizable=bool(_run_flag(args, "only_quantizable", True)),
            # corpus
            target_tokens=int(_run_flag(args, "target_tokens", 50_000) or 50_000),
            seed=int(_run_flag(args, "seed", 42)
                     if _run_flag(args, "seed", 42) is not None else 42),
            # shared mode knobs
            mode=step_mode,
            max_docs=int(_run_flag(args, "max_docs", 32) or 32),
            bands_per_role=int(_run_flag(args, "bands_per_role", 3) or 3),
            convert_script=_run_flag(args, "convert_script", None),
            require_bf16=bool(_run_flag(args, "require_bf16", False)),
            llama_imatrix=None,
            chunks=int(_run_flag(args, "chunks", 64)
                       if _run_flag(args, "chunks", 64) is not None else 64),
            imatrix_args=_run_flag(args, "imatrix_args", None),
            llama_perplexity=None,
            baseline=None,
            perplexity_args=_run_flag(args, "perplexity_args", None),
            probe_types=_run_flag(args, "probe_types", None),
            fixed_groups=_run_flag(args, "fixed_groups", None),
            # threaded through by odg fit (hardware-derived budget)
            budget_mb=_run_flag(args, "budget_mb", None),
            budget_ratio=_run_flag(args, "budget_ratio", None),
            optimizer=_run_flag(args, "optimizer", "dp_mckp") or "dp_mckp",
            kld_objective=_run_flag(args, "kld_objective", "mean") or "mean",
            certificate=_run_flag(args, "certificate", "bounded") or "bounded",
            lipschitz=_run_flag(args, "lipschitz", None),
            splice_farm=_run_flag(args, "splice_farm", None),
            jobs=int(_run_flag(args, "jobs", 1) or 1),
            pareto_ratios=_run_flag(args, "pareto_ratios", None),
            base_type=_run_flag(args, "base_type", None),
            llama_quantize=None,
            strict=bool(_run_flag(args, "strict", False)),
        )

        # Warn-and-confirm: a done step whose recorded inputs disagree with
        # the pipeline flags keeps its checkpoint unless the user confirms
        # a re-run (or passed --force). Non-interactive sessions warn and keep.
        if not step_args.force:
            check_run = _run_flag(args, "run", None)
            if check_run is None:
                latest = store.latest_run_for_model(args.model)
                check_run = latest.run_id if latest is not None else None
            if check_run is not None:
                try:
                    expected = _pipeline_expected_inputs(step_id, args, fmt)
                except ValueError as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 1
                if _check_stale_step(
                    store, check_run, step_id, expected,
                    interactive=interactive,
                ):
                    step_args.force = True

        try:
            code = fn(step_args)
        except Exception as exc:  # noqa: BLE001
            ui.error(STEPS_BY_ID[step_id].number, step_id, exc, "(pipeline)")
            results.append((step_id, "failed"))
            ui.pipeline_summary(results, explain=not args.no_explain)
            return 1

        if code != 0:
            results.append((step_id, "failed"))
            ui.pipeline_summary(results, explain=not args.no_explain)
            ui.warn(
                f"Pipeline stopped at {step_id} (exit {code}). "
                f"Fix the issue, then: odg run -m {args.model} --from-step {step_id}",
                explain=not args.no_explain,
            )
            return code

        results.append((step_id, "done"))
        # After resolve, pin run id so later steps hit the same run
        if step_id == "resolve" and not args.run:
            store = _store(args)
            meta = store.latest_run_for_model(args.model)
            if meta is not None:
                args.run = meta.run_id
        # The resolve step may have created the run or set its quant format;
        # refresh both the store handle inputs and the format used for the
        # stale-check comparison below.
        if step_id == "resolve":
            fmt = _pipeline_fmt(store, args)

    ui.pipeline_summary(results, explain=not args.no_explain)

    # Auto-render report.html when the pipeline reached validate (best-effort)
    report_path = None
    if any(sid == "validate" and st == "done" for sid, st in results):
        try:
            from report import generate_report

            store = _store(args)
            meta = _require_run(store, model=args.model, run_id=args.run)
            report_path = generate_report(Path(meta.root)).report_path
        except Exception:  # noqa: BLE001 — report must never fail the pipeline
            report_path = None

    if not args.no_explain:
        lines = [
            f"Pipeline finished. Inspect: odg status --model {args.model}",
            "Report card lives under steps/15_validate/ when validate ran.",
        ]
        if report_path:
            lines.append(f"Full report: {report_path}")
        ui.next_step("\n".join(lines))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from quant_formats import get_format
    from resolve import resolve_model
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    meta = store.get_or_create_run(
        args.model,
        run_id=args.run,
        resume=not args.new_run,
    )

    # --- choose target quant format (initial user decision) ---
    try:
        if args.quant:
            fmt = get_format(args.quant)
        elif meta.quant_format and not args.new_run:
            fmt = get_format(meta.quant_format)
        elif args.no_ask or not sys.stdin.isatty():
            fmt = get_format(None)
        else:
            chosen = ui.prompt_quant_format(
                default_id="q4_k_m",
                existing_id=meta.quant_format,
            )
            fmt = get_format(chosen)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    store.set_quant_format(
        meta.run_id, quant_format=fmt.id, quant_label=fmt.label
    )
    meta = store.load_run(meta.run_id)

    if print_explain:
        _banner(args.model, meta.run_id, meta.root)
        ui.show_quant_choice(fmt, explain=True)

    # Skip if already done
    if store.is_step_done(meta.run_id, "resolve") and not args.force:
        out = store.read_step_output(meta.run_id, "resolve")
        if out is not None:
            out = {**out, "quant_format": fmt.id, "quant_label": fmt.label}
        ui.already_done(
            1,
            "resolve",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "resolve"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    input_data = {
        "model": args.model,
        "prefer_hf": args.prefer_hf,
        "download_weights": args.download_weights,
        "quant_format": fmt.id,
        "quant_label": fmt.label,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "resolve", input_data, force=args.force
        )
    except StepAlreadyDone:
        # race / edge — treat as skip
        return cmd_resolve(args)

    try:
        with ui.working('Resolving model…', explain=print_explain):
            result = resolve_model(
                args.model,
                cache_dir=args.cache_dir,
                download_weights=args.download_weights,
                prefer_hf=args.prefer_hf,
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "resolve", str(exc))
        ui.error(1, "resolve", exc, store.step_path(meta.run_id, "resolve"))
        return 1

    payload = result.to_dict()
    payload["quant_format"] = fmt.id
    payload["quant_label"] = fmt.label
    payload["quant_base_type"] = fmt.base_type
    payload["quant_budget_ratio"] = fmt.budget_ratio
    log_text = "\n".join(result.steps_log) + "\n"
    log_text += f"quant_format={fmt.id} ({fmt.label})\n"
    store.complete_step(
        meta.run_id,
        "resolve",
        payload,
        log_text=log_text,
    )

    if print_explain:
        _explain(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=tuple(x.strip() for x in 'input.json, output.json, status.json, log.txt'.split(',')),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))

    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        if print_explain:
            ui.info(f"Also wrote copy → {args.out}")

    if args.no_explain:
        print(text)
    else:
        ui.json_panel(payload, title="resolve result")

    if not result.hf_repo_id and not result.local_path:
        return 2
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    from load import load_model
    from load import tensor_index_from_resolve
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "resolve"):
        print(
            "ERROR: Step 01 (resolve) is not done for this run.\n"
            f"  Run: odg resolve --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    resolve_out = store.read_step_output(meta.run_id, "resolve")
    if not resolve_out:
        print("ERROR: missing resolve output.json", file=sys.stderr)
        return 1

    if print_explain:
        _banner_load(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "load") and not args.force:
        out = store.read_step_output(meta.run_id, "load")
        ui.already_done(
            2,
            "load",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "load"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    input_data = {
        "from_step": "resolve",
        "local_path": resolve_out.get("local_path"),
        "source_is_quantized": resolve_out.get("source_is_quantized"),
    }

    try:
        step_dir = store.begin_step(meta.run_id, "load", input_data, force=args.force)
    except StepAlreadyDone:
        return cmd_load(args)

    try:
        with ui.working('Loading model / parsing GGUF…', explain=print_explain):
            loaded = load_model(resolve_out)
            # Full tensor index for Step 03 (separate artifact — can be large)
            tensors = tensor_index_from_resolve(resolve_out)
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "load", str(exc))
        ui.error(2, "load", exc, store.step_path(meta.run_id, "load"))
        return 1

    payload = loaded.to_dict()
    log_text = "\n".join(loaded.steps_log) + "\n"
    store.complete_step(
        meta.run_id,
        "load",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensor_index.json": json.dumps(tensors, indent=2).encode("utf-8") + b"\n",
        },
    )

    if print_explain:
        _explain_load(loaded)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=tuple(x.strip() for x in 'input.json, output.json, status.json, log.txt, tensor_index.json'.split(',')),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="load result")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_enumerate(args: argparse.Namespace) -> int:
    from enumerate import enumerate_tensors
    from enumerate import to_tsv
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "load"):
        print(
            "ERROR: Step 02 (load) is not done.\n"
            f"  Run: odg load --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_enumerate(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "enumerate") and not args.force:
        out = store.read_step_output(meta.run_id, "enumerate")
        ui.already_done(
            3,
            "enumerate",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "enumerate"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    index_path = store.step_path(meta.run_id, "load") / "tensor_index.json"
    if not index_path.is_file():
        print(f"ERROR: missing {index_path}", file=sys.stderr)
        return 1
    tensor_index = json.loads(index_path.read_text())

    input_data = {
        "from_step": "load",
        "tensor_index_path": str(index_path),
        "n_tensors_in": len(tensor_index),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "enumerate", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_enumerate(args)

    try:
        with ui.working('Enumerating tensors…', explain=print_explain):
            result = enumerate_tensors(tensor_index)
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "enumerate", str(exc))
        ui.error(3, "enumerate", exc, store.step_path(meta.run_id, "enumerate"))
        return 1

    payload = result.summary_dict()
    full = result.to_dict()
    tsv = to_tsv(result.tensors)
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "enumerate",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensors.json": json.dumps(full, indent=2).encode("utf-8") + b"\n",
            "tensors.tsv": tsv.encode("utf-8"),
        },
    )

    if print_explain:
        _explain_enumerate(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=tuple(x.strip() for x in 'output.json, tensors.json, tensors.tsv, status.json, log.txt'.split(',')),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="enumerate summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    from classify import classify_tensors
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "enumerate"):
        print(
            "ERROR: Step 03 (enumerate) is not done.\n"
            f"  Run: odg enumerate --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_classify(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "classify") and not args.force:
        out = store.read_step_output(meta.run_id, "classify")
        ui.already_done(
            4,
            "classify",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "classify"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    tensors_path = store.step_path(meta.run_id, "enumerate") / "tensors.json"
    if not tensors_path.is_file():
        print(f"ERROR: missing {tensors_path}", file=sys.stderr)
        return 1
    enum_full = json.loads(tensors_path.read_text())
    tensors = enum_full.get("tensors") or []

    load_out = store.read_step_output(meta.run_id, "load") or {}
    n_layers = load_out.get("layer_count")

    input_data = {
        "from_step": "enumerate",
        "tensors_path": str(tensors_path),
        "n_tensors_in": len(tensors),
        "n_layers": n_layers,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "classify", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_classify(args)

    try:
        with ui.working('Classifying tensors…', explain=print_explain):
            result = classify_tensors(tensors, n_layers=n_layers)
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "classify", str(exc))
        ui.error(4, "classify", exc, store.step_path(meta.run_id, "classify"))
        return 1

    payload = result.summary_dict()
    full = result.to_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    # TSV for browsing
    tsv_lines = [
        "index\tname\trole\tdepth\tlayer\tgroup_id\tquantizable\tdtype\tshape\tnbytes"
    ]
    for t in result.tensors:
        shape = "x".join(str(d) for d in t.shape)
        tsv_lines.append(
            f"{t.index}\t{t.name}\t{t.role}\t{t.depth or ''}\t"
            f"{'' if t.layer is None else t.layer}\t{t.group_id}\t"
            f"{int(t.quantizable)}\t{t.dtype}\t{shape}\t{t.nbytes}"
        )
    tsv = "\n".join(tsv_lines) + "\n"

    store.complete_step(
        meta.run_id,
        "classify",
        payload,
        log_text=log_text,
        extra_artifacts={
            "classified.json": json.dumps(full, indent=2).encode("utf-8") + b"\n",
            "classified.tsv": tsv.encode("utf-8"),
        },
    )

    if print_explain:
        _explain_classify(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'classified.json', 'classified.tsv', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="classify summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    import hashlib

    from catalog import build_catalog
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "classify"):
        print(
            "ERROR: Step 04 (classify) is not done.\n"
            f"  Run: odg classify --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_catalog(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "catalog") and not args.force:
        out = store.read_step_output(meta.run_id, "catalog")
        ui.already_done(
            5,
            "catalog",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "catalog"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    classified_path = store.step_path(meta.run_id, "classify") / "classified.json"
    if not classified_path.is_file():
        print(f"ERROR: missing {classified_path}", file=sys.stderr)
        return 1
    classified = json.loads(classified_path.read_text())
    tensors = classified.get("tensors") or []

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    load_out = store.read_step_output(meta.run_id, "load") or {}

    source_path = load_out.get("source_path") or resolve_out.get("local_path")
    source_sha = resolve_out.get("source_sha256")
    if source_path and not source_sha:
        try:
            h = hashlib.sha256()
            with open(source_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            source_sha = h.hexdigest()
        except OSError:
            source_sha = None

    input_data = {
        "from_step": "classify",
        "classified_path": str(classified_path),
        "n_tensors_in": len(tensors),
        "source_path": source_path,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "catalog", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_catalog(args)

    try:
        with ui.working('Building tensor catalog…', explain=print_explain):
            catalog = build_catalog(
                tensors,
                model_ref=meta.model_ref,
                hf_repo_id=resolve_out.get("hf_repo_id"),
                source_path=source_path,
                source_backend=load_out.get("backend"),
                source_is_quantized=bool(
                    load_out.get("source_is_quantized")
                    or resolve_out.get("source_is_quantized")
                ),
                source_sha256=source_sha,
                n_layers=load_out.get("layer_count") or classified.get("n_layers"),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "catalog", str(exc))
        ui.error(5, "catalog", exc, store.step_path(meta.run_id, "catalog"))
        return 1

    payload = catalog.summary_dict()
    full = catalog.to_dict()
    log_text = "\n".join(catalog.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "catalog",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(full, indent=2).encode("utf-8") + b"\n",
        },
    )

    if print_explain:
        _explain_catalog(catalog)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=tuple(x.strip() for x in 'output.json, tensor_catalog.json, status.json, log.txt'.split(',')),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="catalog summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_weight_features(args: argparse.Namespace) -> int:
    from store import StepAlreadyDone
    from weight_features import compute_catalog_weight_features

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "catalog"):
        print(
            "ERROR: Step 05 (catalog) is not done.\n"
            f"  Run: odg catalog --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_weight_features(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "weight_features") and not args.force:
        out = store.read_step_output(meta.run_id, "weight_features")
        ui.already_done(
            6,
            "weight_features",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "weight_features"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    catalog_path = store.step_path(meta.run_id, "catalog") / "tensor_catalog.json"
    if not catalog_path.is_file():
        print(f"ERROR: missing {catalog_path}", file=sys.stderr)
        return 1
    catalog = json.loads(catalog_path.read_text())
    source_path = catalog.get("source_path")
    load_out = store.read_step_output(meta.run_id, "load") or {}
    if not source_path:
        source_path = load_out.get("source_path")

    input_data = {
        "from_step": "catalog",
        "catalog_path": str(catalog_path),
        "source_path": source_path,
        "only_quantizable": bool(args.only_quantizable),
        "n_tensors_in": len(catalog.get("tensors") or {}),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "weight_features", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_weight_features(args)

    try:
        with ui.working("Computing weight features…", explain=print_explain):
            updated, result = compute_catalog_weight_features(
                catalog,
                source_path=source_path,
                only_quantizable=bool(args.only_quantizable),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "weight_features", str(exc))
        ui.error(6, "weight_features", exc, store.step_path(meta.run_id, "weight_features"))
        return 1

    payload = result.summary_dict()
    # Keep payload JSON-friendly / not huge: drop full group_features map from summary
    # (still written to weight_features.json + updated catalog)
    payload_out = {
        "model_ref": result.model_ref,
        "source_path": result.source_path,
        "source_is_quantized": result.source_is_quantized,
        "n_tensors": result.n_tensors,
        "n_with_features": result.n_with_features,
        "n_skipped": result.n_skipped,
        "catalog_sha256": result.catalog_sha256,
        "hardest_groups": result.hardest_groups,
        "easiest_groups": result.easiest_groups,
        "steps_log": result.steps_log,
        "notes": result.notes,
    }
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "weight_features",
        payload_out,
        log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(updated, indent=2).encode("utf-8")
            + b"\n",
            "group_features.json": json.dumps(
                result.group_features, indent=2
            ).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_weight_features(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'tensor_catalog.json', 'group_features.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload_out, title="weight-features summary")
    else:
        print(json.dumps(payload_out, indent=2))

    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    from corpus import build_corpus
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Corpus needs resolve descriptor; weight_features is the logical prior step
    if not store.is_step_done(meta.run_id, "weight_features"):
        print(
            "ERROR: Step 06 (weight-features) is not done.\n"
            f"  Run: odg weight-features --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_corpus(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "corpus") and not args.force:
        out = store.read_step_output(meta.run_id, "corpus")
        ui.already_done(
            7,
            "corpus",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "corpus"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    desc = resolve_out.get("descriptor") or {}
    chat_template = desc.get("chat_template") or "gemma3"
    specialty = desc.get("specialty_domain")

    input_data = {
        "from_step": "weight_features",
        "chat_template": chat_template,
        "specialty_domain": specialty,
        "target_tokens": int(args.target_tokens),
        "seed": int(args.seed),
        "splits": {"calib": 0.6, "search": 0.2, "heldout": 0.2},
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "corpus", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_corpus(args)

    try:
        with ui.working('Building calibration corpus…', explain=print_explain):
            result, manifest = build_corpus(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                chat_template=chat_template,
                specialty_domain=specialty,
                target_tokens=int(args.target_tokens),
                seed=int(args.seed),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "corpus", str(exc))
        ui.error(7, "corpus", exc, store.step_path(meta.run_id, "corpus"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    # Files already written into step_dir by build_corpus; also store manifest
    store.complete_step(
        meta.run_id,
        "corpus",
        payload,
        log_text=log_text,
        extra_artifacts={
            "corpus_manifest.json": json.dumps(manifest, indent=2).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_corpus(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'calib.txt', 'search.txt', 'heldout.txt', 'corpus_manifest.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="corpus summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_activation_features(args: argparse.Namespace) -> int:
    from activation_features import compute_catalog_activation_features
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "corpus"):
        print(
            "ERROR: Step 07 (corpus) is not done.\n"
            f"  Run: odg corpus --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1
    if not store.is_step_done(meta.run_id, "weight_features"):
        print(
            "ERROR: Step 06 (weight-features) is not done.\n"
            f"  Run: odg weight-features --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_activation_features(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "activation_features") and not args.force:
        out = store.read_step_output(meta.run_id, "activation_features")
        ui.already_done(
            8,
            "activation_features",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "activation_features"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    catalog_path = (
        store.step_path(meta.run_id, "weight_features") / "tensor_catalog.json"
    )
    if not catalog_path.is_file():
        catalog_path = store.step_path(meta.run_id, "catalog") / "tensor_catalog.json"
    if not catalog_path.is_file():
        print(f"ERROR: missing catalog at {catalog_path}", file=sys.stderr)
        return 1

    calib_path = store.step_path(meta.run_id, "corpus") / "calib.txt"
    if not calib_path.is_file():
        print(f"ERROR: missing {calib_path}", file=sys.stderr)
        return 1

    catalog = json.loads(catalog_path.read_text())
    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    corpus_out = store.read_step_output(meta.run_id, "corpus") or {}

    # Prefer local HF dir if resolve downloaded BF16; else hub id
    hf_local = None
    local_path = resolve_out.get("local_path")
    if local_path and Path(local_path).is_dir():
        hf_local = local_path
    hf_id = resolve_out.get("hf_repo_id")
    # Don't treat Ollama blob file as HF
    if local_path and Path(local_path).is_file():
        hf_local = None

    input_data = {
        "from_steps": ["weight_features", "corpus"],
        "catalog_path": str(catalog_path),
        "calib_path": str(calib_path),
        "mode": args.mode,
        "max_docs": int(args.max_docs),
        "hf_repo_id": hf_id,
        "hf_local_path": hf_local,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "activation_features", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_activation_features(args)

    try:
        with ui.working('Computing activation features…', explain=print_explain):
            updated, result = compute_catalog_activation_features(
                catalog,
                calib_path=calib_path,
                mode=args.mode,
                hf_model_id=hf_id,
                hf_local_path=hf_local,
                max_forward_docs=int(args.max_docs),
                corpus_domain_counts=corpus_out.get("domain_counts"),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "activation_features", str(exc))
        ui.error(8, "activation_features", exc, store.step_path(meta.run_id, "activation_features"))
        return 1

    payload = {
        "model_ref": result.model_ref,
        "method": result.method,
        "calib_path": result.calib_path,
        "n_docs_used": result.n_docs_used,
        "n_tokens_est": result.n_tokens_est,
        "n_tensors": result.n_tensors,
        "n_with_features": result.n_with_features,
        "catalog_sha256": result.catalog_sha256,
        "hardest_groups": result.hardest_groups,
        "easiest_groups": result.easiest_groups,
        "steps_log": result.steps_log,
        "notes": result.notes,
    }
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "activation_features",
        payload,
        log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(updated, indent=2).encode("utf-8")
            + b"\n",
            "activation_features.json": json.dumps(
                {
                    "method": result.method,
                    "hardest_groups": result.hardest_groups,
                    "easiest_groups": result.easiest_groups,
                    "group_activation_features": updated.get(
                        "group_activation_features"
                    ),
                },
                indent=2,
            ).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_activation_features(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'tensor_catalog.json', 'activation_features.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="activation-features summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_freeze_gguf(args: argparse.Namespace) -> int:
    from freeze import freeze_gguf
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "activation_features"):
        print(
            "ERROR: Step 08 (activation-features) is not done.\n"
            f"  Run: odg activation-features --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_freeze_gguf(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "freeze_gguf") and not args.force:
        out = store.read_step_output(meta.run_id, "freeze_gguf")
        ui.already_done(
            9,
            "freeze_gguf",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "freeze_gguf"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    load_out = store.read_step_output(meta.run_id, "load") or {}

    source_path = load_out.get("source_path") or resolve_out.get("local_path")
    source_is_quantized = bool(
        load_out.get("source_is_quantized") or resolve_out.get("source_is_quantized")
    )

    hf_local = None
    local_path = resolve_out.get("local_path")
    if local_path and Path(local_path).is_dir():
        hf_local = local_path

    # Catalog names for verification (prefer activation-features catalog)
    catalog_names: list[str] = []
    for step_id in ("activation_features", "weight_features", "catalog"):
        cpath = store.step_path(meta.run_id, step_id) / "tensor_catalog.json"
        if cpath.is_file():
            cat = json.loads(cpath.read_text())
            catalog_names = list((cat.get("tensors") or {}).keys())
            break

    input_data = {
        "from_step": "activation_features",
        "mode": args.mode,
        "source_path": source_path,
        "hf_local_path": hf_local,
        "require_bf16": bool(args.require_bf16),
        "convert_script": str(args.convert_script) if args.convert_script else None,
        "n_catalog_tensors": len(catalog_names),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "freeze_gguf", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_freeze_gguf(args)

    try:
        with ui.working('Freezing BF16/reference GGUF…', explain=print_explain):
            result = freeze_gguf(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                source_path=source_path,
                source_is_quantized=source_is_quantized,
                hf_local_path=hf_local,
                catalog_tensor_names=catalog_names,
                mode=args.mode,
                convert_script=args.convert_script,
                require_bf16=bool(args.require_bf16),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "freeze_gguf", str(exc))
        ui.error(9, "freeze_gguf", exc, store.step_path(meta.run_id, "freeze_gguf"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"
    manifest = {
        **payload,
        "gguf_sha256_file": result.gguf_path + ".sha256",
    }

    # GGUF already in step_dir; record sha + manifest as extras
    store.complete_step(
        meta.run_id,
        "freeze_gguf",
        payload,
        log_text=log_text,
        extra_artifacts={
            "freeze_manifest.json": json.dumps(manifest, indent=2).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_freeze_gguf(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'model-*.gguf', '*.sha256', 'freeze_manifest.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="freeze-gguf summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_imatrix(args: argparse.Namespace) -> int:
    from imatrix import build_imatrix
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "freeze_gguf"):
        print(
            "ERROR: Step 09 (freeze-gguf) is not done.\n"
            f"  Run: odg freeze-gguf --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1
    if not store.is_step_done(meta.run_id, "corpus"):
        print(
            "ERROR: Step 07 (corpus) is not done.\n"
            f"  Run: odg corpus --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_imatrix(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "imatrix") and not args.force:
        out = store.read_step_output(meta.run_id, "imatrix")
        ui.already_done(
            10,
            "imatrix",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "imatrix"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    freeze_out = store.read_step_output(meta.run_id, "freeze_gguf") or {}
    gguf_path = freeze_out.get("gguf_path")
    if not gguf_path or not Path(gguf_path).is_file():
        # fallback: look in step dir
        step9 = store.step_path(meta.run_id, "freeze_gguf")
        for name in ("model-bf16.gguf", "model-ref.gguf"):
            cand = step9 / name
            if cand.is_file():
                gguf_path = str(cand)
                break
    if not gguf_path:
        print("ERROR: frozen GGUF path missing from Step 09", file=sys.stderr)
        return 1

    calib_path = store.step_path(meta.run_id, "corpus") / "calib.txt"
    if not calib_path.is_file():
        print(f"ERROR: missing {calib_path}", file=sys.stderr)
        return 1

    catalog = None
    for step_id in ("activation_features", "weight_features", "catalog"):
        cpath = store.step_path(meta.run_id, step_id) / "tensor_catalog.json"
        if cpath.is_file():
            catalog = json.loads(cpath.read_text())
            break

    chunks = int(args.chunks)
    if chunks <= 0:
        chunks_arg = None
    else:
        chunks_arg = chunks
    try:
        imatrix_args = split_extra_args(
            getattr(args, "imatrix_args", None), "--imatrix-args"
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    input_data = {
        "from_steps": ["freeze_gguf", "corpus"],
        "gguf_path": gguf_path,
        "gguf_sha256": freeze_out.get("gguf_sha256"),
        "calib_path": str(calib_path),
        "mode": args.mode,
        "chunks": chunks_arg,
        "imatrix_args": imatrix_args,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "imatrix", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_imatrix(args)

    try:
        with ui.working('Building importance matrix…', explain=print_explain):
            result = build_imatrix(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                gguf_path=gguf_path,
                calib_path=calib_path,
                catalog=catalog,
                gguf_sha256=freeze_out.get("gguf_sha256"),
                mode=args.mode,
                llama_imatrix=args.llama_imatrix,
                n_chunks=chunks_arg,
                extra_args=imatrix_args,
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "imatrix", str(exc))
        ui.error(10, "imatrix", exc, store.step_path(meta.run_id, "imatrix"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "imatrix",
        payload,
        log_text=log_text,
        extra_artifacts={
            "imatrix_manifest.json": json.dumps(payload, indent=2).encode("utf-8")
            + b"\n",
        },
    )

    if print_explain:
        _explain_imatrix(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'imatrix.gguf|proxy', 'imatrix_manifest.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="imatrix summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_reband(args: argparse.Namespace) -> int:
    from reband import BANDS_PER_ROLE, reband_catalog
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "imatrix"):
        print(
            "ERROR: Step 10 (imatrix) is not done.\n"
            f"  Run: odg imatrix --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        ui.step_banner(
            11,
            "Imatrix re-banding (11b)",
            model=meta.model_ref,
            run_id=meta.run_id,
            root=str(meta.root),
            goal="Re-cut role@depth bands on measured importance cliffs.",
            bullets=[
                "Reads tensor_catalog.json + imatrix_proxy.json.",
                "Same band count per role as thirds; only boundaries move.",
            ],
            explain=True,
        )

    if store.is_step_done(meta.run_id, "reband") and not args.force:
        out = store.read_step_output(meta.run_id, "reband")
        ui.already_done(
            11,
            "reband",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "reband"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    catalog = None
    for step_id in ("activation_features", "weight_features", "catalog"):
        cpath = store.step_path(meta.run_id, step_id) / "tensor_catalog.json"
        if cpath.is_file():
            catalog = json.loads(cpath.read_text())
            break
    if not catalog:
        print("ERROR: tensor_catalog.json not found", file=sys.stderr)
        return 1
    proxy_path = store.step_path(meta.run_id, "imatrix") / "imatrix_proxy.json"
    if not proxy_path.is_file():
        print(f"ERROR: missing {proxy_path}", file=sys.stderr)
        return 1
    proxy = json.loads(proxy_path.read_text())
    if not (proxy.get("tensors") or {}):
        print(
            "ERROR: imatrix_proxy.json has no per-tensor importance; "
            "reband needs per-tensor scores.",
            file=sys.stderr,
        )
        return 1

    bands = int(getattr(args, "bands_per_role", None) or BANDS_PER_ROLE)
    if bands < 1:
        print("ERROR: --bands-per-role must be >= 1", file=sys.stderr)
        return 1

    # Real per-channel stats preferred; proxy per-tensor fallback inside.
    imatrix_gguf = store.step_path(meta.run_id, "imatrix") / "imatrix.gguf"
    imatrix_gguf_arg = str(imatrix_gguf) if imatrix_gguf.is_file() else None

    input_data = {
        "from_steps": ["imatrix", "activation_features"],
        "bands_per_role": bands,
        "imatrix_gguf": imatrix_gguf_arg,
        "n_groups_before": len(catalog.get("groups") or {}),
    }
    try:
        step_dir = store.begin_step(
            meta.run_id, "reband", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_reband(args)

    try:
        with ui.working('Re-banding depth groups…', explain=print_explain):
            new_catalog, report = reband_catalog(
                catalog, proxy, bands_per_role=bands,
                imatrix_gguf=imatrix_gguf_arg,
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "reband", str(exc))
        ui.error(11, "reband", exc, store.step_path(meta.run_id, "reband"))
        return 1

    (step_dir / "tensor_catalog.json").write_text(
        json.dumps(new_catalog, indent=2) + "\n", encoding="utf-8"
    )
    (step_dir / "reband_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "n_groups_before": report["n_groups_before"],
        "n_groups_after": report["n_groups_after"],
        "score_source": report["score_source"],
        "roles_rebanded": report["roles_rebanded"],
        "skipped_roles": report["skipped_roles"],
        "boundaries": report["boundaries"],
        "catalog_sha256": report["catalog_sha256"],
    }
    log_text = (
        f"rebanded {report['n_groups_before']} -> "
        f"{report['n_groups_after']} groups "
        f"({len(report['roles_rebanded'])} roles)\n"
    )
    store.complete_step(
        meta.run_id, "reband", payload, log_text=log_text,
        extra_artifacts={
            "tensor_catalog.json": json.dumps(new_catalog, indent=2).encode()
            + b"\n",
            "reband_report.json": json.dumps(report, indent=2).encode()
            + b"\n",
        },
    )
    if print_explain:
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=("output.json", "tensor_catalog.json", "reband_report.json",
                   "status.json", "log.txt"),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="reband summary")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_reference_logits(args: argparse.Namespace) -> int:
    from logits import cache_reference_logits
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "imatrix"):
        print(
            "ERROR: Step 10 (imatrix) is not done.\n"
            f"  Run: odg imatrix --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1
    if not store.is_step_done(meta.run_id, "freeze_gguf"):
        print(
            "ERROR: Step 09 (freeze-gguf) is not done.\n"
            f"  Run: odg freeze-gguf --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1
    if not store.is_step_done(meta.run_id, "corpus"):
        print(
            "ERROR: Step 07 (corpus) is not done.\n"
            f"  Run: odg corpus --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_reference_logits(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "reference_logits") and not args.force:
        out = store.read_step_output(meta.run_id, "reference_logits")
        ui.already_done(
            11,
            "reference_logits",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "reference_logits"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    freeze_out = store.read_step_output(meta.run_id, "freeze_gguf") or {}
    gguf_path = freeze_out.get("gguf_path")
    if not gguf_path or not Path(gguf_path).is_file():
        step9 = store.step_path(meta.run_id, "freeze_gguf")
        for name in ("model-bf16.gguf", "model-ref.gguf"):
            cand = step9 / name
            if cand.is_file():
                gguf_path = str(cand)
                break
    if not gguf_path:
        print("ERROR: frozen GGUF path missing from Step 09", file=sys.stderr)
        return 1

    corpus_dir = store.step_path(meta.run_id, "corpus")
    search_path = corpus_dir / "search.txt"
    heldout_path = corpus_dir / "heldout.txt"
    if not search_path.is_file() or not heldout_path.is_file():
        print(
            f"ERROR: need {search_path} and {heldout_path}",
            file=sys.stderr,
        )
        return 1

    input_data = {
        "from_steps": ["freeze_gguf", "corpus", "imatrix"],
        "gguf_path": gguf_path,
        "gguf_sha256": freeze_out.get("gguf_sha256"),
        "search_path": str(search_path),
        "heldout_path": str(heldout_path),
        "mode": args.mode,
    }
    try:
        perplexity_args = split_extra_args(
            getattr(args, "perplexity_args", None), "--perplexity-args"
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    input_data["perplexity_args"] = perplexity_args

    try:
        step_dir = store.begin_step(
            meta.run_id, "reference_logits", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_reference_logits(args)

    try:
        with ui.working('Caching reference logits…', explain=print_explain):
            result = cache_reference_logits(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                gguf_path=gguf_path,
                search_path=search_path,
                heldout_path=heldout_path,
                gguf_sha256=freeze_out.get("gguf_sha256"),
                mode=args.mode,
                llama_perplexity=args.llama_perplexity,
                extra_args=perplexity_args,
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "reference_logits", str(exc))
        ui.error(11, "reference_logits", exc, store.step_path(meta.run_id, "reference_logits"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "reference_logits",
        payload,
        log_text=log_text,
    )

    if print_explain:
        _explain_reference_logits(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'logits_manifest.json', 'logits-*.bin|MISSING', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="reference-logits summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


#: Canonical catalog resolution order for every step that consumes probe
#: groups (sensitivity, optimize, export). The rebanded catalog MUST win
#: wherever it exists: group_ids are only meaningful within one catalog,
#: and sizing/probing/exporting against different catalogs in one run
#: silently mis-maps groups to tensor sets (caught once by the measured-
#: size sanity check in optimize — never again by construction).
PROBE_CATALOG_ORDER = (
    "reband", "activation_features", "weight_features", "catalog",
)


def resolve_probe_catalog(store, run_id):
    """Return (catalog_dict, source_step) or (None, None)."""
    for step_id in PROBE_CATALOG_ORDER:
        cpath = store.step_path(run_id, step_id) / "tensor_catalog.json"
        if cpath.is_file():
            return json.loads(cpath.read_text()), step_id
    return None, None


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from sensitivity import build_sensitivity_table
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "reference_logits"):
        print(
            "ERROR: Step 11 (reference-logits) is not done.\n"
            f"  Run: odg reference-logits --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    try:
        meta, fmt = _apply_quant_format(
            store, meta, args.quant, explain=print_explain
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if print_explain:
        _banner_sensitivity(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "sensitivity") and not args.force:
        out = store.read_step_output(meta.run_id, "sensitivity")
        ui.already_done(
            12,
            "sensitivity",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "sensitivity"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    catalog, catalog_source = resolve_probe_catalog(store, meta.run_id)
    if not catalog:
        print("ERROR: tensor_catalog.json not found", file=sys.stderr)
        return 1
    if catalog_source == "reband":
        # Warn if the rebanded catalog predates the current features: the
        # reband embeds the input catalog sha in its report.
        rep_path = store.step_path(meta.run_id, "reband") / "reband_report.json"
        cur_path = (
            store.step_path(meta.run_id, "activation_features") / "tensor_catalog.json"
        )
        try:
            rep = json.loads(rep_path.read_text()) if rep_path.is_file() else {}
            cur = json.loads(cur_path.read_text()) if cur_path.is_file() else {}
            if (
                rep.get("prev_sha256")
                and cur.get("catalog_sha256")
                and rep["prev_sha256"] != cur["catalog_sha256"]
            ):
                print(
                    "WARNING: rebanded catalog was cut from an older features "
                    "catalog; re-run: odg reband --force",
                    file=sys.stderr,
                )
        except (OSError, ValueError):
            pass

    freeze_out = store.read_step_output(meta.run_id, "freeze_gguf") or {}
    search_path = store.step_path(meta.run_id, "corpus") / "search.txt"
    imatrix_proxy = store.step_path(meta.run_id, "imatrix") / "imatrix_proxy.json"

    # Llama-mode probe inputs: frozen model GGUF + step-11 search KL cache.
    model_gguf = freeze_out.get("gguf_path")
    kl_base_bin = None
    logits_manifest = store.step_path(meta.run_id, "reference_logits") / "logits_manifest.json"
    if logits_manifest.is_file():
        try:
            kl_base_bin = (json.loads(logits_manifest.read_text()) or {}).get(
                "logits_search_path"
            )
        except (OSError, ValueError):
            kl_base_bin = None

    # Prefer proxy mode when llama requested but unavailable path
    mode = args.mode
    if mode == "auto":
        mode = "proxy"

    baseline = args.baseline or fmt.baseline_type
    if getattr(args, "probe_types", None):
        probe_types = [t.strip().upper() for t in args.probe_types.split(",") if t.strip()]
        if not probe_types:
            raise ValueError("--probe-types was empty")
    else:
        from sensitivity import default_probe_grid

        # Profile grid plus every ladder type any group needs (e.g. Q8
        # for Q8-pinned groups); per-group filtering keeps it tight.
        probe_types = default_probe_grid(
            catalog, list(fmt.probe_types), start_type=baseline,
        )

    input_data = {
        "from_steps": ["reference_logits", "imatrix", "corpus"],
        "mode": mode,
        "baseline": baseline,
        "probe_types": probe_types,
        "quant_format": fmt.id,
        "kld_objective": getattr(args, "kld_objective", "mean"),
        "certificate": getattr(args, "certificate", "bounded"),
        "lipschitz": getattr(args, "lipschitz", None),
        "budget_mb": getattr(args, "budget_mb", None),
        "budget_ratio": (
            float(getattr(args, "budget_ratio"))
            if getattr(args, "budget_ratio", None) is not None
            else float(fmt.budget_ratio)
        ),
        "gguf_sha256": freeze_out.get("gguf_sha256"),
        "search_path": str(search_path),
        "n_catalog_groups": len(catalog.get("groups") or {}),
        "catalog_source": catalog_source,
        "jobs": args.jobs,
        "splice_farm": (
            str(getattr(args, "splice_farm", None))
            if getattr(args, "splice_farm", None) is not None
            else None
        ),
    }
    try:
        sens_perplexity_args = split_extra_args(
            getattr(args, "perplexity_args", None), "--perplexity-args"
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    input_data["perplexity_args"] = sens_perplexity_args
    sens_fixed_groups = frozenset(
        g.strip() for g in str(getattr(args, "fixed_groups", None) or "").split(",")
        if g.strip()
    )
    input_data["fixed_groups"] = sorted(sens_fixed_groups)
    # Trial quants benefit from the run's imatrix.gguf (importance-weighted
    # K-quants); without it K-means fall back to dumb rounding and every
    # measured column is pessimistic. Mirrors the cmd_reband pattern.
    imatrix_gguf = store.step_path(meta.run_id, "imatrix") / "imatrix.gguf"
    input_data["imatrix_gguf"] = (
        str(imatrix_gguf) if imatrix_gguf.is_file() else None
    )

    # No sensitivity-level budget flags: the pricing reference is derived,
    # not asked for. It is the intended step-13 budget exactly (same
    # threaded flags) — Pareto targets above the solve budget are dropped,
    # so no solve is ever looser than intended. Computed before begin_step
    # so input.json records it.
    from optimizer import (
        default_budget_bytes as _sens_budget,
        pricing_reference_budget as _sens_ref,
    )

    _sens_bm = getattr(args, "budget_mb", None)
    if _sens_bm is not None:
        _intended = int(float(_sens_bm) * 1024 * 1024)
    else:
        _sens_br = getattr(args, "budget_ratio", None)
        _intended = _sens_budget(
            catalog,
            ratio=float(_sens_br)
            if _sens_br is not None else float(fmt.budget_ratio),
        )
    _reference = _sens_ref(catalog, intended_bytes=_intended)
    input_data["intended_bytes"] = _intended
    input_data["pricing_reference_bytes"] = _reference

    try:
        step_dir = store.begin_step(
            meta.run_id, "sensitivity", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_sensitivity(args)

    try:
        with ui.working('Probing sensitivity (ΔKLD)…', explain=print_explain):
            result, _rows = build_sensitivity_table(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                catalog=catalog,
                gguf_sha256=freeze_out.get("gguf_sha256"),
                search_path=search_path if search_path.is_file() else None,
                imatrix_proxy_path=imatrix_proxy if imatrix_proxy.is_file() else None,
                mode=mode,
                probe_types=probe_types,
                baseline_type=baseline,
                model_gguf=model_gguf,
                kl_base_bin=kl_base_bin,
                trials_dir=step_dir / "trials",
                jobs=args.jobs,
                llama_quantize=args.llama_quantize,
                llama_perplexity=args.llama_perplexity,
                perplexity_args=sens_perplexity_args,
                imatrix_gguf=input_data["imatrix_gguf"],
                fixed_groups=sens_fixed_groups,
                certificate_mode=getattr(args, "certificate", "bounded"),
                kld_objective=getattr(args, "kld_objective", "mean"),
                lipschitz_L=getattr(args, "lipschitz", None),
                pricing_budget_bytes=_reference,
                splice_farm=getattr(args, "splice_farm", None),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "sensitivity", str(exc))
        ui.error(12, "sensitivity", exc, store.step_path(meta.run_id, "sensitivity"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "sensitivity",
        payload,
        log_text=log_text,
    )

    if print_explain:
        _explain_sensitivity(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=tuple(x.strip() for x in 'output.json, sensitivity.json, status.json, log.txt'.split(',')),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="sensitivity summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from optimizer import optimize_recipes
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "sensitivity"):
        print(
            "ERROR: Step 12 (sensitivity) is not done.\n"
            f"  Run: odg sensitivity --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    try:
        meta, fmt = _apply_quant_format(
            store, meta, args.quant, explain=print_explain
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if print_explain:
        _banner_optimize(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "optimize") and not args.force:
        out = store.read_step_output(meta.run_id, "optimize")
        ui.already_done(
            13,
            "optimize",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "optimize"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    sens_path = store.step_path(meta.run_id, "sensitivity") / "sensitivity.json"
    if not sens_path.is_file():
        print(f"ERROR: missing {sens_path}", file=sys.stderr)
        return 1
    sensitivity = json.loads(sens_path.read_text())

    catalog, _catalog_source = resolve_probe_catalog(store, meta.run_id)
    if not catalog:
        print("ERROR: tensor_catalog.json not found", file=sys.stderr)
        return 1

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    freeze_out = store.read_step_output(meta.run_id, "freeze_gguf") or {}
    imatrix_out = store.read_step_output(meta.run_id, "imatrix") or {}
    corpus_out = store.read_step_output(meta.run_id, "corpus") or {}

    # File overhead (GGUF header + KV metadata) is measured from the source
    # GGUF's data offset, plus 4 KiB safety for the small KV delta export
    # adds (~512 B observed). It is subtracted from the budget pre-solve so
    # --budget-mb means final file size on disk. Falls back to 0 (recorded)
    # when no source GGUF is available.
    file_overhead_bytes = 0
    _freeze_gguf = freeze_out.get("gguf_path")
    if _freeze_gguf and Path(_freeze_gguf).is_file():
        try:
            from gguf_tensors import gguf_tensor_map

            file_overhead_bytes = (
                int(gguf_tensor_map(Path(_freeze_gguf))["data_offset"]) + 4096
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: cannot read GGUF overhead: {exc}; using 0",
                  file=sys.stderr)
            file_overhead_bytes = 0

    budget_bytes = None
    if args.budget_mb is not None:
        budget_bytes = int(float(args.budget_mb) * 1024 * 1024)
    budget_ratio = (
        float(args.budget_ratio)
        if args.budget_ratio is not None
        else float(fmt.budget_ratio)
    )

    input_data = {
        "from_step": "sensitivity",
        "budget_mb": args.budget_mb,
        "budget_ratio": budget_ratio,
        "quant_format": fmt.id,
        "gguf_sha256": freeze_out.get("gguf_sha256"),
        "optimizer": getattr(args, "optimizer", "dp_mckp"),
        "kld_objective": getattr(args, "kld_objective", "mean"),
        "certificate": getattr(args, "certificate", "bounded"),
        "lipschitz": getattr(args, "lipschitz", None),
        "jobs": getattr(args, "jobs", 1),
        "pareto_ratios": getattr(args, "pareto_ratios", None),
        "fixed_groups": sorted(
            g.strip()
            for g in str(getattr(args, "fixed_groups", None) or "").split(",")
            if g.strip()
        ),
        "file_overhead_bytes": file_overhead_bytes,
    }

    pareto_ratios = None
    if getattr(args, "pareto_ratios", None):
        try:
            pareto_ratios = [float(r) for r in str(args.pareto_ratios).split(",")]
        except ValueError:
            print(f"ERROR: bad --pareto-ratios: {args.pareto_ratios}", file=sys.stderr)
            return 1

    fixed_groups = frozenset(
        g.strip() for g in str(getattr(args, "fixed_groups", None) or "").split(",")
        if g.strip()
    )

    imatrix_groups = None
    imatrix_proxy = store.step_path(meta.run_id, "imatrix") / "imatrix_proxy.json"
    if imatrix_proxy.is_file():
        try:
            imatrix_groups = (json.loads(imatrix_proxy.read_text()) or {}).get("groups")
        except (OSError, ValueError):
            imatrix_groups = None

    try:
        step_dir = store.begin_step(
            meta.run_id, "optimize", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_optimize(args)

    try:
        with ui.working('Optimizing quant recipe…', explain=print_explain):
            result = optimize_recipes(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                catalog=catalog,
                sensitivity=sensitivity,
                budget_bytes=budget_bytes,
                budget_ratio=budget_ratio,
                hf_repo_id=resolve_out.get("hf_repo_id"),
                gguf_sha256=freeze_out.get("gguf_sha256"),
                imatrix_sha256=imatrix_out.get("imatrix_sha256"),
                corpus_id=corpus_out.get("corpus_id"),
                optimizer=getattr(args, "optimizer", "dp_mckp"),
                kld_objective=getattr(args, "kld_objective", "mean"),
                certificate_mode=getattr(args, "certificate", "bounded"),
                lipschitz_L=getattr(args, "lipschitz", None),
                jobs=int(getattr(args, "jobs", 1) or 1),
                pareto_ratios=pareto_ratios,
                imatrix_groups=imatrix_groups,
                fixed_groups=fixed_groups,
                file_overhead_bytes=file_overhead_bytes,
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "optimize", str(exc))
        ui.error(13, "optimize", exc, store.step_path(meta.run_id, "optimize"))
        return 1

    payload = result.summary_dict()
    # Keep summary smaller
    payload_out = {
        "model_ref": result.model_ref,
        "method": result.method,
        "budget_bytes": result.budget_bytes,
        "estimated_bytes": result.estimated_bytes,
        "predicted_delta_kld": result.predicted_delta_kld,
        "n_groups": result.n_groups,
        "recipe_path": result.recipe_path,
        "tensor_type_file": result.tensor_type_file,
        "n_pareto": len(result.pareto_paths),
        "assignments": result.assignments,
        "optimizer": result.optimizer,
        "kld_objective": result.kld_objective,
        "tail_cap": result.tail_cap,
        "auto_cap": result.auto_cap,
        "total_tail_kld": result.total_tail_kld,
        "total_mean_kld": result.total_mean_kld,
        "certificate": result.certificate,
        "steps_log": result.steps_log,
        "notes": result.notes,
    }
    log_text = "\n".join(result.steps_log) + "\n"

    store.complete_step(
        meta.run_id,
        "optimize",
        payload_out,
        log_text=log_text,
    )

    if print_explain:
        _explain_optimize(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('output.json', 'recipe.yaml', 'recipe.tt', 'pareto/', 'optimize_manifest.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload_out, title="optimize summary")
    else:
        print(json.dumps(payload_out, indent=2))

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from export import export_gguf
    from store import StepAlreadyDone

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "optimize"):
        print(
            "ERROR: Step 13 (optimize) is not done.\n"
            f"  Run: odg optimize --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    try:
        meta, fmt = _apply_quant_format(
            store, meta, args.quant, explain=print_explain
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if print_explain:
        _banner_export(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "export") and not args.force:
        out = store.read_step_output(meta.run_id, "export")
        ui.already_done(
            14,
            "export",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "export"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    opt_dir = store.step_path(meta.run_id, "optimize")
    recipe_path = opt_dir / "recipe.yaml"
    recipe_tt = opt_dir / "recipe.tt"
    if not recipe_path.is_file() or not recipe_tt.is_file():
        print(f"ERROR: missing recipe.yaml / recipe.tt in {opt_dir}", file=sys.stderr)
        return 1

    freeze_out = store.read_step_output(meta.run_id, "freeze_gguf") or {}
    gguf_in = freeze_out.get("gguf_path")
    if not gguf_in or not Path(gguf_in).is_file():
        step9 = store.step_path(meta.run_id, "freeze_gguf")
        for name in ("model-bf16.gguf", "model-ref.gguf"):
            if (step9 / name).is_file():
                gguf_in = str(step9 / name)
                break
    if not gguf_in:
        print("ERROR: frozen GGUF missing", file=sys.stderr)
        return 1

    imatrix_out = store.read_step_output(meta.run_id, "imatrix") or {}
    imatrix_path = imatrix_out.get("imatrix_path")
    if imatrix_path and not Path(imatrix_path).is_file():
        imatrix_path = None

    mode = args.mode
    if mode == "auto":
        mode = "dry-run"  # resolved inside export if binary appears

    base_type = args.base_type or fmt.base_type

    input_data = {
        "from_step": "optimize",
        "gguf_in": gguf_in,
        "recipe_path": str(recipe_path),
        "mode": args.mode,
        "base_type": base_type,
        "quant_format": fmt.id,
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "export", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_export(args)

    try:
        with ui.working('Exporting candidate GGUF…', explain=print_explain):
            result = export_gguf(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                gguf_in=gguf_in,
                recipe_path=recipe_path,
                recipe_tt=recipe_tt,
                imatrix_path=imatrix_path,
                mode=args.mode,
                llama_quantize=args.llama_quantize,
                base_type=base_type,
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "export", str(exc))
        ui.error(14, "export", exc, store.step_path(meta.run_id, "export"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"
    store.complete_step(meta.run_id, "export", payload, log_text=log_text)

    if print_explain:
        _explain_export(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('input.json', 'output.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="export summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from store import StepAlreadyDone
    from validate import validate_and_release

    print_explain = not args.no_explain
    store = _store(args)

    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not store.is_step_done(meta.run_id, "export"):
        print(
            "ERROR: Step 14 (export) is not done.\n"
            f"  Run: odg export --model {meta.model_ref}",
            file=sys.stderr,
        )
        return 1

    if print_explain:
        _banner_validate(meta.model_ref, meta.run_id, meta.root)

    if store.is_step_done(meta.run_id, "validate") and not args.force:
        out = store.read_step_output(meta.run_id, "validate")
        ui.already_done(
            15,
            "validate",
            run_id=meta.run_id,
            path=store.step_path(meta.run_id, "validate"),
            output=out,
            summary=store.summary(meta.run_id) if print_explain else None,
            explain=print_explain,
        )
        return 0

    export_out = store.read_step_output(meta.run_id, "export") or {}
    recipe_path = Path(export_out.get("recipe_path") or "")
    if not recipe_path.is_file():
        recipe_path = store.step_path(meta.run_id, "export") / "recipe.yaml"
    if not recipe_path.is_file():
        recipe_path = store.step_path(meta.run_id, "optimize") / "recipe.yaml"
    if not recipe_path.is_file():
        print("ERROR: recipe.yaml not found", file=sys.stderr)
        return 1

    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    desc = resolve_out.get("descriptor") or {}
    sens_path = store.step_path(meta.run_id, "sensitivity") / "sensitivity.json"
    opt_out = store.read_step_output(meta.run_id, "optimize") or {}
    assignments = opt_out.get("assignments") or {}
    opt_manifest_path = (
        store.step_path(meta.run_id, "optimize") / "optimize_manifest.json"
    )
    optimize_manifest = None
    if opt_manifest_path.is_file():
        optimize_manifest = json.loads(opt_manifest_path.read_text())
        if not assignments:
            assignments = (optimize_manifest.get("primary") or {}).get(
                "assignments"
            ) or {}

    catalog, _catalog_source = resolve_probe_catalog(store, meta.run_id)
    if not catalog:
        print("ERROR: tensor_catalog.json not found", file=sys.stderr)
        return 1

    input_data = {
        "from_step": "export",
        "mode": args.mode,
        "strict": bool(args.strict),
        "export_method": export_out.get("method"),
        "gguf_out": export_out.get("gguf_out"),
        "perplexity_args": split_extra_args(
            getattr(args, "perplexity_args", None), "--perplexity-args"
        ),
        "llama_perplexity": (
            str(args.llama_perplexity)
            if getattr(args, "llama_perplexity", None) else None
        ),
    }

    try:
        step_dir = store.begin_step(
            meta.run_id, "validate", input_data, force=args.force
        )
    except StepAlreadyDone:
        return cmd_validate(args)

    try:
        with ui.working('Validating & staging release…', explain=print_explain):
            ref_out = store.read_step_output(meta.run_id, "reference_logits") or {}
            result = validate_and_release(
                model_ref=meta.model_ref,
                out_dir=step_dir,
                recipe_path=recipe_path,
                export_manifest=export_out,
                specialty_domain=desc.get("specialty_domain"),
                sensitivity_path=sens_path if sens_path.is_file() else None,
                catalog=catalog,
                assignments=assignments,
                optimize_manifest=optimize_manifest,
                resolve_descriptor=desc,
                mode=args.mode,
                allow_provisional=not args.strict,
                heldout_txt=ref_out.get("heldout_path"),
                heldout_bin=ref_out.get("logits_heldout_path"),
                llama_perplexity=(
                    str(args.llama_perplexity)
                    if getattr(args, "llama_perplexity", None) else None
                ),
                perplexity_args=split_extra_args(
                    getattr(args, "perplexity_args", None),
                    "--perplexity-args",
                ),
            )
    except Exception as exc:  # noqa: BLE001
        store.fail_step(meta.run_id, "validate", str(exc))
        ui.error(15, "validate", exc, store.step_path(meta.run_id, "validate"))
        return 1

    payload = result.summary_dict()
    log_text = "\n".join(result.steps_log) + "\n"
    store.complete_step(meta.run_id, "validate", payload, log_text=log_text)

    # Mark run complete when validate finishes
    try:
        meta2 = store.load_run(meta.run_id)
        if result.verdict in {"RELEASE", "PROVISIONAL"}:
            meta2.status = "done"
            meta2.current_step = None
            store._write_run_meta(meta2)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass

    if print_explain:
        _explain_validate(result)
        ui.checkpoint_saved(
            run_id=meta.run_id,
            step_dir=step_dir,
            files=('input.json', 'output.json', 'status.json', 'log.txt'),
            explain=True,
        )
        ui.run_summary(store.summary(meta.run_id))
        ui.json_panel(payload, title="validate summary")
    else:
        print(json.dumps(payload, indent=2))

    return 0 if result.verdict != "FAIL" else 2


def cmd_status(args: argparse.Namespace) -> int:
    store = _store(args)
    if args.run:
        meta = store.load_run(args.run)
    elif args.model:
        meta = store.latest_run_for_model(args.model)
        if meta is None:
            print(f"No run found for model {args.model!r}", file=sys.stderr)
            return 1
    else:
        runs = store.list_runs()
        if not runs:
            ui.info("No runs yet. Run: odg resolve --model …")
            return 0
        meta = runs[0]
    ui.run_summary(store.summary(meta.run_id))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = _store(args)
    runs = store.list_runs()
    if not runs:
        ui.info(f"No runs under {store.runs_dir}")
        return 0
    ui.runs_table(runs)
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from hardware import list_devices_rows

    ui.devices_table(list_devices_rows(), explain=not args.no_explain)
    if not args.no_explain:
        ui.info(
            "Use with: odg fit --model <ref> --device <id>\n"
            "Or describe directly: odg fit --model <ref> --gpu 24GB / --ram 32GB"
        )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Hardware-aware pipeline: derive the byte budget, then run steps 01–15."""
    import hardware

    print_explain = not args.no_explain

    # 1 · Build the hardware profile from intent flags (or detect).
    try:
        if args.device or args.gpu or args.ram:
            profile = hardware.profile_from_flags(
                gpu=args.gpu,
                ram=args.ram,
                device=args.device,
                cpu_only=bool(args.cpu_only),
            )
        else:
            profile = hardware.detect_profile()
            if profile is None:
                print(
                    "ERROR: could not detect hardware. Describe it instead:\n"
                    "  odg fit --model … --gpu 24GB | --ram 32GB | --device rtx-3060-12gb\n"
                    "  (see: odg devices)",
                    file=sys.stderr,
                )
                return 1
            ui.warn(
                f"No hardware flags given — auto-detected: {profile.id} "
                f"({profile.memory_pool_gb():g} GB {profile.kind}). "
                "Pass --device/--gpu/--ram to override.",
                explain=print_explain,
            )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ui.step_banner(
        0,
        "Hardware-aware fit",
        model=args.model,
        run_id=args.run or "(auto)",
        root=str(getattr(args, "artifacts", None) or Path.cwd() / "artifacts"),
        goal="Translate hardware into a weight-byte budget, then optimize under it.",
        bullets=[
            f"Hardware: {profile.id} ({profile.memory_pool_gb():g} GB {profile.kind})",
            f"Context: {args.ctx or '(model default, capped at 4096)'} · KV {args.kv_dtype}",
            "budget = pool × usable − KV cache − runtime overhead",
        ],
        explain=print_explain,
    )

    # 2 · Resolve first — the KV-cache math needs the architecture descriptor.
    resolve_args = argparse.Namespace(
        command="run",
        artifacts=args.artifacts,
        model=args.model,
        quant=args.quant,
        no_ask=True,
        new_run=bool(args.new_run),
        run=args.run,
        force=bool(args.force),
        prefer_hf=bool(args.prefer_hf),
        download_weights=bool(args.download_weights),
        until="resolve",
        from_step=None,
        quiet=bool(args.quiet),
        no_explain=args.no_explain,
    )
    code = cmd_run(resolve_args)
    if code != 0:
        return code

    store = _store(args)
    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    resolve_out = store.read_step_output(meta.run_id, "resolve") or {}
    descriptor = resolve_out.get("descriptor") or {}

    # 3 · Derive the budget and show every subtraction.
    try:
        plan = hardware.derive_budget(
            profile, descriptor, ctx=args.ctx, kv_dtype=args.kv_dtype
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ui.section("hardware → budget", explain=print_explain)
    ui.kv(plan.display_rows(), explain=print_explain)
    ui.notes(plan.notes, explain=print_explain)

    fit_plan_path = Path(meta.root) / "fit_plan.json"
    store.write_json(fit_plan_path, plan.summary_dict())
    ui.info(f"Fit plan saved: {fit_plan_path}", explain=print_explain)

    # A resumed run keeps its old recipe unless the user forces a re-optimize.
    if store.is_step_done(meta.run_id, "optimize") and not args.force:
        opt_in = store.step_path(meta.run_id, "optimize") / "input.json"
        prev_budget = None
        if opt_in.is_file():
            prev_budget = (json.loads(opt_in.read_text()) or {}).get("budget_mb")
        if prev_budget is None or abs(float(prev_budget) - plan.weight_budget_mb) > 1.0:
            ui.warn(
                "This run already has an optimized recipe under a different "
                "budget — it will be kept. Pass --force (or --new-run) to "
                f"re-optimize for {plan.weight_budget_mb:.0f} MiB.",
                explain=print_explain,
            )

    # 4 · Run the rest of the pipeline under the derived budget.
    if args.until and args.until.replace("-", "_") == "resolve":
        return 0
    pipeline_args = argparse.Namespace(
        command="run",
        artifacts=args.artifacts,
        model=args.model,
        quant=args.quant,
        no_ask=True,
        new_run=False,
        run=meta.run_id,
        force=bool(args.force),
        prefer_hf=bool(args.prefer_hf),
        download_weights=bool(args.download_weights),
        until=args.until,
        from_step="load",
        quiet=bool(args.quiet),
        no_explain=args.no_explain,
        budget_mb=plan.weight_budget_mb,
        budget_ratio=None,
        mode=_run_flag(args, "mode", "auto") or "auto",
        target_tokens=int(_run_flag(args, "target_tokens", 50_000) or 50_000),
        seed=int(_run_flag(args, "seed", 42)
                 if _run_flag(args, "seed", 42) is not None else 42),
        max_docs=int(_run_flag(args, "max_docs", 32) or 32),
        freeze_mode=_run_flag(args, "freeze_mode", "auto") or "auto",
        convert_script=_run_flag(args, "convert_script", None),
        require_bf16=bool(_run_flag(args, "require_bf16", False)),
        chunks=int(_run_flag(args, "chunks", 64)
                   if _run_flag(args, "chunks", 64) is not None else 64),
        imatrix_args=_run_flag(args, "imatrix_args", None),
        perplexity_args=_run_flag(args, "perplexity_args", None),
        bands_per_role=int(_run_flag(args, "bands_per_role", 3) or 3),
        jobs=int(_run_flag(args, "jobs", 1) or 1),
        probe_types=_run_flag(args, "probe_types", None),
        fixed_groups=_run_flag(args, "fixed_groups", None),
        optimizer=_run_flag(args, "optimizer", "dp_mckp") or "dp_mckp",
        kld_objective=_run_flag(args, "kld_objective", "mean") or "mean",
        certificate=_run_flag(args, "certificate", "bounded") or "bounded",
        lipschitz=_run_flag(args, "lipschitz", None),
        pareto_ratios=_run_flag(args, "pareto_ratios", None),
        splice_farm=_run_flag(args, "splice_farm", None),
        export_mode=_run_flag(args, "export_mode", "auto") or "auto",
        base_type=_run_flag(args, "base_type", None),
        validate_mode=_run_flag(args, "validate_mode", "auto") or "auto",
        strict=bool(_run_flag(args, "strict", False)),
        only_quantizable=bool(_run_flag(args, "only_quantizable", True)),
    )
    return cmd_run(pipeline_args)


def cmd_experiment(args: argparse.Namespace) -> int:
    """Same-model / same-harness experiment. Quantization is the only variable."""
    from experiment import (
        VARIANTS,
        VARIANT_LABELS,
        compare,
        default_root,
        prepare,
        render_comparison_markdown,
        results_dir,
        run_variant,
    )

    print_explain = not args.no_explain
    root = Path(args.root) if args.root else default_root()
    model = args.model or "google/functiongemma-270m-it"

    if args.action == "prepare":
        ui.step_banner(
            0,
            "Experiment prepare",
            model=model,
            run_id="(experiment)",
            root=str(root),
            goal="Same BF16 checkpoint → Q4_K_M / Q5_K_M / Q6_K / OpenDynamicGGUF.",
            bullets=[
                "Original weights stay untouched",
                "Uniform baselines share one imatrix with ODG",
                "Evaluation config is pinned in benchmark/config.json",
            ],
            explain=print_explain,
        )
        try:
            with ui.working("Preparing variants…", explain=print_explain):
                manifest = prepare(
                    root,
                    model=args.model,
                    force=args.force,
                    skip_odg=args.skip_odg,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if print_explain:
            sizes = manifest.get("sizes") or {}
            rows = [("model", manifest.get("model")), ("n_params", manifest.get("n_params"))]
            for vid, info in sizes.items():
                mb = info.get("mb")
                status = "ok" if info.get("present") else "missing"
                rows.append(
                    (
                        VARIANT_LABELS.get(vid, vid),
                        f"{status} · {mb:.1f} MB" if mb is not None else status,
                    )
                )
            ui.section("experiment prepare")
            ui.kv(rows)
            ui.notes(manifest.get("notes") or [])
            ui.next_step("odg experiment run --all   # or ./benchmark/run_all.sh")
        else:
            print(json.dumps(manifest, indent=2))
        return 0

    if args.action == "run":
        if args.all:
            variants = list(VARIANTS)
        elif args.variant:
            variants = [args.variant]
        else:
            print("ERROR: pass --variant V or --all", file=sys.stderr)
            return 1
        ui.step_banner(
            0,
            "Experiment run",
            model=model,
            run_id="(experiment)",
            root=str(root),
            goal="Identical lm-eval config on every variant. Only weights change.",
            bullets=[
                f"Variants: {', '.join(VARIANT_LABELS.get(v, v) for v in variants)}",
                f"Suite: {args.suite}",
                "Tasks: MMLU, GSM8K, HellaSwag, ARC-Challenge, TruthfulQA",
            ],
            explain=print_explain,
        )
        rc = 0
        for vid in variants:
            try:
                with ui.working(f"Evaluating {VARIANT_LABELS.get(vid, vid)}…", explain=print_explain):
                    payload = run_variant(
                        vid,
                        root,
                        suite=args.suite,
                        limit_override=args.limit,
                        gguf_backend=args.gguf_backend,
                        force=args.force,
                        skip_quality=args.skip_quality,
                        skip_behavior=args.skip_behavior,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR ({vid}): {exc}", file=sys.stderr)
                rc = 1
                continue
            if print_explain:
                q = payload.get("quality") or {}
                rows = [
                    ("variant", payload.get("label")),
                    ("pin", payload.get("config_fingerprint")),
                    ("size", f"{(payload.get('memory') or {}).get('mb') or '—'} MB"),
                ]
                if q.get("skipped"):
                    rows.append(("quality", f"skipped — {q.get('reason')}"))
                else:
                    for tid, entry in (q.get("tasks") or {}).items():
                        score = entry.get("score")
                        rows.append(
                            (
                                tid,
                                "—" if score is None else f"{score * 100:.1f}"
                                if 0 <= float(score) <= 1
                                else str(score),
                            )
                        )
                ui.section(str(payload.get("label") or vid))
                ui.kv(rows)
                ui.notes(payload.get("notes") or [])
            else:
                print(json.dumps(payload, indent=2))
        if print_explain:
            ui.next_step("odg experiment compare")
        return rc

    # compare
    ui.step_banner(
        0,
        "Experiment compare",
        model=model,
        run_id="(experiment)",
        root=str(root),
        goal="One table: quality, size, behavior, inference — same pin, every variant.",
        bullets=["Bold = best quantized score per task", "Size vs Q4_K_M is part of the claim"],
        explain=print_explain,
    )
    with ui.working("Writing comparison table…", explain=print_explain):
        comp = compare(root)
    md = render_comparison_markdown(comp)
    out_md = results_dir(root) / "comparison.md"
    if print_explain:
        print(md)
        ui.notes(comp.get("warnings") or [])
        ui.next_step(f"Table written to {out_md}")
    else:
        print(json.dumps(comp, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from report import generate_report

    print_explain = not args.no_explain
    store = _store(args)
    try:
        meta = _require_run(store, model=args.model, run_id=args.run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        with ui.working("Rendering report.html…", explain=print_explain):
            result = generate_report(Path(meta.root), out_path=args.out)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if print_explain:
        ui.section("report")
        ui.kv(
            [
                ("run", meta.run_id),
                ("sections", f"{result.sections_rendered} rendered"),
                (
                    "missing",
                    ", ".join(result.sections_missing) or "none",
                ),
                ("path", result.report_path),
            ]
        )
        ui.next_step(f"Open it: open {result.report_path}")
    else:
        print(json.dumps(result.summary_dict(), indent=2))

    if args.open_browser:
        import webbrowser

        webbrowser.open(Path(result.report_path).resolve().as_uri())
    return 0


def _require_run(store, *, model: str | None, run_id: str | None):
    if run_id:
        return store.load_run(run_id)
    if model:
        meta = store.latest_run_for_model(model)
        if meta is None:
            raise ValueError(
                f"No run for model {model!r}. First run: odg resolve --model {model}"
            )
        return meta
    runs = store.list_runs()
    if not runs:
        raise ValueError("No runs yet. First run: odg resolve --model …")
    return runs[0]


def _banner(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        1,
        "Resolve model",
        model=model,
        run_id=run_id,
        root=root,
        goal="Turn a model ref into a durable local path + architecture descriptor.",
        bullets=[
            "Choose target quant format (--quant or interactive picker)",
            "Ollama tags → local GGUF by default (no HF login)",
            "Pass --prefer-hf for original BF16 from Hugging Face",
        ],
    )


def _banner_load(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        2,
        "Load model",
        model=model,
        run_id=run_id,
        root=root,
        goal="Open weights and parse header + tensor index.",
        bullets=[
            "Input: Step 01 resolve output (local_path)",
            "For Ollama GGUF: index tensors without dequantizing all weights into RAM",
        ],
    )


def _banner_enumerate(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        3,
        "Enumerate tensors",
        model=model,
        run_id=run_id,
        root=root,
        goal="Flat inventory of every tensor (name, shape, dtype, nbytes).",
        bullets=["No roles yet — that is Step 04"],
    )


def _banner_classify(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        4,
        "Classify tensors",
        model=model,
        run_id=run_id,
        root=root,
        goal="Assign role, depth bucket, group_id, and quantizable flag.",
        bullets=["Roles: attn_q, ffn_up, norm, …"],
    )


def _banner_catalog(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        5,
        "Build tensor catalog",
        model=model,
        run_id=run_id,
        root=root,
        goal="Canonical tensor_catalog.json (names + groups + provenance).",
        bullets=["Merges classify + load + resolve into one durable catalog"],
    )


def _banner_weight_features(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        6,
        "Weight features",
        model=model,
        run_id=run_id,
        root=root,
        goal="Per-tensor / per-group weight statistics that prioritize probes.",
        bullets=["Dequant GGUF tensors as needed; features prioritize, ΔKLD decides"],
    )


def _banner_corpus(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        7,
        "Calibration corpus",
        model=model,
        run_id=run_id,
        root=root,
        goal="Build train / calib / held-out prompt banks with chat template.",
        bullets=["Held-out never used for search — only final validation"],
    )


def _banner_activation_features(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        8,
        "Activation features",
        model=model,
        run_id=run_id,
        root=root,
        goal="Activation / routing stats (or proxy) to refine probe priority.",
        bullets=["Prefers real forward passes when torch + HF weights available"],
    )


def _banner_freeze_gguf(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        9,
        "Freeze reference GGUF",
        model=model,
        run_id=run_id,
        root=root,
        goal="Freeze the reference GGUF used for imatrix / logits / export.",
        bullets=["Promotes Ollama/HF source into a stable model-ref.gguf"],
    )


def _banner_imatrix(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        10,
        "Importance matrix",
        model=model,
        run_id=run_id,
        root=root,
        goal="Collect importance stats for llama-quantize (or proxy).",
        bullets=["Uses llama-imatrix when LLAMA_CPP_DIR is set"],
    )


def _banner_reference_logits(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        11,
        "Reference logits",
        model=model,
        run_id=run_id,
        root=root,
        goal="Cache reference logits for ΔKLD probes and held-out gates.",
        bullets=["Manifest + markers when llama.cpp / runner is missing"],
    )


def _banner_sensitivity(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        12,
        "Sensitivity table",
        model=model,
        run_id=run_id,
        root=root,
        goal="Probe ΔKLD vs Δbytes per group × quant candidate.",
        bullets=["Features prioritize; measured ΔKLD decides"],
    )


def _banner_optimize(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        13,
        "Optimize recipe",
        model=model,
        run_id=run_id,
        root=root,
        goal="Greedy knapsack → recipe.yaml / recipe.tt + Pareto frontier.",
        bullets=["Respects size / quality budgets from sensitivity table"],
    )


def _banner_export(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        14,
        "Export GGUF",
        model=model,
        run_id=run_id,
        root=root,
        goal="Apply recipe via llama-quantize (or dry-run command script).",
        bullets=["Writes candidate GGUF or quantize_command.sh"],
    )


def _banner_validate(model: str, run_id: str, root: str) -> None:
    ui.step_banner(
        15,
        "Validate & release",
        model=model,
        run_id=run_id,
        root=root,
        goal="Tiered gates → RELEASE / PROVISIONAL / FAIL + report card.",
        bullets=["Held-out only — never search/calib for final judgment"],
    )


def _explain(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Kind", result.kind.value),
            ("Working local_path", result.local_path),
            ("Weights ready", result.weights_ready),
            ("Source quantized?", result.source_is_quantized),
            ("Upstream HF (later)", result.hf_repo_id),
        ]
    )
    if result.rejected_quantized_source:
        ui.warn(str(result.rejected_quantized_source))
    d = result.descriptor
    ui.section("architecture descriptor")
    ui.kv(
        [
            ("family", d.family),
            ("layer_count", d.layer_count),
            ("embedding_length", d.embedding_length),
            ("parameter_count", d.parameter_count),
            ("context_length", d.context_length),
            ("specialty_domain", d.specialty_domain),
            ("ollama_quant", d.ollama_quantization),
        ],
        check=False,
    )
    if d.notes:
        ui.notes(d.notes)
    if result.weights_ready and result.local_path:
        msg = "Local weights ready → proceed to Step 02 (inspect / catalog)."
        if result.source_is_quantized:
            msg += (
                "\nSource is already quantized — fine for plumbing; "
                "for real dynamic quant quality later use --prefer-hf BF16."
            )
        ui.next_step(msg)
    else:
        ui.next_step(
            "Weights not ready. For Ollama tags, re-run without --prefer-hf, "
            "or login to Hugging Face and use --prefer-hf --download-weights."
        )


def _explain_load(loaded) -> None:
    ui.section("what happened")
    ui.bullets(loaded.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Backend", loaded.backend),
            ("Source path", loaded.source_path),
            ("Tensors indexed", loaded.n_tensors),
            ("File size", f"{loaded.file_size_bytes:,} bytes"),
            ("Architecture", loaded.architecture),
            ("Layers / embed", f"{loaded.layer_count} / {loaded.embedding_length}"),
            ("Vocab size", loaded.vocab_size),
            ("Quantized source?", loaded.source_is_quantized),
            ("Dtype summary", loaded.dtype_summary),
        ]
    )
    if loaded.sample_tensors:
        ui.section("sample tensors")
        ui.bullets(
            [f"{t.name:40} shape={t.shape}  dtype={t.dtype}" for t in loaded.sample_tensors[:8]]
        )
    if loaded.notes:
        ui.notes(loaded.notes)
    ui.next_step("Step 03 — enumerate/classify every tensor into the catalog.")


def _explain_enumerate(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Tensors", result.n_tensors),
            ("Total elements", f"{result.total_elements:,}"),
            ("Approx nbytes", f"{result.total_nbytes:,}"),
            ("Dtype summary", result.dtype_summary),
        ]
    )
    ui.section("by layer (tensor counts)")
    items = list(result.layer_summary.items())
    rows = [f"layer {k:>6}: {v} tensors" for k, v in items[:6]]
    if len(items) > 8:
        rows.append("…")
        rows.extend(f"layer {k:>6}: {v} tensors" for k, v in items[-2:])
    ui.bullets(rows)
    ui.section("sample (sorted by name)")
    ui.bullets(
        [
            f"{t.name:42} {'×'.join(str(d) for d in t.shape):16} {t.dtype:6}  {t.nbytes:>10,} B"
            for t in result.tensors[:10]
        ]
    )
    ui.next_step("Step 04 — classify each tensor into a role (attn_q, ffn_up, norm, …).")


def _explain_classify(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Tensors", result.n_tensors),
            ("Layers", result.n_layers),
            ("Coverage", f"{result.coverage:.1%} (non-other)"),
            ("Role summary", result.role_summary),
            ("Quantizable", result.quantizable_summary),
            ("Probe groups", len(result.group_summary)),
        ]
    )
    ui.section("groups (role@depth)")
    rows = [f"{gid:28} {n:4} tensors" for gid, n in list(result.group_summary.items())[:20]]
    if len(result.group_summary) > 20:
        rows.append(f"… +{len(result.group_summary) - 20} more")
    ui.bullets(rows)
    ui.section("sample")
    ui.bullets(
        [
            f"[{'Q' if t.quantizable else '—'}] {t.role:12} {t.depth or '-':7}  {t.name}"
            for t in result.tensors[:12]
        ]
    )
    if result.other_names:
        ui.section("unmatched")
        ui.bullets(result.other_names[:10])
    ui.next_step("Step 05 — build tensor_catalog.json (HF/GGUF names + groups).")


def _explain_catalog(catalog) -> None:
    ui.section("what happened")
    ui.bullets(catalog.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Tensors", catalog.n_tensors),
            ("Groups", catalog.n_groups),
            ("Quantizable", catalog.n_quantizable),
            ("Source", catalog.source_path),
            ("Backend", catalog.source_backend),
            ("SHA256", (catalog.source_sha256 or "")[:16] + "…"),
        ]
    )
    ui.next_step("Step 06 — weight features for probe prioritization.")


def _fmt_group_hints(items) -> str:
    out = []
    for item in (items or [])[:5]:
        if isinstance(item, dict):
            out.append(str(item.get("group_id") or item.get("id") or item))
        else:
            out.append(str(item))
    return ", ".join(out) or "-"


def _explain_weight_features(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Tensors featured", f"{result.n_with_features}/{result.n_tensors}"),
            ("Skipped", result.n_skipped),
            ("Source quantized?", result.source_is_quantized),
            ("Hardest groups", _fmt_group_hints(result.hardest_groups)),
            ("Easiest groups", _fmt_group_hints(result.easiest_groups)),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 07 — build calibration / held-out corpus.")


def _explain_corpus(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Corpus id", result.corpus_id),
            ("Docs (calib/search/heldout)", f"{result.n_calib} / {result.n_search} / {result.n_heldout}"),
            ("Tokens est total", f"{result.tokens_est_total:,}"),
            ("Chat template", result.chat_template or "-"),
            ("Domain counts", result.domain_counts),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 08 — activation features on calib docs.")


def _explain_activation_features(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Method", result.method),
            ("Docs used", result.n_docs_used),
            ("Tokens (est)", result.n_tokens_est),
            ("Tensors", f"{result.n_with_features}/{result.n_tensors}"),
            ("Hardest groups", _fmt_group_hints(result.hardest_groups)),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 09 — freeze reference GGUF for imatrix / export.")


def _explain_freeze_gguf(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Method", result.method),
            ("GGUF path", result.gguf_path),
            ("Bytes", f"{result.gguf_nbytes:,}"),
            ("SHA256", (result.gguf_sha256 or "")[:16] + ("…" if result.gguf_sha256 else "")),
            ("BF16 reference?", result.is_bf16_reference),
            ("Tensors", result.n_tensors),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 10 — importance matrix (imatrix).")


def _explain_imatrix(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Method", result.method),
            ("Imatrix path", result.imatrix_path or result.proxy_path or "-"),
            ("Tensors scored", result.n_tensors_scored),
            ("Chunks", result.n_chunks if result.n_chunks is not None else "-"),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 11 — cache reference logits.")


def _explain_reference_logits(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Method", result.method),
            ("Cache key", result.cache_key),
            ("Search logits", result.logits_search_path or "-"),
            ("Held-out logits", result.logits_heldout_path or "-"),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 12 — sensitivity probes (ΔKLD / Δbytes).")


def _explain_sensitivity(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Method", result.method),
            ("Rows", result.n_rows),
            ("Groups probed", result.n_groups_probed),
            ("Baseline", result.baseline_type),
            ("Probe types", ", ".join(result.probe_types) or "-"),
        ]
    )
    ui.notes(result.notes)
    ui.next_step("Step 13 — optimize mixed-precision recipe.")


def _explain_optimize(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.kv(
        [
            ("Method", result.method),
            ("Recipe", result.recipe_path),
            ("Tensor-type file", result.tensor_type_file),
            ("Budget bytes", f"{result.budget_bytes:,}"),
            ("Estimated bytes", f"{result.estimated_bytes:,}"),
            ("Predicted ΔKLD", f"{result.predicted_delta_kld:.6g}"),
            ("Groups", result.n_groups),
        ]
        + (
            [
                ("Tail KLD", f"{result.total_tail_kld:.6g}"),
                ("Certificate", f"{result.certificate['mode']} "
                 f"(probed={result.certificate['probed_columns']} "
                 f"excluded={result.certificate['excluded_columns']})"),
            ]
            if result.certificate
            else []
        )
    )
    ui.notes(result.notes)
    ui.next_step("Step 14 — export candidate GGUF.")


def _explain_export(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    rows = [
        ("Method", result.method),
        ("Output GGUF", result.gguf_out or "(dry-run / missing)"),
    ]
    if result.gguf_out_nbytes:
        rows.append(("Size", f"{result.gguf_out_nbytes / (1024**2):.1f} MiB"))
    if result.estimated_bytes:
        rows.append(("Recipe estimate", f"{result.estimated_bytes / (1024**2):.1f} MiB"))
    rows.append(("Command", " ".join(result.command[:6]) + " …"))
    ui.kv(rows)
    ui.notes(result.notes)
    ui.next_step("Step 15 — validate on held-out; stage release or feedback.")


def _explain_validate(result) -> None:
    ui.section("what happened")
    ui.bullets(result.steps_log)
    ui.section("verdict")
    ui.verdict_badge(result.verdict)
    ui.kv(
        [
            ("Verdict", result.verdict),
            ("Method", result.method),
            ("Tier1 pass", result.tier1.get("pass")),
            ("Tier2 pass", result.tier2.get("pass")),
            ("Report", result.report_path),
            ("Release dir", result.release_dir or "-"),
        ]
    )
    if result.report_card_paths:
        ui.kv(
            [
                ("Report card HTML", result.report_card_paths.get("html")),
                ("Report card MD", result.report_card_paths.get("md")),
            ]
        )
    if result.feedback:
        ui.section("feedback to optimizer")
        ui.bullets([f"{f.get('constraint')}: {f.get('action')}" for f in result.feedback])
    ui.notes(result.notes)
    nxt = "Steps 01–15 complete (plumbing). Install llama.cpp for real quant/KLD."
    if result.report_card_paths.get("html"):
        nxt += f"\nOpen report card: {result.report_card_paths['html']}"
    ui.next_step(nxt)


if __name__ == "__main__":
    raise SystemExit(main())
