"""Same-model / same-benchmark experiment for OpenDynamicGGUF.

The only independent variable is quantization. Model, tokenizer, harness,
tasks, shots, seed, batch size, and sample limit stay pinned.

Default test model: google/functiongemma-270m-it (FunctionGemma 270M).

    BF16 (original HF)
      → Q4_K_M / Q5_K_M / Q6_K  (uniform llama-quantize, same imatrix)
      → OpenDynamicGGUF         (dynamic recipe, same imatrix / frozen GGUF)

Then EleutherAI lm-evaluation-harness on the identical task list.

Quality numbers are never estimated: if lm-eval or a GGUF is missing, the
variant is recorded as skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_bins import find_llama_binary

SCHEMA = "odg/experiment/v1"

# Default test model: local Ollama tag (no Hugging Face download).
DEFAULT_MODEL = "functiongemma:latest"

TASKS: tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "hellaswag",
    "arc_challenge",
    "truthfulqa_mc2",
)

# Primary metric keys, in preference order, matching harness output.
PRIMARY_METRICS: dict[str, tuple[str, ...]] = {
    "mmlu": ("acc,none", "acc"),
    "gsm8k": ("exact_match,strict-match", "exact_match,none", "exact_match"),
    "hellaswag": ("acc_norm,none", "acc_norm", "acc,none"),
    "arc_challenge": ("acc_norm,none", "acc_norm", "acc,none"),
    "truthfulqa_mc2": ("acc,none", "acc"),
}

TASK_LABELS: dict[str, str] = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C",
    "truthfulqa_mc2": "TruthfulQA",
}

# Uniform llama-quantize baselines. ODG is produced by the pipeline, not here.
UNIFORM_TYPES: tuple[tuple[str, str], ...] = (
    ("q4_k_m", "Q4_K_M"),
    ("q5_k_m", "Q5_K_M"),
    ("q6_k", "Q6_K"),
)

VARIANTS: tuple[str, ...] = ("bf16", "q4_k_m", "q5_k_m", "q6_k", "odg")

VARIANT_LABELS: dict[str, str] = {
    "bf16": "BF16",
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q6_k": "Q6_K",
    "odg": "OpenDynamicGGUF",
}


# ---------------------------------------------------------------------------
# Pinned config
# ---------------------------------------------------------------------------


def default_config() -> dict[str, Any]:
    """The experiment pin. ``num_fewshot`` is None → harness per-task defaults."""
    return {
        "schema": SCHEMA,
        "model": DEFAULT_MODEL,
        "tasks": list(TASKS),
        "batch_size": 8,
        "seed": 0,
        "num_fewshot": None,
        "log_samples": True,
        "gguf_backend": "gguf",  # llama-cpp-python; "hf" uses gguf_file=
        "default_suite": "dev",
        "suites": {
            "dev": {
                "limit": 32,
                "description": "Laptop-scale slice for FunctionGemma 270M testing.",
            },
            "paper": {
                "limit": None,
                "description": "Full tasks, harness default shots, no sample cap.",
            },
        },
        "imatrix_chunks": 16,
        "variants": list(VARIANTS),
    }


def load_config(root: Path) -> dict[str, Any]:
    cfg = default_config()
    path = Path(root) / "config.json"
    if path.is_file():
        overlay = json.loads(path.read_text())
        if not isinstance(overlay, dict):
            raise ValueError(f"Invalid config: {path}")
        cfg.update({k: v for k, v in overlay.items() if k != "suites"})
        if isinstance(overlay.get("suites"), dict):
            cfg["suites"] = {**cfg["suites"], **overlay["suites"]}
    return cfg


def write_default_config(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    if not path.is_file():
        path.write_text(json.dumps(default_config(), indent=2) + "\n")
    return path


def config_fingerprint(cfg: dict[str, Any], *, suite: str, limit: int | None) -> str:
    pin = {
        "model": cfg["model"],
        "tasks": list(cfg["tasks"]),
        "batch_size": cfg["batch_size"],
        "seed": cfg["seed"],
        "num_fewshot": cfg["num_fewshot"],
        "suite": suite,
        "limit": limit,
        "gguf_backend": cfg.get("gguf_backend"),
    }
    blob = json.dumps(pin, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def suite_limit(cfg: dict[str, Any], suite: str) -> int | None:
    suites = cfg.get("suites") or {}
    if suite not in suites:
        known = ", ".join(sorted(suites))
        raise ValueError(f"Unknown suite {suite!r}. Suites: {known}")
    return suites[suite].get("limit")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_root() -> Path:
    return Path(__file__).resolve().parent / "benchmark"


def models_dir(root: Path) -> Path:
    return Path(root) / "models"


def results_dir(root: Path) -> Path:
    return Path(root) / "results"


def hf_dir(root: Path) -> Path:
    return models_dir(root) / "hf"


def variant_gguf(root: Path, variant: str) -> Path:
    names = {
        "bf16": "functiongemma-270m-bf16.gguf",
        "q4_k_m": "functiongemma-270m-q4_k_m.gguf",
        "q5_k_m": "functiongemma-270m-q5_k_m.gguf",
        "q6_k": "functiongemma-270m-q6_k.gguf",
        "odg": "functiongemma-270m-odg.gguf",
    }
    if variant not in names:
        raise ValueError(f"Unknown variant {variant!r}")
    return models_dir(root) / names[variant]


def variant_result_dir(root: Path, variant: str) -> Path:
    return results_dir(root) / variant


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Parameter count / compression
# ---------------------------------------------------------------------------


def n_params_from_safetensors(path: Path) -> int:
    """Count parameters from a safetensors header (no tensor load)."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    total = 0
    for key, meta in header.items():
        if key == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape") or []
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n
    return total


def n_params_from_hf(model_dir: Path) -> int | None:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None
    # sharded: each tensor lives in exactly one file
    total = 0
    for p in files:
        total += n_params_from_safetensors(p)
    return total or None


def compression_stats(
    *,
    nbytes: int | None,
    n_params: int | None,
    bf16_bytes: int | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bytes": nbytes,
        "gb": None if nbytes is None else round(nbytes / (1024**3), 4),
        "mb": None if nbytes is None else round(nbytes / (1024**2), 2),
        "n_params": n_params,
        "bytes_per_parameter": None,
        "compression_ratio_vs_bf16": None,
    }
    if nbytes and n_params:
        out["bytes_per_parameter"] = round(nbytes / n_params, 4)
    if nbytes and bf16_bytes:
        out["compression_ratio_vs_bf16"] = round(bf16_bytes / nbytes, 3)
    return out


# ---------------------------------------------------------------------------
# Device / backend
# ---------------------------------------------------------------------------


def detect_device() -> str:
    env = os.environ.get("ODG_EXPERIMENT_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def lm_eval_available() -> tuple[bool, str | None]:
    try:
        import lm_eval  # type: ignore[import-not-found]

        return True, getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        return False, None


# ---------------------------------------------------------------------------
# Eval command — the pin lives here
# ---------------------------------------------------------------------------


def _gguf_model_args(gguf: Path, device: str) -> str:
    n_gpu = 0 if device == "cpu" else -1
    return f"model={gguf},n_gpu_layers={n_gpu}"


def build_eval_command(
    *,
    variant: str,
    cfg: dict[str, Any],
    root: Path,
    out_dir: Path,
    limit: int | None,
    device: str,
    gguf_backend: str | None = None,
) -> list[str]:
    """
    Build the lm-eval argv. Identical for every variant except model_args.

    ``num_fewshot`` is intentionally omitted so each task keeps the harness
    default (MMLU 5, GSM8K 5, HellaSwag 10, ARC-C 25, TruthfulQA 0).
    """
    tasks = ",".join(cfg["tasks"])
    backend = gguf_backend or cfg.get("gguf_backend") or "gguf"
    hf = hf_dir(root)
    has_hf = (hf / "config.json").is_file()
    gguf = variant_gguf(root, variant)

    if variant == "bf16" and has_hf and backend == "hf":
        model = "hf"
        model_args = (
            f"pretrained={hf},tokenizer={hf},dtype=auto,trust_remote_code=True"
        )
    elif backend == "hf":
        tok = str(hf) if has_hf else str(gguf.parent)
        model = "hf"
        model_args = (
            f"pretrained={gguf.parent},gguf_file={gguf.name},"
            f"tokenizer={tok},trust_remote_code=True"
        )
    else:
        model = "gguf"
        model_args = _gguf_model_args(gguf, device)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        model,
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        str(cfg["batch_size"]),
        "--seed",
        str(cfg["seed"]),
        "--output_path",
        str(out_dir),
        "--device",
        device,
    ]
    if cfg.get("log_samples"):
        cmd.append("--log_samples")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


# ---------------------------------------------------------------------------
# Local source (Ollama / GGUF path) — no Hugging Face
# ---------------------------------------------------------------------------


def _link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def local_ollama_gguf(tag: str) -> Path | None:
    """Resolve an Ollama blob from the on-disk manifest. Does not call `ollama`."""
    try:
        from resolve import _digest_to_blob, default_ollama_root, find_manifest
    except ImportError:
        return None
    try:
        manifest_path = find_manifest(tag)
        payload = json.loads(manifest_path.read_text())
    except (OSError, FileNotFoundError, json.JSONDecodeError, ValueError):
        return None
    root = default_ollama_root()
    for layer in payload.get("layers") or []:
        if "image.model" not in str(layer.get("mediaType") or ""):
            continue
        blob = _digest_to_blob(root, layer["digest"])
        if blob.is_file():
            return blob
    return None


def _existing_freeze_gguf(model: str) -> Path | None:
    slug = model.replace(":", "-")
    repo = Path(__file__).resolve().parent / "artifacts" / "runs"
    if not repo.is_dir():
        return None
    matches = sorted(
        repo.glob(f"*{slug}*/steps/09_freeze_gguf/model-ref.gguf"),
        reverse=True,
    )
    for p in matches:
        if p.is_file():
            return p
    return None


def locate_source_gguf(model: str) -> Path | None:
    """Find a local GGUF: explicit path, Ollama blob, or a previous freeze step."""
    p = Path(model).expanduser()
    if p.is_file() and p.suffix == ".gguf":
        return p
    blob = local_ollama_gguf(model if ":" in model else f"{model}:latest")
    if blob is not None:
        return blob
    return _existing_freeze_gguf(model)


def _source_dtype_label(gguf: Path) -> str:
    try:
        from load import open_gguf

        summary = open_gguf(gguf).get("dtype_summary") or {}
    except Exception:  # noqa: BLE001
        return "GGUF"
    weight = {k: v for k, v in summary.items() if k != "F32"}
    if not weight:
        return "F32"
    return max(weight, key=weight.get)


def _is_bf16_family(gguf: Path) -> bool:
    try:
        from load import open_gguf

        summary = open_gguf(gguf).get("dtype_summary") or {}
    except Exception:  # noqa: BLE001
        return False
    allowed = {"BF16", "F16", "F32"}
    return all(k in allowed for k in summary) and (
        int(summary.get("BF16") or 0) + int(summary.get("F16") or 0) > 0
    )


def _n_params_from_gguf(gguf: Path) -> int | None:
    try:
        from load import open_gguf

        n = open_gguf(gguf).get("parameter_count")
        return int(n) if n else None
    except Exception:  # noqa: BLE001
        return None


def _find_recipe_tt(root: Path) -> Path | None:
    bases = [
        Path(root) / "artifacts" / "runs",
        Path(__file__).resolve().parent / "artifacts" / "runs",
    ]
    for base in bases:
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*/steps/14_export/recipe.tt"), reverse=True):
            if p.is_file():
                return p
    return None


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------


def _download_hf(model: str, dest: Path, *, force: bool) -> dict[str, Any]:
    dest = Path(dest)
    log: list[str] = []
    if (dest / "config.json").is_file() and not force:
        n = n_params_from_hf(dest)
        log.append(f"HF snapshot already present: {dest}")
        return {"path": str(dest), "n_params": n, "skipped": True, "log": log}

    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    log.append(f"Downloading {model} → {dest}")
    local = snapshot_download(
        repo_id=model,
        local_dir=str(dest),
        token=token,
        ignore_patterns=["*.gguf", "*.bin.gz"],
    )
    n = n_params_from_hf(Path(local))
    log.append(f"Downloaded. n_params={n}")
    return {"path": str(Path(local)), "n_params": n, "skipped": False, "log": log}


def _convert_bf16(hf: Path, outfile: Path, *, force: bool) -> dict[str, Any]:
    from freeze import convert_hf_to_bf16_gguf, find_convert_script

    log: list[str] = []
    if outfile.is_file() and not force:
        log.append(f"BF16 GGUF already present: {outfile.name}")
        return {
            "path": str(outfile),
            "bytes": outfile.stat().st_size,
            "skipped": True,
            "log": log,
        }
    script = find_convert_script()
    if script is None:
        return {
            "path": None,
            "skipped": True,
            "reason": "convert_hf_to_gguf.py not found (set LLAMA_CPP_DIR)",
            "log": log,
        }
    log.append(f"Converting HF → BF16 GGUF via {script}")
    last_err: Exception | None = None
    for outtype in ("bf16", "f16"):
        try:
            convert_hf_to_bf16_gguf(
                hf_dir=hf, outfile=outfile, convert_script=script, outtype=outtype
            )
            log.append(f"Wrote {outfile.name} outtype={outtype}")
            return {
                "path": str(outfile),
                "bytes": outfile.stat().st_size,
                "outtype": outtype,
                "skipped": False,
                "log": log,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.append(f"outtype={outtype} failed: {exc}")
    return {
        "path": None,
        "skipped": True,
        "reason": f"HF→GGUF conversion failed: {last_err}",
        "log": log,
    }


def _build_imatrix(
    bf16: Path, calib: Path, outfile: Path, *, chunks: int, force: bool
) -> dict[str, Any]:
    from imatrix import find_llama_imatrix, run_llama_imatrix

    log: list[str] = []
    if outfile.is_file() and not force:
        log.append(f"imatrix already present: {outfile.name}")
        return {"path": str(outfile), "skipped": True, "log": log}
    binary = find_llama_imatrix()
    if binary is None:
        return {
            "path": None,
            "skipped": True,
            "reason": "llama-imatrix not found",
            "log": log,
        }
    if not calib.is_file():
        return {
            "path": None,
            "skipped": True,
            "reason": f"calib text missing: {calib}",
            "log": log,
        }
    log.append(f"Running llama-imatrix chunks={chunks}")
    try:
        run_llama_imatrix(
            binary=binary,
            model_gguf=bf16,
            calib_txt=calib,
            outfile=outfile,
            n_chunks=chunks,
        )
    except Exception as exc:  # noqa: BLE001
        return {"path": None, "skipped": True, "reason": str(exc), "log": log}
    return {
        "path": str(outfile),
        "bytes": outfile.stat().st_size,
        "skipped": False,
        "log": log,
    }


def _quantize_uniform(
    src: Path,
    outfile: Path,
    ftype: str,
    *,
    imatrix: Path | None,
    force: bool,
    allow_requantize: bool = False,
) -> dict[str, Any]:
    log: list[str] = []
    if outfile.is_file() and not force:
        log.append(f"{outfile.name} already present")
        return {
            "path": str(outfile),
            "bytes": outfile.stat().st_size,
            "skipped": True,
            "log": log,
        }
    binary = find_llama_binary("llama-quantize")
    if binary is None:
        return {
            "path": None,
            "skipped": True,
            "reason": "llama-quantize not found (set LLAMA_CPP_DIR)",
            "log": log,
        }
    cmd = [str(binary)]
    if allow_requantize:
        cmd.append("--allow-requantize")
        log.append("Source is already quantized — llama-quantize --allow-requantize")
    if imatrix is not None and imatrix.is_file():
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(src), str(outfile), ftype]
    log.append("Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not outfile.is_file():
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return {
            "path": None,
            "skipped": True,
            "reason": f"llama-quantize {ftype} failed: {tail}",
            "log": log,
        }
    return {
        "path": str(outfile),
        "bytes": outfile.stat().st_size,
        "command": cmd,
        "skipped": False,
        "log": log,
    }


def _prepare_odg(
    *,
    model: str,
    root: Path,
    dest: Path,
    gguf_in: Path,
    force: bool,
    allow_requantize: bool = False,
) -> dict[str, Any]:
    log: list[str] = []
    if dest.is_file() and not force:
        log.append(f"ODG GGUF already present: {dest.name}")
        return {
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "skipped": True,
            "log": log,
        }

    recipe = _find_recipe_tt(root)
    binary = find_llama_binary("llama-quantize")
    if recipe is not None and binary is not None and gguf_in.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        clean = dest.parent / "recipe.tt"
        lines = [
            ln.strip()
            for ln in recipe.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        clean.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [str(binary)]
        if allow_requantize:
            cmd.append("--allow-requantize")
        cmd += [
            "--tensor-type-file",
            str(clean),
            str(gguf_in),
            str(dest),
            "Q4_K_M",
        ]
        log.append(f"ODG from existing recipe {recipe} (no Hugging Face)")
        log.append("Running: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dest.is_file():
            return {
                "path": str(dest),
                "bytes": dest.stat().st_size,
                "recipe": str(recipe),
                "command": cmd,
                "skipped": False,
                "log": log,
            }
        log.append(
            "recipe quantize failed: "
            + (proc.stderr or proc.stdout or "")[-800:]
        )
        return {
            "path": None,
            "skipped": True,
            "reason": "llama-quantize --tensor-type-file failed on the local recipe",
            "log": log,
        }

    artifacts = Path(root) / "artifacts"
    log.append(f"Falling back to odg run (local, no --prefer-hf) → {artifacts}")
    try:
        from cli import main as odg_main
        from store import RunStore
    except ImportError as exc:
        return {
            "path": None,
            "skipped": True,
            "reason": f"odg CLI import failed: {exc}",
            "log": log,
        }

    argv = [
        "--artifacts",
        str(artifacts),
        "run",
        "--model",
        model,
        "--quant",
        "q4_k_m",
        "--no-ask",
        "--quiet",
    ]
    if force:
        argv.append("--new-run")
    rc = odg_main(argv)
    if rc not in (0, None):
        log.append(f"odg run exited {rc}")

    store = RunStore(artifacts)
    meta = store.latest_run_for_model(model)
    if meta is None:
        return {
            "path": None,
            "skipped": True,
            "reason": "odg run produced no CURRENT run",
            "log": log,
        }
    export_out = store.read_step_output(meta.run_id, "export") or {}
    gguf_out = export_out.get("gguf_out")
    if not gguf_out or not Path(gguf_out).is_file():
        return {
            "path": None,
            "skipped": True,
            "reason": (
                "ODG export was dry_run (no GGUF). Install llama-quantize and "
                f"re-run: odg --artifacts {artifacts} export --model {model} "
                "--mode llama --force"
            ),
            "run_id": meta.run_id,
            "log": log,
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gguf_out, dest)
    log.append(f"Copied {gguf_out} → {dest}")
    return {
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "run_id": meta.run_id,
        "source": gguf_out,
        "skipped": False,
        "log": log,
    }


def prepare(
    root: Path,
    *,
    model: str | None = None,
    force: bool = False,
    skip_odg: bool = False,
) -> dict[str, Any]:
    """Copy the local source GGUF, emit uniform Q4/Q5/Q6 + ODG. No Hugging Face."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_default_config(root)
    cfg = load_config(root)
    model = model or cfg["model"]
    models_dir(root).mkdir(parents=True, exist_ok=True)

    steps: dict[str, Any] = {}
    notes: list[str] = []

    src = locate_source_gguf(model)
    if src is None:
        raise RuntimeError(
            f"No local GGUF for {model!r}. Pull it with Ollama "
            f"(`ollama pull {model}`) or pass a path to a .gguf file. "
            "This experiment does not download from Hugging Face."
        )

    ref_path = variant_gguf(root, "bf16")
    if force or not ref_path.is_file():
        how = _link_or_copy(src, ref_path)
        notes.append(f"Linked local source {src} → {ref_path.name} via {how}")
    else:
        notes.append(f"Source GGUF already present: {ref_path.name}")
    dtype_label = _source_dtype_label(ref_path)
    bf16_family = _is_bf16_family(ref_path)
    allow_requantize = not bf16_family
    n_params = _n_params_from_gguf(ref_path)
    steps["source"] = {
        "path": str(src),
        "linked": str(ref_path),
        "dtype": dtype_label,
        "bf16_family": bf16_family,
        "n_params": n_params,
        "bytes": ref_path.stat().st_size,
    }
    if allow_requantize:
        notes.append(
            f"Local source is {dtype_label}, not BF16. Uniform/ODG GGUFs are "
            "requantized from this file (--allow-requantize). That is a plumbing "
            "run, not a paper-quality BF16 comparison."
        )

    imatrix_path = models_dir(root) / "imatrix.gguf"
    calib = root / "calib.txt"
    imatrix = _build_imatrix(
        ref_path,
        calib,
        imatrix_path,
        chunks=int(cfg.get("imatrix_chunks") or 16),
        force=force,
    )
    steps["imatrix"] = imatrix
    notes.extend(imatrix.get("log") or [])
    if imatrix.get("reason"):
        notes.append(str(imatrix["reason"]))
    imatrix_file = Path(imatrix["path"]) if imatrix.get("path") else None

    uniforms: dict[str, Any] = {}
    for vid, ftype in UNIFORM_TYPES:
        uniforms[vid] = _quantize_uniform(
            ref_path,
            variant_gguf(root, vid),
            ftype,
            imatrix=imatrix_file,
            force=force,
            allow_requantize=allow_requantize,
        )
        notes.extend(uniforms[vid].get("log") or [])
        if uniforms[vid].get("reason"):
            notes.append(str(uniforms[vid]["reason"]))
    steps["uniform"] = uniforms

    if skip_odg:
        steps["odg"] = {"skipped": True, "reason": "--skip-odg"}
        notes.append("ODG GGUF skipped (--skip-odg). Produce it later with run_odg.sh")
    else:
        steps["odg"] = _prepare_odg(
            model=model,
            root=root,
            dest=variant_gguf(root, "odg"),
            gguf_in=ref_path,
            force=force,
            allow_requantize=allow_requantize,
        )
        notes.extend(steps["odg"].get("log") or [])
        if steps["odg"].get("reason"):
            notes.append(str(steps["odg"]["reason"]))

    ref_bytes = ref_path.stat().st_size
    sizes: dict[str, Any] = {}
    for vid in VARIANTS:
        p = variant_gguf(root, vid)
        nbytes = p.stat().st_size if p.is_file() else None
        sizes[vid] = {
            "kind": "gguf",
            "path": str(p) if p.is_file() else None,
            "present": bool(nbytes),
            **compression_stats(
                nbytes=nbytes, n_params=n_params, bf16_bytes=ref_bytes
            ),
        }

    manifest = {
        "schema": SCHEMA,
        "model": model,
        "created_at": _utc_now(),
        "n_params": n_params,
        "bf16_bytes": ref_bytes,
        "reference_label": dtype_label if not bf16_family else "BF16",
        "source_is_quantized": allow_requantize,
        "steps": {
            k: {kk: vv for kk, vv in v.items() if kk != "log"}
            if isinstance(v, dict)
            else v
            for k, v in steps.items()
        },
        "sizes": sizes,
        "notes": notes,
        "config": {
            "tasks": cfg["tasks"],
            "batch_size": cfg["batch_size"],
            "seed": cfg["seed"],
            "num_fewshot": cfg["num_fewshot"],
            "gguf_backend": cfg.get("gguf_backend"),
        },
    }
    if isinstance(manifest["steps"].get("uniform"), dict):
        manifest["steps"]["uniform"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "log"}
            for k, v in uniforms.items()
        }
    _write_json(root / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------


def _find_harness_results(out_dir: Path) -> dict[str, Any] | None:
    if not out_dir.is_dir():
        return None
    candidates: list[Path] = []
    for p in out_dir.rglob("*.json"):
        if "samples_" in p.name:
            continue
        candidates.append(p)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "results" in data:
            data["_path"] = str(p)
            return data
    return None


def pick_metric(
    task_id: str, metrics: dict[str, Any]
) -> tuple[str, float] | None:
    if not isinstance(metrics, dict):
        return None
    for key in PRIMARY_METRICS.get(task_id, ("acc,none", "acc")):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return key, float(val)
    for key, val in metrics.items():
        if key.endswith("_stderr,none") or "stderr" in key:
            continue
        if isinstance(val, (int, float)) and not key.startswith("alias"):
            return key, float(val)
    return None


def extract_quality(raw: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    results = raw.get("results") or {}
    groups = raw.get("groups") or {}
    out_tasks: dict[str, Any] = {}
    for task_id in tasks:
        metrics = groups.get(task_id) or results.get(task_id) or {}
        picked = pick_metric(task_id, metrics)
        out_tasks[task_id] = {
            "score": None if picked is None else picked[1],
            "metric": None if picked is None else picked[0],
            "metrics": metrics,
        }
    return {
        "skipped": False,
        "harness": {
            "name": "lm-eval",
            "version": (raw.get("config") or {}).get("model")
            and raw.get("config", {}).get("pretty_env_info"),
        },
        "n_shot": raw.get("n-shot") or raw.get("n_shot"),
        "versions": raw.get("versions"),
        "tasks": out_tasks,
        "source": raw.get("_path"),
    }


def run_quality(
    *,
    variant: str,
    cfg: dict[str, Any],
    root: Path,
    out_dir: Path,
    limit: int | None,
    device: str,
    gguf_backend: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    log: list[str] = []
    notes: list[str] = []
    ok, version = lm_eval_available()
    if not ok:
        notes.append(
            "lm-eval not installed — quality skipped "
            "(pip install 'lm-eval[hf]' transformers torch)"
        )
        return (
            {"skipped": True, "reason": "lm-eval-harness not installed", "tasks": {}},
            log,
            notes,
        )

    gguf = variant_gguf(root, variant)
    if not gguf.is_file():
        notes.append(f"{VARIANT_LABELS[variant]} GGUF missing: {gguf}")
        return (
            {
                "skipped": True,
                "reason": f"GGUF not found: {gguf}",
                "tasks": {},
            },
            log,
            notes,
        )

    harness_dir = out_dir / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_eval_command(
        variant=variant,
        cfg=cfg,
        root=root,
        out_dir=harness_dir,
        limit=limit,
        device=device,
        gguf_backend=gguf_backend,
    )
    log.append(f"lm-eval {version}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "lm_eval.log").write_text(
        (proc.stdout or "") + "\n" + (proc.stderr or ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        reason = f"lm-eval failed (exit {proc.returncode}): {' / '.join(tail)}"
        notes.append(reason)
        notes.append(
            "GGUF eval in lm-eval is known to mismatch HF scores in some "
            "setups (EleutherAI/lm-evaluation-harness#2887). Do not treat a "
            "single GGUF number as authoritative."
        )
        return (
            {"skipped": True, "reason": reason, "tasks": {}, "command": cmd},
            log,
            notes,
        )

    raw = _find_harness_results(harness_dir)
    if raw is None:
        notes.append("lm-eval exited 0 but no results JSON was found")
        return (
            {
                "skipped": True,
                "reason": "no harness results JSON",
                "tasks": {},
                "command": cmd,
            },
            log,
            notes,
        )
    quality = extract_quality(raw, list(cfg["tasks"]))
    quality["harness"] = {"name": "lm-eval", "version": version}
    quality["command"] = cmd
    quality["limit"] = limit
    for task_id, entry in quality["tasks"].items():
        log.append(f"  {task_id}: {entry.get('metric')}={entry.get('score')}")
    return quality, log, notes


def parse_perplexity_log(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    patterns = {
        "perplexity": r"(?:Final estimate:\s*)?PPL\s*=\s*([\d.]+)",
        "kl_divergence": r"Mean\s+KL(?:\s+divergence)?\s*[:=]\s*([\d.]+)",
        "token_agreement": r"(?:Mean\s+)?(?:n_same_top|Same top|token accuracy)\s*[:=]\s*([\d.]+)",
        "delta_ppl": r"Mean\s+ΔPPL\s*[:=]\s*([-\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))
    return out


def run_behavior(
    *,
    variant: str,
    root: Path,
    out_dir: Path,
    bf16_gguf: Path | None,
    kl_base: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    """Perplexity + KL / token agreement via llama-perplexity (honest skip)."""
    log: list[str] = []
    heldout = Path(root) / "heldout.txt"
    binary = find_llama_binary("llama-perplexity")
    if binary is None:
        log.append("llama-perplexity not found — behavior skipped")
        return {"skipped": True, "reason": "llama-perplexity not found"}, log
    if not heldout.is_file():
        log.append("heldout.txt missing — behavior skipped")
        return {"skipped": True, "reason": "heldout.txt missing"}, log

    if variant == "bf16":
        model = bf16_gguf
    else:
        model = variant_gguf(root, variant)
    if model is None or not model.is_file():
        return {"skipped": True, "reason": f"no GGUF for {variant}"}, log

    cmd = [str(binary), "-m", str(model), "-f", str(heldout)]
    if kl_base is not None and kl_base.is_file() and variant != "bf16":
        cmd += ["--kl-divergence", str(kl_base)]
    log.append("Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (out_dir / "perplexity.log").write_text(text, encoding="utf-8")
    parsed = parse_perplexity_log(text)
    if proc.returncode != 0 and not parsed:
        log.append(f"llama-perplexity failed (exit {proc.returncode})")
        return {"skipped": True, "reason": "llama-perplexity failed", "log_tail": text[-800:]}, log
    parsed["skipped"] = False
    parsed["command"] = cmd
    return parsed, log


def dump_kl_base(bf16_gguf: Path, heldout: Path, outfile: Path) -> Path | None:
    from logits import find_llama_perplexity, run_kl_divergence_base

    binary = find_llama_perplexity()
    if binary is None or not bf16_gguf.is_file() or not heldout.is_file():
        return None
    if outfile.is_file():
        return outfile
    try:
        run_kl_divergence_base(
            binary=binary, model_gguf=bf16_gguf, text_file=heldout, outfile=outfile
        )
    except Exception:  # noqa: BLE001
        return None
    return outfile if outfile.is_file() else None


def parse_llama_bench_json(stdout: str) -> dict[str, Any] | None:
    """Extract pp/tg tokens-per-second from ``llama-bench -o json`` output."""
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        entries = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None
    out: dict[str, Any] = {"measured": True, "tool": "llama-bench"}
    for e in entries:
        if not isinstance(e, dict):
            continue
        tps = e.get("avg_ts")
        if tps is None:
            continue
        if int(e.get("n_prompt") or 0) > 0 and int(e.get("n_gen") or 0) == 0:
            out["pp_tps"] = round(float(tps), 2)
        elif int(e.get("n_gen") or 0) > 0:
            out["tg_tps"] = round(float(tps), 2)
        if "backend" not in out and e.get("backends"):
            out["backend"] = e["backends"]
        if "model_type" not in out and e.get("model_type"):
            out["model_type"] = e["model_type"]
    return out if ("pp_tps" in out or "tg_tps" in out) else None


def run_llama_bench(
    gguf_path: Path,
    *,
    llama_bench: Path | str | None = None,
    n_prompt: int = 512,
    n_gen: int = 128,
    timeout_s: int = 1800,
) -> tuple[dict[str, Any] | None, list[str]]:
    log: list[str] = []
    binary = find_llama_binary("llama-bench", llama_bench)
    if binary is None:
        log.append("llama-bench not found (set LLAMA_CPP_DIR) — throughput skipped")
        return None, log
    cmd = [
        str(binary),
        "-m",
        str(gguf_path),
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-o",
        "json",
    ]
    log.append("Running: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.append(f"llama-bench timed out after {timeout_s}s — throughput skipped")
        return None, log
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        log.append(f"llama-bench failed (exit {proc.returncode}): {' / '.join(tail)}")
        return None, log
    result = parse_llama_bench_json(proc.stdout)
    if result is None:
        log.append("Could not parse llama-bench JSON output — throughput skipped")
        return None, log
    log.append(
        f"Throughput: pp {result.get('pp_tps', '-')} t/s · tg {result.get('tg_tps', '-')} t/s"
    )
    return result, log


def run_inference(gguf: Path | None, out_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if gguf is None or not gguf.is_file():
        return None, ["no GGUF — throughput skipped"]
    result, log = run_llama_bench(gguf)
    if result is not None:
        _write_json(out_dir / "throughput.json", result)
    return result, log


def file_size_payload(
    variant: str, root: Path, n_params: int | None, bf16_bytes: int | None
) -> dict[str, Any]:
    gguf = variant_gguf(root, variant)
    nbytes = gguf.stat().st_size if gguf.is_file() else None
    extra = {"gguf_path": str(gguf) if gguf.is_file() else None}
    if variant == "bf16" and nbytes is None:
        hf = hf_dir(root)
        nbytes = sum(p.stat().st_size for p in hf.glob("*.safetensors")) or None
    stats = compression_stats(nbytes=nbytes, n_params=n_params, bf16_bytes=bf16_bytes)
    stats.update(extra)
    return stats


def run_variant(
    variant: str,
    root: Path,
    *,
    suite: str = "dev",
    limit_override: int | None = None,
    gguf_backend: str | None = None,
    force: bool = False,
    skip_quality: bool = False,
    skip_behavior: bool = False,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Variants: {', '.join(VARIANTS)}")

    root = Path(root)
    cfg = load_config(root)
    limit = suite_limit(cfg, suite) if limit_override is None else limit_override
    device = detect_device()
    fp = config_fingerprint(cfg, suite=suite, limit=limit)
    out_dir = variant_result_dir(root, variant)
    result_path = out_dir / "result.json"
    if result_path.is_file() and not force:
        existing = json.loads(result_path.read_text())
        existing.setdefault("notes", []).append("Reused existing result.json (--force to rerun)")
        return existing

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mp = root / "manifest.json"
    if mp.is_file():
        manifest = json.loads(mp.read_text())
    n_params = manifest.get("n_params")
    bf16_bytes = manifest.get("bf16_bytes")
    label = VARIANT_LABELS[variant]
    if variant == "bf16" and manifest.get("reference_label"):
        label = str(manifest["reference_label"])

    notes: list[str] = []
    log: list[str] = [
        f"variant={variant} suite={suite} limit={limit} device={device} pin={fp}"
    ]

    if skip_quality:
        quality: dict[str, Any] = {"skipped": True, "reason": "--skip-quality", "tasks": {}}
        notes.append("Quality skipped (--skip-quality)")
    else:
        quality, qlog, qnotes = run_quality(
            variant=variant,
            cfg=cfg,
            root=root,
            out_dir=out_dir,
            limit=limit,
            device=device,
            gguf_backend=gguf_backend,
        )
        log.extend(qlog)
        notes.extend(qnotes)

    bf16_gguf = variant_gguf(root, "bf16")
    bf16_gguf_ok = bf16_gguf if bf16_gguf.is_file() else None
    kl_base = None
    if not skip_behavior and bf16_gguf_ok is not None:
        kl_base = dump_kl_base(
            bf16_gguf_ok, root / "heldout.txt", models_dir(root) / "logits-heldout.bin"
        )

    if skip_behavior:
        behavior: dict[str, Any] = {"skipped": True, "reason": "--skip-behavior"}
    else:
        behavior, blog = run_behavior(
            variant=variant,
            root=root,
            out_dir=out_dir,
            bf16_gguf=bf16_gguf_ok,
            kl_base=kl_base,
        )
        log.extend(blog)

    gguf_for_bench = bf16_gguf_ok if variant == "bf16" else variant_gguf(root, variant)
    throughput, tlog = run_inference(gguf_for_bench, out_dir)
    log.extend(tlog)

    payload = {
        "schema": SCHEMA,
        "variant": variant,
        "label": label,
        "model": cfg["model"],
        "suite": suite,
        "limit": limit,
        "device": device,
        "config_fingerprint": fp,
        "created_at": _utc_now(),
        "quality": quality,
        "behavior": behavior,
        "throughput": throughput,
        "memory": file_size_payload(variant, root, n_params, bf16_bytes),
        "notes": notes,
        "log": log,
    }
    _write_json(result_path, payload)
    payload["result_path"] = str(result_path)
    return payload


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def load_variant_results(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for vid in VARIANTS:
        p = variant_result_dir(root, vid) / "result.json"
        if p.is_file():
            out[vid] = json.loads(p.read_text())
    return out


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    if 0.0 <= value <= 1.0:
        return f"{value * 100:.1f}"
    return f"{value:.3f}"


def _fmt_mb(nbytes: int | float | None) -> str:
    if not nbytes:
        return "—"
    return f"{float(nbytes) / (1024**2):.1f} MB"


def _best_quantized(scores: dict[str, float | None]) -> str | None:
    quantized = {
        k: v
        for k, v in scores.items()
        if k != "bf16" and isinstance(v, (int, float))
    }
    if not quantized:
        return None
    return max(quantized, key=lambda k: quantized[k])


def build_comparison(root: Path) -> dict[str, Any]:
    root = Path(root)
    cfg = load_config(root)
    results = load_variant_results(root)
    manifest = {}
    if (root / "manifest.json").is_file():
        manifest = json.loads((root / "manifest.json").read_text())

    fingerprints = {v: r.get("config_fingerprint") for v, r in results.items()}
    unique_fp = {fp for fp in fingerprints.values() if fp}
    pin_ok = len(unique_fp) <= 1

    quality_rows: list[dict[str, Any]] = []
    for task_id in cfg["tasks"]:
        scores: dict[str, float | None] = {}
        metric_name = None
        for vid in VARIANTS:
            entry = ((results.get(vid) or {}).get("quality") or {}).get("tasks") or {}
            t = entry.get(task_id) or {}
            scores[vid] = t.get("score")
            metric_name = metric_name or t.get("metric")
        quality_rows.append(
            {
                "task": task_id,
                "label": TASK_LABELS.get(task_id, task_id),
                "metric": metric_name,
                "scores": scores,
                "best_quantized": _best_quantized(scores),
            }
        )

    compression_rows: list[dict[str, Any]] = []
    for vid in VARIANTS:
        mem = (results.get(vid) or {}).get("memory") or (manifest.get("sizes") or {}).get(vid) or {}
        compression_rows.append(
            {
                "variant": vid,
                "label": (results.get(vid) or {}).get("label") or VARIANT_LABELS[vid],
                "mb": mem.get("mb"),
                "bytes": mem.get("bytes") or mem.get("gguf_bytes"),
                "bytes_per_parameter": mem.get("bytes_per_parameter"),
                "compression_ratio_vs_bf16": mem.get("compression_ratio_vs_bf16"),
                "present": bool(mem.get("bytes") or mem.get("gguf_bytes") or mem.get("present")),
            }
        )

    behavior_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for vid in VARIANTS:
        r = results.get(vid) or {}
        b = r.get("behavior") or {}
        behavior_rows.append(
            {
                "variant": vid,
                "label": (results.get(vid) or {}).get("label") or VARIANT_LABELS[vid],
                "perplexity": None if b.get("skipped") else b.get("perplexity"),
                "kl_divergence": None if b.get("skipped") else b.get("kl_divergence"),
                "token_agreement": None if b.get("skipped") else b.get("token_agreement"),
                "skipped": bool(b.get("skipped")),
            }
        )
        tp = r.get("throughput") or {}
        inference_rows.append(
            {
                "variant": vid,
                "label": (results.get(vid) or {}).get("label") or VARIANT_LABELS[vid],
                "pp_tps": tp.get("pp_tps"),
                "tg_tps": tp.get("tg_tps"),
                "backend": tp.get("backend"),
                "skipped": not bool(tp),
            }
        )

    odg_size = next((r["bytes"] for r in compression_rows if r["variant"] == "odg"), None)
    q4_size = next((r["bytes"] for r in compression_rows if r["variant"] == "q4_k_m"), None)
    size_note = None
    if odg_size and q4_size:
        ratio = odg_size / q4_size
        if ratio > 1.25:
            size_note = (
                f"OpenDynamicGGUF is {ratio:.2f}× the size of Q4_K_M — a higher "
                "score at a much larger footprint is not an impressive quantization result."
            )
        elif 0.8 <= ratio <= 1.25:
            size_note = (
                f"OpenDynamicGGUF is {ratio:.2f}× the size of Q4_K_M (similar "
                "compression). Quality deltas at this size are the claim that matters."
            )

    return {
        "schema": SCHEMA,
        "model": cfg["model"],
        "created_at": _utc_now(),
        "pin_ok": pin_ok,
        "config_fingerprints": fingerprints,
        "variants_present": list(results),
        "quality": quality_rows,
        "compression": compression_rows,
        "behavior": behavior_rows,
        "inference": inference_rows,
        "n_params": manifest.get("n_params"),
        "size_note": size_note,
        "variant_labels": {
            vid: (
                (results.get(vid) or {}).get("label")
                or (manifest.get("reference_label") if vid == "bf16" else None)
                or VARIANT_LABELS[vid]
            )
            for vid in VARIANTS
        },
        "warnings": [
            "Do not blindly trust a single GGUF lm-eval number; see "
            "EleutherAI/lm-evaluation-harness#2887.",
            "FunctionGemma 270M is a function-calling specialist — absolute "
            "MMLU/GSM8K scores will be low. Compare retention vs the source, not SOTA.",
        ],
    }


def render_comparison_markdown(comp: dict[str, Any]) -> str:
    variants = [v for v in VARIANTS]
    labels = comp.get("variant_labels") or {v: VARIANT_LABELS[v] for v in variants}
    header = (
        "| Benchmark | "
        + " | ".join(labels.get(v, VARIANT_LABELS[v]) for v in variants)
        + " |"
    )
    sep = "| --- |" + " --- |" * len(variants)
    lines = [
        f"# OpenDynamicGGUF experiment — `{comp.get('model')}`",
        "",
        "Same model, same benchmark, same harness. Only quantization changes.",
        "",
        "## Quality",
        "",
        header,
        sep,
    ]
    for row in comp.get("quality") or []:
        cells = []
        best = row.get("best_quantized")
        for v in variants:
            raw = (row.get("scores") or {}).get(v)
            text = _fmt_score(raw)
            if v == best and raw is not None:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + row["label"] + " | " + " | ".join(cells) + " |")

    lines += ["", "## Compression", "", "| Model | Size | bytes/param | vs BF16 |", "| --- | ---: | ---: | ---: |"]
    for row in comp.get("compression") or []:
        bpw = row.get("bytes_per_parameter")
        ratio = row.get("compression_ratio_vs_bf16")
        lines.append(
            f"| {row['label']} | {_fmt_mb(row.get('bytes'))} | "
            f"{'—' if bpw is None else f'{bpw:.3f}'} | "
            f"{'—' if ratio is None else f'{ratio:.2f}×'} |"
        )

    if comp.get("size_note"):
        lines += ["", f"> {comp['size_note']}"]

    lines += [
        "",
        "## Model behavior",
        "",
        "| Model | Perplexity | KL vs BF16 | Token agreement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comp.get("behavior") or []:
        if row.get("skipped") and row.get("perplexity") is None:
            ppl = kl = agr = "—"
        else:
            ppl = "—" if row.get("perplexity") is None else f"{row['perplexity']:.3f}"
            kl = "—" if row.get("kl_divergence") is None else f"{row['kl_divergence']:.4f}"
            agr = "—" if row.get("token_agreement") is None else f"{row['token_agreement']:.3f}"
        lines.append(f"| {row['label']} | {ppl} | {kl} | {agr} |")

    lines += [
        "",
        "## Inference",
        "",
        "| Model | Prompt tok/s | Gen tok/s |",
        "| --- | ---: | ---: |",
    ]
    for row in comp.get("inference") or []:
        pp = "—" if row.get("pp_tps") is None else f"{row['pp_tps']:.1f}"
        tg = "—" if row.get("tg_tps") is None else f"{row['tg_tps']:.1f}"
        lines.append(f"| {row['label']} | {pp} | {tg} |")

    lines += ["", "## Warnings"]
    for w in comp.get("warnings") or []:
        lines.append(f"- {w}")
    if not comp.get("pin_ok"):
        lines.append(
            "- Config fingerprints differ across variants — results are **not** comparable."
        )
    lines.append("")
    return "\n".join(lines)


def compare(root: Path) -> dict[str, Any]:
    comp = build_comparison(root)
    md = render_comparison_markdown(comp)
    out = results_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "comparison.json", comp)
    (out / "comparison.md").write_text(md + "\n")
    return comp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_prepare(manifest: dict[str, Any]) -> None:
    print(f"model     : {manifest.get('model')}")
    print(f"n_params  : {manifest.get('n_params')}")
    print("sizes:")
    for vid, info in (manifest.get("sizes") or {}).items():
        status = "ok" if info.get("present") else "missing"
        mb = info.get("mb")
        mb_s = "—" if mb is None else f"{mb:.1f} MB"
        print(f"  {VARIANT_LABELS.get(vid, vid):<18} {status:<8} {mb_s}")
    for n in manifest.get("notes") or []:
        print(f"note: {n}")


def _print_run(payload: dict[str, Any]) -> None:
    print(f"variant   : {payload.get('label')} ({payload.get('variant')})")
    print(f"suite     : {payload.get('suite')}  limit={payload.get('limit')}")
    print(f"pin       : {payload.get('config_fingerprint')}")
    q = payload.get("quality") or {}
    if q.get("skipped"):
        print(f"quality   : skipped — {q.get('reason')}")
    else:
        for tid, entry in (q.get("tasks") or {}).items():
            print(f"  {TASK_LABELS.get(tid, tid):<12} {_fmt_score(entry.get('score'))}")
    for n in payload.get("notes") or []:
        print(f"note: {n}")
    if payload.get("result_path"):
        print(f"wrote     : {payload['result_path']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odg-experiment",
        description="Same-model / same-benchmark quantization experiment "
        "(FunctionGemma 270M by default).",
    )
    p.add_argument("action", choices=("prepare", "run", "compare"))
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Experiment directory (default: ./benchmark)",
    )
    p.add_argument("--model", "-m", default=None, help="HF repo id")
    p.add_argument(
        "--variant",
        "-v",
        default=None,
        help="bf16 | q4_k_m | q5_k_m | q6_k | odg",
    )
    p.add_argument("--all", action="store_true", help="Run every prepared variant")
    p.add_argument("--suite", default="dev", help="dev (default) or paper")
    p.add_argument("--limit", type=int, default=None, help="Override per-task sample cap")
    p.add_argument(
        "--gguf-backend",
        choices=("gguf", "hf"),
        default=None,
        help="gguf (llama-cpp-python, default) or hf (gguf_file=)",
    )
    p.add_argument("--skip-odg", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-behavior", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root) if args.root else default_root()
    root.mkdir(parents=True, exist_ok=True)

    if args.action == "prepare":
        try:
            manifest = prepare(
                root, model=args.model, force=args.force, skip_odg=args.skip_odg
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        _print_prepare(manifest)
        print(f"wrote     : {root / 'manifest.json'}")
        return 0

    if args.action == "run":
        variants: list[str]
        if args.all:
            variants = list(VARIANTS)
        elif args.variant:
            variants = [args.variant]
        else:
            print("ERROR: pass --variant V or --all", file=sys.stderr)
            return 1
        rc = 0
        for vid in variants:
            try:
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
            _print_run(payload)
            print()
        return rc

    if args.action == "compare":
        comp = compare(root)
        print(render_comparison_markdown(comp))
        print(f"wrote: {results_dir(root) / 'comparison.md'}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
