"""Step 12b — measured per-group probes (llama mode).

For one ``(group, quant_type)`` column: quantize a trial GGUF where every
tensor uses the baseline type except this group's tensors (pinned to the
probe type via a tensor-type file), then run stock ``llama-perplexity
--kl-divergence`` against the step-11 KL base on the search split and parse
the Mean / 99.0% KLD lines.

Each probe is two subprocesses; callers parallelize with a thread pool
(``--jobs``). No CUDA code, no new dependencies.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from kld import parse_llama_perplexity_kl


def _run(cmd: list[str], *, what: str) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} failed (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n{log[-4000:]}"
        )
    return log


def measure_column(
    *,
    model_gguf: str | Path,
    group_regex: str | None,
    probe_type: str,
    baseline_type: str,
    search_txt: str | Path,
    kl_base_bin: str | Path,
    work_dir: str | Path,
    tag: str,
    llama_quantize: str | Path | None = None,
    llama_perplexity: str | Path | None = None,
    imatrix: str | Path | None = None,
    perplexity_args: list[str] | None = None,
    keep_trial: bool = False,
) -> dict[str, Any]:
    """Quantize one trial and measure its KL vs the reference base.

    ``group_regex=None`` measures the all-baseline config (the run's anchor
    for deltas). Returns absolute (non-delta) metrics: ``kld_mean``,
    ``kld_tail_1pct`` (P99), plus ``same_top_p``/``perplexity`` when the
    log has them. Raises loudly on any tool failure or missing KL line.
    """
    from llama_bins import find_llama_binary
    from logits import find_llama_perplexity as find_ppl

    qbin = find_llama_binary("llama-quantize", llama_quantize)
    if qbin is None:
        raise RuntimeError(
            "llama-quantize not found (PATH, LLAMA_CPP_DIR, or --llama-quantize)."
        )
    pbin = find_ppl(llama_perplexity)
    if pbin is None:
        raise RuntimeError(
            "llama-perplexity not found (PATH, LLAMA_CPP_DIR, or --llama-perplexity)."
        )
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    trial = work / f"trial-{tag}.gguf"

    qcmd = [str(qbin)]
    tt_path = None
    if group_regex is not None:
        tt_path = work / f"trial-{tag}.tt"
        # No comment lines: some llama.cpp builds reject '#' in type files.
        tt_path.write_text(
            f"{group_regex}={probe_type.lower()}\n",
            encoding="utf-8",
        )
        qcmd += ["--tensor-type-file", str(tt_path)]
    if imatrix is not None and Path(imatrix).is_file():
        qcmd += ["--imatrix", str(imatrix)]
    qcmd += [str(model_gguf), str(trial), baseline_type.lower()]
    qlog = _run(qcmd, what=f"llama-quantize trial {tag}")
    if not trial.is_file():
        raise RuntimeError(f"llama-quantize exited 0 but trial missing: {trial}")
    (work / f"trial-{tag}.quantize.log").write_text(qlog, encoding="utf-8")

    pcmd = [
        str(pbin), "-m", str(trial), "-f", str(search_txt),
        "--kl-divergence", "--kl-divergence-base", str(kl_base_bin),
    ]
    if perplexity_args:
        pcmd += list(perplexity_args)
    plog = _run(pcmd, what=f"llama-perplexity probe {tag}")
    (work / f"trial-{tag}.perplexity.log").write_text(plog, encoding="utf-8")

    measured = parse_llama_perplexity_kl(plog)  # hard error if lines missing
    if not keep_trial:
        trial.unlink(missing_ok=True)
    measured["trial_tag"] = tag
    return measured
