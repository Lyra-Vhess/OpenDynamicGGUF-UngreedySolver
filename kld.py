"""Step 12a — tail-KLD metric (Spec 2.1).

Definition (search split only):
  kld_mean      = "Mean KLD" line of a stock ``llama-perplexity
                  --kl-divergence`` run (mean over scored tokens)
  kld_tail_1pct = "99.0% KLD" line of the same run: the 99th percentile of
                  per-token KLD, i.e. the threshold above which the worst 1%
                  of tokens sit.

Notes on the definition change (no upstream C++ patch, per user decision):
  The spec's original tail-mean (mean of the top 1%) needs the full
  per-token array, which stock ``llama-perplexity`` does not dump. The P99
  threshold is measured by the stock binary, targets the same right tail,
  and is a conservative stand-in: the top-1% mean is always >= the 99th
  percentile, so driving down P99 drives down the tail. If a future binary
  (or ``--kld-output`` patch) provides per-token arrays, use
  :func:`compute_kld_metrics` for the exact tail-mean.

Hard rule: a missing "99.0% KLD" line is a hard error, never a silent
fallback. Proxy (feature-estimated) rows carry ``kld_tail_1pct = None``;
the DP tail objective refuses them loudly (see optimizer).

CPU-only, numpy-only for the array path; the log parser is stdlib-only.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

_FLOAT = r"([-\d.eE+]+)"

_PATTERNS: dict[str, str] = {
    "kld_mean": rf"Mean\s+KLD:\s*{_FLOAT}",
    "kld_tail_1pct": rf"99\.0%\s+KLD:\s*{_FLOAT}",
    "kld_p999": rf"99\.9%\s+KLD:\s*{_FLOAT}",
    "kld_max": rf"Maximum\s+KLD:\s*{_FLOAT}",
    "kld_median": rf"Median\s+KLD:\s*{_FLOAT}",
    "same_top_p": rf"Same top p:\s*{_FLOAT}",
    "perplexity": rf"Mean PPL\(Q\)\s*:\s*{_FLOAT}",
}

#: Lines required for a usable measurement. Absence raises (hard error).
_REQUIRED = ("kld_mean", "kld_tail_1pct")


def parse_llama_perplexity_kl(text: str) -> dict[str, Any]:
    """Parse the KL summary of a stock ``llama-perplexity`` log.

    Returns mean KLD, the P99 tail (``kld_tail_1pct``), plus best-effort
    extras (p999/max/median/same-top/perplexity, None when absent).
    ``n_tokens`` is None: the stock summary prints no token count.

    Raises:
        ValueError: if a required line (Mean KLD, 99.0% KLD) is missing.
    """
    out: dict[str, Any] = {}
    for key, pat in _PATTERNS.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    missing = [k for k in _REQUIRED if out[k] is None]
    if missing:
        raise ValueError(
            "llama-perplexity output is missing required KL lines "
            f"{missing} (need 'Mean KLD:' and '99.0% KLD:'). "
            "Was this run made with --kl-divergence --kl-divergence-base?"
        )
    out["n_tokens"] = None
    return out


def compute_kld_metrics(k: Any) -> dict[str, Any]:
    """Exact mean + top-1% mean from a per-token array (future/array path).

    Uses ``numpy.partition`` (no full sort). Never mutates the input.

    N=0 choice: returns ``(nan, nan, n=0)`` rather than raising, so probe
    plumbing can record the failure explicitly.
    """
    arr = np.asarray(k, dtype=np.float64)
    # np.asarray on a list always copies; on an ndarray it may share, but
    # np.partition below returns a new array and never writes into `arr`,
    # so the caller's buffer is untouched either way.
    n = int(arr.size)
    if n == 0:
        return {"kld_mean": float("nan"), "kld_tail_1pct": float("nan"), "n_tokens": 0}
    mean = float(np.mean(arr))
    k_cut = max(1, int(math.ceil(0.01 * n)))
    if k_cut >= n:
        tail = float(np.mean(arr))
    else:
        part = np.partition(arr, n - k_cut)
        tail = float(np.mean(part[n - k_cut :]))
    return {"kld_mean": mean, "kld_tail_1pct": tail, "n_tokens": n}


def update_row_with_kld_array(row: dict[str, Any], k: Any) -> dict[str, Any]:
    """Attach exact KLD metrics to a sensitivity row in place."""
    m = compute_kld_metrics(k)
    row["kld_mean"] = m["kld_mean"]
    row["kld_tail_1pct"] = m["kld_tail_1pct"]
    row["n_tokens"] = m["n_tokens"]
    return row


def update_row_with_measured_kl(
    row: dict[str, Any], measured: dict[str, Any]
) -> dict[str, Any]:
    """Attach a parsed stock-perplexity measurement to a row in place."""
    row["kld_mean"] = measured["kld_mean"]
    row["kld_tail_1pct"] = measured["kld_tail_1pct"]
    row["n_tokens"] = measured.get("n_tokens")
    for extra in ("kld_p999", "kld_max", "kld_median", "same_top_p", "perplexity"):
        if measured.get(extra) is not None:
            row[extra] = measured[extra]
    return row


def load_kld_array(path: str | Path) -> np.ndarray:
    """Load a per-token KLD dump: .npy, .json (list or {"kld": [...]}), or whitespace/comma txt."""
    p = Path(path).expanduser()
    if p.suffix == ".npy":
        return np.asarray(np.load(str(p)), dtype=np.float64).ravel()
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("kld", "kld_values", "values", "data"):
                if key in data:
                    data = data[key]
                    break
        return np.asarray(data, dtype=np.float64).ravel()
    text = p.read_text(encoding="utf-8").strip().replace(",", " ")
    if not text:
        return np.asarray([], dtype=np.float64)
    return np.asarray([float(t) for t in text.split()], dtype=np.float64)
