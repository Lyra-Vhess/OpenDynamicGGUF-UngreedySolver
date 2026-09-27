"""Step 12a — tail-KLD metric (Spec 2.1).

Definition (search split only):
  kld_mean     = mean(k) over N scored tokens
  kld_tail_1pct = mean of top ceil(0.01 * N) values of k

``kld_tail_1pct`` is the optimizer objective; ``kld_mean`` is retained for
backward compatibility / model-card comparison.

Per-token extraction mechanism (Checkpoint 0 finding):
  Upstream ``llama-perplexity --kl-divergence`` computes a per-token
  ``kld_values`` array internally (see perplexity.cpp) but prints only
  summary percentiles (max, 99.9%, 99%, median, ...). There is NO flag that
  dumps the full array. The minimal patch is a ``--kld-output <file>`` flag
  that writes the ``kld_values`` float array (one value per line) after the
  run. ``load_kld_array`` below accepts that dump (txt/npy/json) so the
  repo side needs no change once the binary patch lands.

CPU-only, numpy-only. No new dependencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: Provisional multiplier mapping proxy mean-KLD to tail-KLD when no
#: per-token array exists (proxy mode). Real llama probes must use
#: measured arrays via :func:`compute_kld_metrics`. Documented here so the
#: approximation is auditable, not silent.
PROXY_TAIL_MULTIPLIER = 8.0


def compute_kld_metrics(k: Any) -> dict[str, Any]:
    """Compute mean + tail-1% KLD from a per-token array.

    Uses ``numpy.partition`` (no full sort). Never mutates the input.

    N=0 choice: returns ``(nan, nan, n=0)`` rather than raising, so probe
    plumbing can record the failure explicitly. Documented per Spec 4(i).
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


def proxy_tail_from_mean(
    mean_kld: float, multiplier: float = PROXY_TAIL_MULTIPLIER
) -> float:
    """Provisional tail estimate for proxy-mode rows (no per-token data)."""
    return float(mean_kld) * float(multiplier)


def update_row_with_kld_array(row: dict[str, Any], k: Any) -> dict[str, Any]:
    """Attach measured KLD metrics to a sensitivity row in place."""
    m = compute_kld_metrics(k)
    row["kld_mean"] = m["kld_mean"]
    row["kld_tail_1pct"] = m["kld_tail_1pct"]
    row["n_tokens"] = m["n_tokens"]
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
