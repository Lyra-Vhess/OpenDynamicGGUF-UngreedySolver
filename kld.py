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

Also hosts the ``--save-all-logits`` dump reader and entropy/KL diagnostic;
run it with ``python -m kld --ref ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
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


# --- llama.cpp --save-all-logits dumps (entropy / KL diagnostic) ---

MAGIC = b"_logits_"
HEADER_LEN = 20  # magic(8) + n_ctx(u32) + n_vocab(i32) + n_chunk(i32)


def read_logit_dump(path: str | Path) -> dict:
    """Parse header + memmap per-token (scale, min_log_prob, u16 vector)."""
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(HEADER_LEN)
    if len(header) < HEADER_LEN or header[:8] != MAGIC:
        raise ValueError(f"Not a llama.cpp logits dump: {path}")
    _nctx, n_vocab, n_chunk = struct.unpack("<iii", header[8:20])
    if n_vocab <= 0 or n_vocab > 10_000_000:
        raise ValueError(f"Implausible n_vocab={n_vocab} in {path}")
    n_vocab = int(n_vocab)
    stride = n_vocab + 4  # uint16s per record
    # Eval-token section is n_chunk * n_ctx int32 (model input, skipped).
    n_ctx = struct.unpack("<I", header[8:12])[0]
    base = HEADER_LEN + int(n_chunk) * int(n_ctx) * 4
    size = path.stat().st_size
    rec_bytes = stride * 2
    if (size - base) % rec_bytes:
        raise ValueError(
            f"{path}: trailing {size - base} bytes not a whole record "
            f"(stride {stride} u16). Wrong n_vocab?")
    n_tokens = (size - base) // rec_bytes
    if n_tokens <= 0:
        raise ValueError(f"No records in {path}")
    raw = np.memmap(path, dtype="<u2", mode="r")
    off = base // 2
    mat = np.lib.stride_tricks.as_strided(
        raw[off: off + n_tokens * stride],
        shape=(n_tokens, stride),
        strides=(stride * 2, 2),
    )
    return {"n_vocab": n_vocab, "n_tokens": int(n_tokens), "mat": mat,
            "_raw": raw}


def logprobs_block(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one block into (scales, min_log_probs, u16 vectors)."""
    blk = np.ascontiguousarray(mat)
    f32 = blk[:, :4].view("<f4").astype(np.float64)
    return f32[:, 0], f32[:, 1], blk[:, 4:].astype(np.float64)


def block_stats(
    qmat: np.ndarray, pmat: np.ndarray | None, block: int = 128,
) -> dict:
    """Mean entropy (nats/bits), top-1 mass, and KL(P||Q) over row blocks.

    Entropy is computed on the renormalized distribution (the ≤(max−16)
    tail floored to 0 on disk carries ~2.7% phantom mass; renormalizing
    removes it symmetrically for both dumps). The KL mirrors the stock
    tool exactly (perplexity.cpp log_softmax): NO renormalization on
    either side — expectation under raw decoded P over entries with
    P-logprob above −16, against raw decoded Q. Renormalizing Q first
    inflates the mean ~5x (0.033 vs logged 0.0064: a uniform +log(1.027)
    shift from the phantom floor mass).
    """
    n = qmat.shape[0]
    h_sum = t1_sum = kl_sum = 0.0
    for s in range(0, n, block):
        qs, qm, qv = logprobs_block(qmat[s: s + block])
        qlog = qv * qs[:, None] + qm[:, None]
        q = np.exp(qlog - qlog.max(axis=1, keepdims=True))
        q /= q.sum(axis=1, keepdims=True)
        h_sum += float(-(q * np.log(q + 1e-300)).sum())
        t1_sum += float(q.max(axis=1).sum())
        if pmat is not None:
            ps, pm, pv = logprobs_block(pmat[s: s + block])
            plog = pv * ps[:, None] + pm[:, None]
            keep = plog > -16.0
            p = np.exp(plog, out=plog)
            # Mirror the tool: raw decoded values both sides, no
            # renormalization (identical dumps give exactly 0).
            kl_sum += float(
                (p * (np.log(p + 1e-300) - qlog) * keep).sum())
    out = {
        "n_tokens": n,
        "mean_entropy_nats": h_sum / n,
        "mean_entropy_bits": h_sum / n / np.log(2.0),
        "mean_top1_mass": t1_sum / n,
    }
    if pmat is not None:
        out["mean_kl_p_q"] = kl_sum / n
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Shannon entropy + KL from llama.cpp --save-all-logits dumps")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--cand", type=Path, default=None)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--expected-kl", type=float, default=None)
    args = ap.parse_args(argv)

    ref = read_logit_dump(args.ref)
    n = ref["n_tokens"]
    pmat = ref["mat"][:n]
    out = {"P": block_stats(pmat, None, block=args.block)}
    if args.cand is not None:
        cand = read_logit_dump(args.cand)
        if cand["n_vocab"] != ref["n_vocab"]:
            raise ValueError("vocab mismatch between dumps")
        m = min(n, cand["n_tokens"])
        if m != n:
            print(f"NOTE: token counts differ; using first {m} rows")
        qmat = cand["mat"][:m]
        qst = block_stats(qmat, pmat[:m], block=args.block)
        qst["entropy_gap_bits_vs_P"] = (
            qst["mean_entropy_bits"] - out["P"]["mean_entropy_bits"])
        if args.expected_kl is not None:
            qst["kl_reproduces_logged"] = abs(
                qst["mean_kl_p_q"] - args.expected_kl)
        out["Q"] = qst
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
