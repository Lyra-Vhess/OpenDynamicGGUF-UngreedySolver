"""Shannon entropy + KL from llama.cpp `--save-all-logits` dumps.

Dump layout (from llama.cpp `tools/perplexity/perplexity.cpp`):
  header: magic ``_logits_`` (8B), n_ctx u32, n_vocab i32, n_chunk i32;
  eval tokens: n_chunk × n_ctx int32 (skipped — model input, not output);
  per *scored* token (second half of each context window):
  ``[scale f32][min_log_prob f32][n_vocab uint16 log-probs]`` with stride
  ``nv = n_vocab + 4`` uint16s. Each entry stores
  ``round((logit - min_logit) / scale)`` for logits within 16 of the max,
  else 0; true log-prob reconstructs as ``u16 * scale + min_log_prob``
  (uint16 rounding over a ≤16-wide range — essentially lossless).

Usage:
  python3 logit_entropy.py --ref P-logits.bin --cand Q-logits.bin
      [--expected-kl 0.006412]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
