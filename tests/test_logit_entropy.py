"""kld.py: llama.cpp u16 dump parsing, entropy/KL math."""
import struct

import numpy as np
import pytest

from kld import block_stats, read_logit_dump

NV = 257


def _write_dump(path, logit_rows):
    """Write the llama.cpp layout: header, eval tokens, u16 records."""
    nt = logit_rows.shape[0]
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", 512, NV, 1))
        f.write(struct.pack("<%di" % 512, *([1] * 512)))
        for row in logit_rows:
            mx, mn = float(row.max()), float(row.min())
            mn = max(mn, mx - 16)
            scale = (mx - mn) / 65535.0
            f.write(struct.pack("<ff", scale, mn - mx))
            q = np.clip(np.round((row - mn) / scale), 0, 65535).astype("<u2")
            f.write(q.tobytes())  # stride is exactly NV+4 u16: no padding


def test_read_and_entropy_matches_analytic(tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(16, NV))
    p = tmp_path / "q.bin"
    _write_dump(p, logits)
    got = read_logit_dump(p)
    assert got["n_vocab"] == NV
    assert got["n_tokens"] == 16
    st = block_stats(np.ascontiguousarray(got["mat"]), None, block=7)
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    e /= e.sum(axis=1, keepdims=True)
    h = float(-(e * np.log(e)).sum() / 16)
    assert st["mean_entropy_nats"] == pytest.approx(h, rel=1e-4)
    assert st["mean_top1_mass"] == pytest.approx(
        float(e.max(axis=1).mean()), rel=1e-4)
    assert st["mean_entropy_bits"] == pytest.approx(h / np.log(2), rel=1e-4)


def test_kl_self_zero_and_sharpening(tmp_path):
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, size=(8, NV))
    _write_dump(tmp_path / "a.bin", logits)
    _write_dump(tmp_path / "b.bin", logits)
    a = np.ascontiguousarray(read_logit_dump(tmp_path / "a.bin")["mat"])
    b = np.ascontiguousarray(read_logit_dump(tmp_path / "b.bin")["mat"])
    st = block_stats(a, b)
    assert st["mean_kl_p_q"] == pytest.approx(0.0, abs=1e-9)
    _write_dump(tmp_path / "s.bin", logits * 2.0)
    s = np.ascontiguousarray(read_logit_dump(tmp_path / "s.bin")["mat"])
    st_s = block_stats(s, a)
    assert st_s["mean_entropy_nats"] < st["mean_entropy_nats"]
    assert st_s["mean_kl_p_q"] > 0


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "junk.bin"
    p.write_bytes(b"nope" * 10)
    with pytest.raises(ValueError, match="[Nn]ot a llama"):
        read_logit_dump(p)
