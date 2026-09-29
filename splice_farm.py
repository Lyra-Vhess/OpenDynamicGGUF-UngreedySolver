"""Splice-farm trial backend for step-12 GPU probing (llama mode).

A farm is a directory of precomputed per-tensor rung shards: one full
model split into single-tensor shards, quantized once per ladder rung
with ``llama-quantize --keep-split`` (catch-all ``.*=<RUNG>`` tensor-type
file). A probe trial is then assembled by symlinking one shard per
tensor — the group's shards from the probe rung's set, everything else
from the background set — and running stock ``llama-perplexity`` on the
assembly. No per-probe quantize, no bulk writes: per-probe cost is 720
symlinks plus one perplexity run.

Gate results (E4B farm at the default path): stock perplexity loads
shard-set assemblies with KL identical to full-file trials (gate-1),
and a 720-symlink assembly is byte-identical to the equivalent
pipeline trial on all 720 tensors with identical KL logs (gate-2).

Hard constraints (violations raise loudly, never silently degrade):
- Symlinks MUST preserve the source shard basenames (``q-%05d-of-%05d``):
  the GGUF split loader checks the internal split index against the
  filename.
- One tensor per shard: coarse multi-tensor shards over-quantize
  co-resident tensors and fail exactness, so the farm is per-tensor.
- The farm bakes in the model + imatrix + quantize build it was made
  from: rung shards carry the importance weights of the farm-build
  imatrix. A farm is valid only for runs of the same model with the
  same imatrix; the run manifest records the farm path for audit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kld import parse_llama_perplexity_kl

#: Probe/baseline rung → farm subdirectory. Q8_0 is the background set
#: (a catch-all Q8 quantize equals the positional q8_0 background
#: byte-wise, so no separate Q8 rung set is needed).
RUNG_SUBDIR = {
    "Q2_K": "rung720-q2_k",
    "Q3_K": "rung720-q3_k",
    "Q4_K": "rung720-q4_k",
    "Q5_K": "rung720-q5_k",
    "Q6_K": "rung720-q6_k",
    "Q8_0": "bg720-q8_0",
    "F16": "rung720-f16",
    "F32": "rung720-f32",
}

#: Cached tensor→shard map filename inside the farm root.
MAP_FILENAME = "tensor_shard_map.json"

#: Assembled-trial directory prefix inside the probe work dir.
ASSEMBLY_PREFIX = "farm-"


def _shard_name(idx: int, n_shards: int) -> str:
    return f"q-{idx:05d}-of-{n_shards:05d}.gguf"


def _run(cmd: list[str], *, what: str) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} failed (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n{log[-4000:]}"
        )
    return log


def _scan_shard_map(ref_dir: Path, n_shards: int) -> dict[str, int]:
    """Read one tensor name per shard header from a reference rung set.

    Each farm shard header holds exactly one tensor; the shard index ↔
    tensor correspondence (file order) is identical across all rung sets
    by construction. Returns {tensor_name: 1-based shard idx}.
    """
    from gguf_tensors import gguf_tensor_map

    mapping: dict[str, int] = {}
    for idx in range(1, n_shards + 1):
        shard = ref_dir / _shard_name(idx, n_shards)
        names = list(gguf_tensor_map(shard)["tensors"])
        if len(names) != 1:
            raise RuntimeError(
                f"Farm shard {shard} holds {len(names)} tensors, not 1 — "
                "not a per-tensor farm (coarse shards over-quantize "
                "co-residents; see module docstring). Refusing."
            )
        name = names[0]
        if name in mapping:
            raise RuntimeError(
                f"Farm tensor {name!r} appears in shards "
                f"{mapping[name]} and {idx} — shard map corrupt. Refusing."
            )
        mapping[name] = idx
    return mapping


class SpliceFarm:
    """A validated splice farm rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise RuntimeError(
                f"Splice farm not found: {self.root} (pass --splice-farm "
                "with the precomputed farm directory)."
            )
        missing = [d for d in set(RUNG_SUBDIR.values()) if not (self.root / d).is_dir()]
        if missing:
            raise RuntimeError(
                f"Splice farm {self.root} is missing rung sets: "
                f"{sorted(missing)}. A complete farm has one 720-shard set "
                f"per ladder rung."
            )
        counts = {
            d: len(list((self.root / d).glob("*.gguf")))
            for d in set(RUNG_SUBDIR.values())
        }
        n_shards = next(iter(counts.values()))
        uneven = {d: c for d, c in counts.items() if c != n_shards}
        if n_shards < 1 or uneven:
            raise RuntimeError(
                f"Splice farm {self.root} has uneven rung sets: {counts}. "
                "Every rung set must hold the same shards."
            )
        # Basenames must match across sets (the loader pairs siblings by
        # name); compare sorted listings, cheap.
        ref_names = sorted((self.root / "bg720-q8_0").glob("*.gguf"))
        ref_basenames = [p.name for p in ref_names]
        if ref_basenames != [
            _shard_name(i, n_shards) for i in range(1, n_shards + 1)
        ]:
            raise RuntimeError(
                f"Splice farm {self.root}/bg720-q8_0: shard basenames are "
                f"not q-%05d-of-%05d — the split loader requires that "
                "pattern. First/last: "
                f"{ref_basenames[:1]}/{ref_basenames[-1:]}. Refusing."
            )
        for d in set(RUNG_SUBDIR.values()):
            if d == "bg720-q8_0":
                continue
            other = sorted(p.name for p in (self.root / d).glob("*.gguf"))
            if other != ref_basenames:
                raise RuntimeError(
                    f"Splice farm rung set {d} basenames differ from "
                    "bg720-q8_0 — sibling pairing would break. Refusing."
                )
        self.n_shards = n_shards
        self.tensor_to_idx: dict[str, int] = self._load_or_build_map()

    def _load_or_build_map(self) -> dict[str, int]:
        cache = self.root / MAP_FILENAME
        if cache.is_file():
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                saved = None
            if (
                isinstance(saved, dict)
                and saved.get("n_shards") == self.n_shards
                and isinstance(saved.get("map"), dict)
                and len(saved["map"]) == self.n_shards
            ):
                return {str(k): int(v) for k, v in saved["map"].items()}
        mapping = _scan_shard_map(self.root / "bg720-q8_0", self.n_shards)
        cache.write_text(
            json.dumps(
                {"n_shards": self.n_shards, "map": mapping}, indent=0
            )
            + "\n",
            encoding="utf-8",
        )
        return mapping

    def subdir_for(self, rung: str) -> Path:
        """Farm subdirectory holding the full-model shard set at ``rung``."""
        try:
            sub = RUNG_SUBDIR[rung.upper()]
        except KeyError:
            raise RuntimeError(
                f"Splice farm has no rung set for probe type {rung!r} "
                f"(known: {sorted(RUNG_SUBDIR)}). Refusing."
            ) from None
        return self.root / sub

    def idx_for(self, tensor: str) -> int:
        try:
            return self.tensor_to_idx[tensor]
        except KeyError:
            raise RuntimeError(
                f"Group tensor {tensor!r} has no shard in farm {self.root} "
                "(farm was built from a different model/tensor set). "
                "Refusing — a partial assembly would silently mismeasure."
            ) from None


def load_farm(root: str | Path) -> SpliceFarm:
    """Validate a farm directory and return its handle (scans + caches map)."""
    return SpliceFarm(root)


def assemble_trial(
    farm: SpliceFarm,
    *,
    group_tensors: list[str],
    probe_type: str,
    baseline_type: str,
    dest_dir: str | Path,
    tag: str,
) -> Path:
    """Symlink one shard per tensor into ``dest_dir/farm-{tag}/``.

    The group's tensors come from the probe rung's set, everything else
    from the background set. Basenames are preserved verbatim (loader
    requirement). Returns the assembly's first-shard path — the
    ``llama-perplexity -m`` entry point.
    """
    group_set = set(group_tensors)
    unknown = sorted(group_set - set(farm.tensor_to_idx))
    if unknown:
        raise RuntimeError(
            f"Trial farm-{tag}: {len(unknown)} group tensor(s) have no "
            f"farm shard (e.g. {unknown[0]!r}). Farm/model mismatch — "
            "refusing instead of mismeasuring."
        )
    probe_dir = farm.subdir_for(probe_type)
    bg_dir = farm.subdir_for(baseline_type)
    group_idx = {farm.tensor_to_idx[t] for t in group_set}
    dest = Path(dest_dir) / f"{ASSEMBLY_PREFIX}{tag}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    try:
        for idx in range(1, farm.n_shards + 1):
            name = _shard_name(idx, farm.n_shards)
            src = (probe_dir if idx in group_idx else bg_dir) / name
            os.symlink(os.path.realpath(src), dest / name)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    first = dest / _shard_name(1, farm.n_shards)
    if not first.is_file():
        raise RuntimeError(
            f"Trial farm-{tag}: assembly entry shard missing: {first}."
        )
    return first


def measure_farm_column(
    *,
    farm: SpliceFarm,
    probe_type: str,
    baseline_type: str,
    search_txt: str | Path,
    kl_base_bin: str | Path,
    work_dir: str | Path,
    tag: str,
    llama_perplexity: str | Path | None = None,
    perplexity_args: list[str] | None = None,
    keep_assembly: bool = False,
    group_tensors: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a farm trial and measure its KL vs the reference base.

    Same return shape as ``llama_probe.measure_column`` (absolute,
    non-delta metrics plus ``group_bytes_measured`` /
    ``probe_exempt_tensors`` from the probe rung shards' own headers,
    exact block arithmetic like the pipeline path). The assembly
    directory holds symlinks only and is removed after measuring unless
    ``keep_assembly``. Raises loudly on any tool failure or missing KL
    line. The source anchor still comes from ``measure_source_anchor``
    (the frozen file against itself) — the farm only replaces trial
    builds, never the zero point.
    """
    from gguf_tensors import gguf_tensor_map
    from llama_probe import assert_probe_applied
    from logits import find_llama_perplexity as find_ppl

    pbin = find_ppl(llama_perplexity)
    if pbin is None:
        raise RuntimeError(
            "llama-perplexity not found (PATH, LLAMA_CPP_DIR, or --llama-perplexity)."
        )
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    first = assemble_trial(
        farm,
        group_tensors=list(group_tensors or []),
        probe_type=probe_type,
        baseline_type=baseline_type,
        dest_dir=work,
        tag=tag,
    )

    pcmd = [
        str(pbin), "-m", str(first), "-f", str(search_txt),
        "--kl-divergence", "--kl-divergence-base", str(kl_base_bin),
    ]
    if perplexity_args:
        pcmd += list(perplexity_args)
    plog = _run(pcmd, what=f"llama-perplexity farm probe {tag}")
    (work / f"farm-{tag}.perplexity.log").write_text(plog, encoding="utf-8")

    measured = parse_llama_perplexity_kl(plog)  # hard error if lines missing
    if group_tensors:
        probe_dir = farm.subdir_for(probe_type)
        tmap: dict[str, Any] = {}
        for name in group_tensors:
            shard = probe_dir / _shard_name(farm.idx_for(name), farm.n_shards)
            tmap[name] = gguf_tensor_map(shard)["tensors"].get(name) or {}
        missing = [n for n in group_tensors if not tmap.get(n)]
        if missing:
            raise RuntimeError(
                f"Trial farm-{tag}: {len(missing)} group tensor(s) missing "
                f"from probe rung headers ({missing[0]!r}...). Farm corrupt?"
            )
        measured["probe_exempt_tensors"] = assert_probe_applied(
            tmap, group_tensors, probe_type, baseline_type, tag=f"farm-{tag}",
        )
        measured["group_bytes_measured"] = int(
            sum(int(tmap[n]["nbytes"]) for n in group_tensors)
        )
    if not keep_assembly:
        shutil.rmtree(first.parent, ignore_errors=True)
    measured["trial_tag"] = tag
    return measured
