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

A farm is built once per model + imatrix by ``build_farm`` and carries a
``farm_manifest.json`` (model and imatrix hashes, shard counts, per-rung
byte sizes and type histograms). Builds are resumable: a rung whose shard
set already passes the census is kept, so an interrupted build continues
where it stopped.

Hard constraints (violations raise loudly, never silently degrade):
- Symlinks MUST preserve the source shard basenames (``q-%05d-of-%05d``):
  the GGUF split loader checks the internal split index against the
  filename.
- One tensor per shard: coarse multi-tensor shards over-quantize
  co-resident tensors and fail exactness, so the farm is per-tensor.
- The farm bakes in the model + imatrix + quantize build it was made
  from: rung shards carry the importance weights of the farm-build
  imatrix. ``SpliceFarm`` refuses a farm whose manifest records a
  different model or imatrix; a farm without a manifest is accepted only
  with a warning, after its shard structure is checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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

#: Build manifest filename inside the farm root (written by build_farm).
MANIFEST_FILENAME = "farm_manifest.json"

#: Manifest schema version.
MANIFEST_VERSION = 1

#: Directory holding the per-tensor reference split of the frozen model.
BASE_DIRNAME = "base720"

#: Directory holding the generated tensor-type files ("<rung>.tt").
TYPE_DIRNAME = "tensor_types"

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

    def __init__(
        self,
        root: str | Path,
        *,
        expected_model_sha256: str | None = None,
        expected_imatrix_sha256: str | None = None,
        require_manifest: bool = False,
        warn_legacy: bool = False,
    ) -> None:
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
        self.manifest = read_manifest(self.root)
        if self.manifest is not None:
            self._verify_manifest(
                expected_model_sha256=expected_model_sha256,
                expected_imatrix_sha256=expected_imatrix_sha256,
            )
        elif require_manifest:
            raise RuntimeError(
                f"Splice farm {self.root} has no {MANIFEST_FILENAME}, so it "
                "cannot be checked against this run's model and imatrix. "
                "Rebuild it by passing --splice-farm (the pipeline builds a "
                "farm automatically when the directory is missing), or point "
                "--splice-farm at a farm that has a manifest."
            )
        elif warn_legacy:
            print(
                f"WARNING: splice farm {self.root} has no {MANIFEST_FILENAME}. "
                "Cannot prove it was built from this run's model and imatrix; "
                "only the shard structure was checked.",
                file=sys.stderr,
            )
        self.tensor_to_idx: dict[str, int] = self._load_or_build_map()

    def _verify_manifest(
        self,
        *,
        expected_model_sha256: str | None,
        expected_imatrix_sha256: str | None,
    ) -> None:
        manifest = self.manifest or {}
        version = manifest.get("version")
        if version != MANIFEST_VERSION:
            raise RuntimeError(
                f"Farm manifest {self.root / MANIFEST_FILENAME} has version "
                f"{version!r}, expected {MANIFEST_VERSION}. Rebuild the farm."
            )
        if int(manifest.get("n_shards", -1)) != self.n_shards:
            raise RuntimeError(
                f"Farm manifest {self.root / MANIFEST_FILENAME} records "
                f"{manifest.get('n_shards')} shards, directory holds "
                f"{self.n_shards}. Rebuild the farm."
            )
        rungs = manifest.get("rungs")
        if not isinstance(rungs, dict) or set(rungs) != set(RUNG_SUBDIR):
            raise RuntimeError(
                f"Farm manifest {self.root / MANIFEST_FILENAME} does not "
                f"describe the full ladder (rungs: "
                f"{sorted(rungs) if isinstance(rungs, dict) else rungs!r})."
            )
        for rung, sub in RUNG_SUBDIR.items():
            recorded = (rungs.get(rung) or {}).get("dir")
            if recorded != sub:
                raise RuntimeError(
                    f"Farm manifest maps rung {rung} to {recorded!r}, "
                    f"expected {sub!r}. Refusing."
                )
        recorded_model = manifest.get("model") or {}
        if expected_model_sha256 is not None:
            recorded_sha = recorded_model.get("sha256")
            if recorded_sha is None:
                raise RuntimeError(
                    f"Farm manifest {self.root / MANIFEST_FILENAME} has no model "
                    "sha256, so it cannot be matched to this run's frozen model."
                )
            if recorded_sha != expected_model_sha256:
                raise RuntimeError(
                    f"Splice farm {self.root} was built from model sha256 "
                    f"{recorded_sha}, but this run's frozen model is "
                    f"{expected_model_sha256}. Rung shards carry the farm's "
                    "imatrix and quantize results, so the two cannot be mixed. "
                    "Point --splice-farm at a farm for this model."
                )
        recorded_imatrix = manifest.get("imatrix") or {}
        if expected_imatrix_sha256 is not None:
            recorded_sha = recorded_imatrix.get("sha256")
            if recorded_sha is None:
                raise RuntimeError(
                    f"Farm manifest {self.root / MANIFEST_FILENAME} has no imatrix "
                    "sha256, so it cannot be matched to this run's imatrix."
                )
            if recorded_sha != expected_imatrix_sha256:
                raise RuntimeError(
                    f"Splice farm {self.root} was built with imatrix sha256 "
                    f"{recorded_sha}, but this run uses "
                    f"{expected_imatrix_sha256}. The rung shards embed the "
                    "farm-build importance weights. Point --splice-farm at a "
                    "farm for this imatrix."
                )

    def _manifest_map(self) -> dict[str, int] | None:
        """Tensor→shard map from the manifest's recorded shard order."""
        if self.manifest is None:
            return None
        order = self.manifest.get("tensor_order")
        if not isinstance(order, list) or len(order) != self.n_shards:
            return None
        names = [str(name) for name in order]
        if len(set(names)) != len(names):
            raise RuntimeError(
                f"Farm manifest in {self.root} repeats tensor names; the "
                "recorded shard order is corrupt. Refusing."
            )
        return {name: idx for idx, name in enumerate(names, start=1)}

    def _load_or_build_map(self) -> dict[str, int]:
        from_manifest = self._manifest_map()
        cache = self.root / MAP_FILENAME
        saved: dict[str, int] | None = None
        if cache.is_file():
            try:
                raw = json.loads(cache.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raw = None
            if (
                isinstance(raw, dict)
                and raw.get("n_shards") == self.n_shards
                and isinstance(raw.get("map"), dict)
                and len(raw["map"]) == self.n_shards
            ):
                saved = {str(k): int(v) for k, v in raw["map"].items()}
        if from_manifest is not None:
            if saved is not None and saved != from_manifest:
                raise RuntimeError(
                    f"Farm map cache {cache} disagrees with {MANIFEST_FILENAME} "
                    "tensor order. One of them is stale; rebuild the farm."
                )
            return from_manifest
        if saved is not None:
            return saved
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


def load_farm(
    root: str | Path,
    *,
    expected_model_sha256: str | None = None,
    expected_imatrix_sha256: str | None = None,
    require_manifest: bool = False,
    warn_legacy: bool = False,
) -> SpliceFarm:
    """Validate a farm directory and return its handle (scans + caches map)."""
    return SpliceFarm(
        root,
        expected_model_sha256=expected_model_sha256,
        expected_imatrix_sha256=expected_imatrix_sha256,
        require_manifest=require_manifest,
        warn_legacy=warn_legacy,
    )


# --- farm construction -----------------------------------------------------


def _names_for(prefix: str, n_shards: int) -> list[str]:
    """Shard basenames for a split whose output prefix is ``prefix``."""
    return [
        f"{prefix}-{i:05d}-of-{n_shards:05d}.gguf"
        for i in range(1, n_shards + 1)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(binary: str | None) -> str | None:
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines():
        if line.strip():
            return line.strip()[:200]
    return None


def _type_file_line(rung: str) -> str:
    """Catch-all tensor-type rule: every quantizable tensor gets ``rung``."""
    return f".*={rung.lower()}\n"


def _split_command(split_bin: str, model: Path, out_prefix: Path) -> list[str]:
    """One-tensor-per-shard split. The prefix determines the shard names."""
    return [
        split_bin,
        "--split-max-tensors",
        "1",
        str(model),
        str(out_prefix),
    ]


def _quantize_command(
    quantize_bin: str,
    *,
    type_file: Path,
    imatrix: Path,
    first_shard: Path,
    out_file: Path,
) -> list[str]:
    """Quantize a whole shard set from its first shard, keeping the split.

    ``llama-quantize`` refuses any shard but the first. The trailing
    positional ``q8_0`` is a dummy default: the ``.*=<rung>`` catch-all in
    the tensor-type file overrides every tensor the tool will quantize.
    """
    return [
        quantize_bin,
        "--keep-split",
        "--tensor-type-file",
        str(type_file),
        "--imatrix",
        str(imatrix),
        str(first_shard),
        str(out_file),
        "q8_0",
    ]


def _set_complete(directory: Path, n_shards: int, prefix: str = "q") -> bool:
    if not directory.is_dir():
        return False
    names = sorted(p.name for p in directory.glob("*.gguf"))
    return names == _names_for(prefix, n_shards)


def _census_set(
    directory: Path, n_shards: int, *, what: str, prefix: str = "q"
) -> dict[str, Any]:
    """Check a shard set holds exactly one tensor per shard, in file order.

    Returns the tensor order, ggml-type histogram and total tensor bytes.
    Raises on any structural deviation.
    """
    from gguf_tensors import gguf_tensor_map

    names = sorted(p.name for p in directory.glob("*.gguf"))
    expected = _names_for(prefix, n_shards)
    if names != expected:
        raise RuntimeError(
            f"{what}: shard set in {directory} does not match "
            f"{expected[0]}..{expected[-1]} (found {len(names)} file(s), "
            f"first/last {names[:1]}/{names[-1:]}). Refusing."
        )
    order: list[str] = []
    histogram: dict[str, int] = {}
    total_bytes = 0
    for name in names:
        tensors = gguf_tensor_map(directory / name)["tensors"]
        if len(tensors) != 1:
            raise RuntimeError(
                f"{what}: shard {name} holds {len(tensors)} tensors, not 1. "
                "Coarse shards over-quantize co-resident tensors (see module "
                "docstring). Refusing."
            )
        tensor_name, meta = next(iter(tensors.items()))
        order.append(tensor_name)
        key = str(meta["ggml_type"])
        histogram[key] = histogram.get(key, 0) + 1
        total_bytes += int(meta["nbytes"])
    if len(set(order)) != len(order):
        raise RuntimeError(
            f"{what}: tensor names repeat across shards in {directory}. Refusing."
        )
    return {
        "tensors": order,
        "type_histogram": histogram,
        "nbytes": total_bytes,
    }


def read_manifest(root: str | Path) -> dict[str, Any] | None:
    """Read ``farm_manifest.json``; returns None when absent, raises when bad."""
    path = Path(root).expanduser() / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Farm manifest {path} is unreadable: {exc}") from None
    if not isinstance(data, dict):
        raise RuntimeError(f"Farm manifest {path} is not a JSON object")
    return data


def write_manifest(root: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write ``farm_manifest.json`` under ``root``."""
    path = Path(root).expanduser() / MANIFEST_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def farm_status(root: str | Path) -> dict[str, Any]:
    """Describe a farm directory without raising on an incomplete one.

    ``complete`` only means every rung directory exists with the same
    non-zero shard count. ``load_farm`` and ``build_farm`` still check the
    basenames, tensor order and manifest.
    """
    root_p = Path(root).expanduser()
    status: dict[str, Any] = {
        "root": str(root_p),
        "exists": root_p.is_dir(),
        "complete": False,
        "missing": [],
        "counts": {},
        "manifest": None,
    }
    if not status["exists"]:
        status["missing"] = sorted(set(RUNG_SUBDIR.values()))
        return status
    status["missing"] = sorted(
        d for d in set(RUNG_SUBDIR.values()) if not (root_p / d).is_dir()
    )
    status["counts"] = {
        d: len(list((root_p / d).glob("*.gguf")))
        for d in set(RUNG_SUBDIR.values())
        if d not in status["missing"]
    }
    counts = status["counts"]
    status["complete"] = (
        not status["missing"]
        and len(set(counts.values())) == 1
        and next(iter(counts.values()), 0) > 0
    )
    status["manifest"] = read_manifest(root_p)
    return status


def _identity_clashes(
    manifest: dict[str, Any],
    model: Path,
    model_sha256: str | None,
    imatrix: Path,
    imatrix_sha256: str | None,
) -> list[str]:
    clashes: list[str] = []
    recorded_model = manifest.get("model") or {}
    recorded_size = recorded_model.get("size")
    if recorded_size is not None and int(recorded_size) != model.stat().st_size:
        clashes.append(
            f"model size {model.stat().st_size} != recorded {recorded_size}"
        )
    if model_sha256 and recorded_model.get("sha256") not in (None, model_sha256):
        clashes.append("model sha256 differs")
    recorded_imatrix = manifest.get("imatrix") or {}
    recorded_imatrix_size = recorded_imatrix.get("size")
    if (
        recorded_imatrix_size is not None
        and int(recorded_imatrix_size) != imatrix.stat().st_size
    ):
        clashes.append(
            f"imatrix size {imatrix.stat().st_size} != recorded {recorded_imatrix_size}"
        )
    if imatrix_sha256 and recorded_imatrix.get("sha256") not in (
        None, imatrix_sha256
    ):
        clashes.append("imatrix sha256 differs")
    return clashes


def build_farm(
    model_gguf: str | Path,
    imatrix: str | Path,
    root: str | Path,
    *,
    rungs: Sequence[str] | None = None,
    llama_split: str | Path | None = None,
    llama_quantize: str | Path | None = None,
    force: bool = False,
    model_sha256: str | None = None,
    imatrix_sha256: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build (or resume) a per-tensor rung-shard farm and write its manifest.

    Recipe, all of it load-bearing:

    1. ``llama-gguf-split --split-max-tensors 1`` on the frozen model, so
       every shard holds one tensor.
    2. Per rung, ``llama-quantize --keep-split`` fed the *first* shard (the
       tool refuses any other) with a ``.*=<rung>`` tensor-type file and
       the run imatrix. ``--keep-split`` writes one shard per input shard,
       so the rung set keeps the one-tensor-per-shard shape.

    Existing complete rung sets are kept unless ``force``; an existing
    manifest for a different model or imatrix is refused, because rung
    shards are quantized with that imatrix and cannot be mixed.
    """
    from gguf_tensors import gguf_tensor_map
    from llama_bins import find_llama_binary

    emit = log or (lambda _message: None)
    model = Path(model_gguf).expanduser()
    imatrix_path = Path(imatrix).expanduser()
    root_path = Path(root).expanduser().resolve()

    if not model.is_file():
        raise RuntimeError(
            f"Splice farm needs the frozen model GGUF; not found: {model}"
        )
    if not imatrix_path.is_file():
        raise RuntimeError(
            f"Splice farm needs imatrix.gguf; not found: {imatrix_path}"
        )

    known_rungs = list(RUNG_SUBDIR)
    if rungs is None:
        selected = known_rungs
    else:
        selected = []
        for raw in rungs:
            rung = str(raw).upper()
            if rung not in RUNG_SUBDIR:
                raise RuntimeError(
                    f"Unknown farm rung {raw!r} (known: {sorted(RUNG_SUBDIR)})"
                )
            if rung not in selected:
                selected.append(rung)
        missing = [r for r in known_rungs if r not in selected]
        if missing:
            raise RuntimeError(
                "A splice farm must carry the full ladder; missing rung(s): "
                f"{missing}. SpliceFarm refuses partial farms, so build all "
                "eight or none."
            )

    existing = read_manifest(root_path)
    if existing is not None:
        clashes = _identity_clashes(
            existing, model, model_sha256, imatrix_path, imatrix_sha256
        )
        if clashes:
            raise RuntimeError(
                f"Splice farm {root_path} was built from different inputs: "
                + "; ".join(clashes)
                + ". Build the farm in a different directory. --force rebuilds "
                "rungs in place and will not mix models or imatrices."
            )

    n_tensors = len(gguf_tensor_map(model)["tensors"])
    root_path.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    previous_tools = (existing or {}).get("tools") or {}
    split_bin: str | None = previous_tools.get("llama_split")
    split_version: str | None = previous_tools.get("llama_split_version")
    base = root_path / BASE_DIRNAME
    if force or not _set_complete(base, n_tensors, prefix="t"):
        resolved = find_llama_binary("llama-gguf-split", llama_split)
        if resolved is None:
            raise RuntimeError(
                "llama-gguf-split not found (PATH, LLAMA_CPP_DIR, or "
                "llama_split=). A farm is built with the llama.cpp tools."
            )
        split_bin = str(resolved)
        split_version = _tool_version(split_bin)
        emit(
            f"splice farm: splitting {model.name} into {n_tensors} "
            "single-tensor shards"
        )
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        split_started = time.monotonic()
        _run(
            _split_command(split_bin, model, base / "t"),
            what="llama-gguf-split (per-tensor split)",
        )
        emit(f"splice farm: split done in {time.monotonic() - split_started:.1f}s")
        if not _set_complete(base, n_tensors, prefix="t"):
            found = len(list(base.glob("*.gguf")))
            raise RuntimeError(
                f"llama-gguf-split wrote {found} shard(s) under {base}, expected "
                f"{n_tensors} one-tensor shards. Refusing."
            )
    else:
        emit(f"splice farm: reusing per-tensor split in {base}")

    first_shard = base / _names_for("t", n_tensors)[0]
    type_dir = root_path / TYPE_DIRNAME
    type_dir.mkdir(parents=True, exist_ok=True)

    quantize_bin: str | None = previous_tools.get("llama_quantize")
    quantize_version: str | None = previous_tools.get("llama_quantize_version")
    rung_info: dict[str, Any] = {}
    tensor_order: list[str] | None = None
    for rung in selected:
        sub = RUNG_SUBDIR[rung]
        out_dir = root_path / sub
        reused = not force and _set_complete(out_dir, n_tensors)
        seconds = 0.0
        if not reused:
            if quantize_bin is None:
                resolved_quantize = find_llama_binary(
                    "llama-quantize", llama_quantize
                )
                if resolved_quantize is None:
                    raise RuntimeError(
                        "llama-quantize not found (PATH, LLAMA_CPP_DIR, or "
                        "llama_quantize=). A farm is built with the llama.cpp tools."
                    )
                quantize_bin = str(resolved_quantize)
                quantize_version = _tool_version(quantize_bin)
            type_file = type_dir / f"{rung.lower()}.tt"
            type_file.write_text(_type_file_line(rung), encoding="utf-8")
            emit(
                f"splice farm: quantizing rung {rung} into {sub}/ "
                f"({n_tensors} shards, catch-all .*={rung.lower()})"
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True)
            rung_started = time.monotonic()
            _run(
                _quantize_command(
                    quantize_bin,
                    type_file=type_file,
                    imatrix=imatrix_path,
                    first_shard=first_shard,
                    out_file=out_dir / "q.gguf",
                ),
                what=f"llama-quantize (rung {rung})",
            )
            seconds = time.monotonic() - rung_started
            emit(f"splice farm: rung {rung} done in {seconds:.1f}s")
        else:
            emit(f"splice farm: reusing rung {rung} in {sub}/")
        census = _census_set(out_dir, n_tensors, what=f"rung {rung}")
        if tensor_order is None:
            tensor_order = census["tensors"]
        elif census["tensors"] != tensor_order:
            raise RuntimeError(
                f"Splice farm rung {rung} tensor order differs from the first "
                "rung set. Sibling pairing would be wrong. Refusing."
            )
        rung_info[rung] = {
            "dir": sub,
            "n_shards": n_tensors,
            "n_tensors": len(census["tensors"]),
            "nbytes": census["nbytes"],
            "type_histogram": census["type_histogram"],
            "reused": reused,
            "seconds": round(seconds, 1),
        }

    if model_sha256 is None:
        emit("splice farm: hashing frozen model for the manifest")
        model_sha256 = sha256_file(model)
    if imatrix_sha256 is None:
        imatrix_sha256 = sha256_file(imatrix_path)

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": (existing or {}).get("created_at") or now_iso,
        "updated_at": now_iso,
        "model": {
            "path": str(model),
            "size": model.stat().st_size,
            "sha256": model_sha256,
            "n_tensors": n_tensors,
        },
        "imatrix": {
            "path": str(imatrix_path),
            "size": imatrix_path.stat().st_size,
            "sha256": imatrix_sha256,
        },
        "n_shards": n_tensors,
        "n_tensors": n_tensors,
        "base": {"dir": BASE_DIRNAME, "prefix": "t", "n_shards": n_tensors},
        "rungs": rung_info,
        "tensor_order": tensor_order,
        "tools": {
            "llama_split": split_bin,
            "llama_quantize": quantize_bin,
            "llama_split_version": split_version,
            "llama_quantize_version": quantize_version,
        },
        "seconds": (existing or {}).get("seconds")
        or round(time.monotonic() - started, 1),
        "last_run_seconds": round(time.monotonic() - started, 1),
    }
    write_manifest(root_path, manifest)
    total_bytes = sum(int(info["nbytes"]) for info in rung_info.values())
    emit(
        f"splice farm: ready at {root_path} "
        f"({len(selected)} rungs, {total_bytes / 1e9:.1f} GB of tensor data)"
    )
    return manifest


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
