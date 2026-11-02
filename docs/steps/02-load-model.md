# Step 02 — Load the model

← [01 Resolve](./01-resolve-model.md) · [Index](./README.md) · Next: [03 Enumerate](./03-enumerate-tensors.md) →

---

## Goal

Open the source from Step 01 and make its tensors **addressable** for later inspection (enumerate / features).

---

## Why it exists

Step 01 only finds the file. Step 02 proves we can open it and builds the tensor index.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | `09_freeze_gguf/output.json` → `gguf_path`; fall back to `01_resolve/output.json` → `local_path` |
| **Output** | `02_load/output.json` + `tensor_index.json` |

Inside `odg run`, step 09 runs before this step. Load indexes the frozen text
GGUF, so the catalog and every later feature describe the 720 text tensors
rather than a multimodal HF checkpoint's audio/vision towers. A standalone
load on a run without step 09 still uses the resolve output.

---

## Two backends

### A) Frozen GGUF (default after step 09)

```text
local_path = ~/.ollama/models/blobs/sha256-…
     │
     ▼
Parse GGUF header
     │
     ▼
Tensor index (name, shape, dtype, offset)
```

Does **not** expand Q8 weights into BF16 RAM.

### B) HF safetensors (fallback without a frozen GGUF)

```text
local_path = HF snapshot dir
     │
     ▼
safetensors mmap index + config.json
```

---

## Command

```bash
odg load --model functiongemma:latest
odg status --model functiongemma:latest
```

Requires Step 01 done for that run. If Step 09 is done, load uses the frozen
GGUF and records its path and SHA in `input.json`.

---

## Done when

- [x] Source file/dir opens
- [x] `n_tensors` known
- [x] Checkpoint written under `steps/02_load/`

## Implementation

[02-load-model-impl.md](./02-load-model-impl.md)

## Next

[Step 03 — Enumerate every tensor](./03-enumerate-tensors.md)
