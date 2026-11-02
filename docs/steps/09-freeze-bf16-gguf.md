# Step 09 — Freeze the GGUF reference

← [01 Resolve](./01-resolve-model.md) · [Index](./README.md) · Next: [02 Load](./02-load-model.md) →

---

## Goal

Produce one **frozen, hashed GGUF** that all llama.cpp probes, imatrix, and exports use.

Ideal: HF BF16 → `model-bf16.gguf`.  
Pragmatic (current Ollama path): promote the working GGUF → `model-ref.gguf` + SHA.

---

## Command

```bash
odg freeze-gguf --model functiongemma:latest
```

Requires Step 01 done. In `odg run` this step now runs second, before load:
steps 02–08 index the frozen text GGUF. Run it after `odg resolve` when using
the step-by-step commands.

For an HF directory, `auto` converts to BF16. Add `--require-bf16` to fail if
the source was a quantized GGUF that had to be promoted:

```bash
odg resolve --model functiongemma:latest --prefer-hf --download-weights --force
odg freeze-gguf --model functiongemma:latest --require-bf16 --force
```

A local `LLAMA_CPP_DIR`/`~/llama.cpp` converter is preferred; otherwise the
latest stable converter is fetched into `~/.cache/odg/llama.cpp/<sha>/`.

---

## Modes

| Mode | Behavior |
|---|---|
| `auto` | HF convert if the resolved source is an HF dir; else promote the source GGUF |
| `hf-convert` | Fail unless `convert_hf_to_gguf.py` + HF dir work |
| `promote` | Always hardlink/symlink/copy the resolved GGUF |

`hf-convert` uses `LLAMA_CPP_DIR` / `~/llama.cpp` when present. Otherwise it fetches the converter for the
latest stable llama.cpp release into `~/.cache/odg/llama.cpp/<sha>/` (checked at most once a day, works
offline once cached).

---

## Outputs

```text
steps/09_freeze_gguf/
  model-bf16.gguf          # or model-ref.gguf if not BF16
  model-*.gguf.sha256
  freeze_manifest.json
  output.json
  status.json
  log.txt
```

Also verifies catalog tensor names ⊆ GGUF tensor index when a catalog exists.

---

## Done when

- [x] Frozen GGUF exists under the step dir with SHA-256
- [x] Catalog names checked against the file when a catalog exists
- [x] Manifest records whether it is a true BF16 reference

## Next

[Step 10 — Build the importance matrix](./10-build-imatrix.md)
