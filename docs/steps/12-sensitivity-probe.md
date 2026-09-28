# Step 12 — Sensitivity probing (trial quantize + measure ΔKLD)

← [11 Logits](./11-cache-reference-logits.md) · [Index](./README.md) · Next: [13 Optimize](./13-optimize-recipe.md) →

---

## Goal

For each **tensor group**, try quantization levels, record **ΔKLD** and **Δbytes**, and fill the sensitivity table the optimizer uses.

---

## Command

```bash
odg sensitivity --model functiongemma:latest
```

Requires Step 11. Uses **search** split only (never heldout).

Modes:
- `auto` / `proxy` — estimate ΔKLD from features + imatrix proxy; Δbytes from type sizes (plumbing)
- `llama` — real `llama-quantize` + `llama-perplexity --kl-divergence` (needs tools + logit caches), with **lazy probing**: the anchor plus each group's floor column is measured first, then column-generation pricing selects only attractive columns for GPU trials. The candidate universe is the full uniform ladder (F32 down to Q2_K) — wide costs nothing because pricing, not grid membership, spends GPU. `--certificate exhaustive` measures the whole universe instead. Reruns resume from trial files on disk.

Flags: `--probe-types` narrows the universe, `--kld-objective` / `--certificate` / `--lipschitz` steer pricing (same flags at step 13 select among the measured columns), `--fixed-groups` skips groups kept at source precision.

---

## Outputs

```text
steps/12_sensitivity/
  sensitivity.json     # full (group, probe) → metrics table
  output.json          # summary + top efficiency / pin hints
  status.json
  log.txt
```

Each row includes: `delta_bytes`, `delta_kld`, `efficiency` (= bytes/ΔKLD), `tensor_type_regex`, `decision_hint`.

---

## Done when

- [x] All quantizable groups have ≥1 probe row
- [x] Table persisted with input hashes
- [x] Search-only (documented)

## Next

[Step 13 — Optimize the recipe](./13-optimize-recipe.md)
