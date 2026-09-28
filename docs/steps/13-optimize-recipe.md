# Step 13 — Optimize the recipe under a size budget

← [12 Probe](./12-sensitivity-probe.md) · [Index](./README.md) · Next: [14 Export](./14-export-gguf.md) →

---

## Goal

Using the sensitivity table, assign a quantization type to every group so size stays under budget while maximizing bytes saved per unit ΔKLD.

Emit `recipe.yaml`, `recipe.tt`, and a **Pareto set** of alternatives.

---

## Command

```bash
odg optimize --model functiongemma:latest
odg optimize --model functiongemma:latest --budget-mb 180 --force
```

Requires Step 12.

| Flag | Meaning |
|---|---|
| `--budget-mb` | Absolute target size (MiB) |
| `--budget-ratio` | Default `0.72` of all-Q6_K estimate when mb omitted |
| `--optimizer` | `dp_mckp` (default, exact) or `greedy` (A/B baseline) |
| `--kld-objective` | `mean` (default, + auto P99 guardrail) or `tail_1pct` |
| `--certificate` | `bounded` (default) or `exhaustive` |
| `--fixed-groups` | Group ids kept at source precision |

---

## Algorithm

DP-MCKP (default): exact multiple-choice knapsack over the uniform
candidate ladder (F32 down to Q2_K for every group — no role floors; only
measured `pin_high` hints floor at Q5_K), with the P99 guardrail and a
termination certificate. See README Stage 7.

If the primary allocation rests on proxy-estimated (never measured) KLD
columns — e.g. solving at a budget looser than the step-12 pricing
reference covered — the manifest records them under
`primary.proxy_kld_columns` and the notes carry a WARNING naming each one.
(The guardrail path hard-errors on such picks first when it is on; the
note covers the guardrail-opted-out remainder.)

---

## Outputs

```text
steps/13_optimize/
  recipe.yaml              # primary odg/recipe/v1
  recipe.tt                # llama-quantize --tensor-type-file
  pareto/*.yaml            # frontier alternatives
  optimize_manifest.json
  output.json
  status.json
  log.txt
```

---

## Done when

- [x] ≥1 recipe written with per-group assignments
- [x] Traceable to sensitivity rows
- [x] Pareto alternatives saved

## Next

[Step 14 — Export the GGUF](./14-export-gguf.md)
