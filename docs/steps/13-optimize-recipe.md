# Step 13 — Optimize the recipe under a size budget

← [12 Probe](./12-sensitivity-probe.md) · [Index](./README.md) · Next: [14 Export](./14-export-gguf.md) →

---

## Goal

Using the sensitivity table, assign a quantization type to every group so size stays under budget while maximizing bytes saved per unit ΔKLD.

Emit `recipe.yaml`, `recipe.tt`, and a **Pareto set** of alternatives at or below the budget — the budget is a hard limit, and Pareto targets above it are dropped (recorded in the manifest), not solved.

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
  pareto/*.yaml            # ratio-grid alternatives at or below budget
  pareto/frontier-bpw-*.yaml  # BPW selections: 3→8 bpw, round-UP rule
  optimize_manifest.json   # includes dropped_pareto_above_budget + frontier
  output.json
  status.json
  log.txt
```

## Predicted KLD, corrected

Summed DP predictions overcount the all-background penalty once per
group: each of the N summed cells carries the other N−1 groups'
background. The manifest reports both the raw sum
(`predicted_delta_kld`) and the corrected value
(`predicted_delta_kld_corrected` = raw − (N−1)·B, where B is the mean of
the probed background-rung cells). Correction is report-only — DP
decisions are unaffected (B cancels per cell in the argmin). When no
background cell was probed the corrected value is null.

## Full frontier + BPW selection

One extra solve at the adjusted budget backtracks **every** distinct
optimum at or below it (free from the DP tables) into
`manifest.frontier.table`. The BPW grid (3, 3.5, 4, 4.5, 5, 5.5, 6, 8,
plus the user budget as the 9th point = the primary recipe) rounds each
point UP to the smallest frontier allocation at/above its byte need and
writes it as `pareto/frontier-bpw-<bpw>.yaml`; duplicate hashes fold
onto one file, points above the frontier top stay unfilled. These are
the user-selectable recipe set step 15 measures (`--frontier`) and
step 14 exports (`--recipe`).
  log.txt
```

---

## Done when

- [x] ≥1 recipe written with per-group assignments
- [x] Traceable to sensitivity rows
- [x] Pareto alternatives saved (all at or below budget)

## Next

[Step 14 — Export the GGUF](./14-export-gguf.md)
