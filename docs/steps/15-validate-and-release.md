# Step 15 — Validate and release

← [14 Export](./14-export-gguf.md) · [Index](./README.md)

---

## Goal

Gate the candidate on held-out criteria, write a report, and stage a release — or return feedback to the optimizer.

---

## Command

```bash
odg validate --model functiongemma:latest
odg validate --model functiongemma:latest --strict   # FAIL if no real GGUF

# primary + full frontier sweep (9 recipes, Tier-1 each, plots):
odg validate --model functiongemma:latest --mode llama --frontier --force

# frontier set only: no primary, exports deferred (plots + JSON stay):
odg validate --model functiongemma:latest --mode llama \
  --budget-mb frontier --force
```

Requires Step 14 (except `--budget-mb frontier`, which needs only
steps 9–13 and exports transiently itself).

`--budget-mb frontier` requires the step-13 exhaustive certificate and
refuses otherwise: frontier optima are only meaningful over a fully
measured table.

The **Quantization Report Card** includes: architecture (layers/tensors), size vs the run's trial-background rung (never a hardcoded rung), compression by role, a per-layer matrix (attn_q/k/v/o + ffn_*), per-group Δbytes / ΔKLD, and a measured-vs-predicted quality section (held-out Tier-1 mean/P99/P999/max/top-1/PPL next to the raw and background-corrected summed predictions).

Verdicts:
- `RELEASE` — candidate GGUF present + tiers pass
- `PROVISIONAL` — dry-run export OK (plumbing); re-export with llama for release
- `FAIL` — gates failed; see `feedback.json`

---

## Outputs

```text
steps/15_validate/
  report.md / report.html
  quantization_report_card.html   # full per-layer / per-group card
  quantization_report_card.md
  quantization_report_card.json
  frontier/               # with --frontier / --budget-mb frontier:
    frontier.json         # per-point predicted + measured Tier-1
    pareto-mean.png / pareto-p99.png / pareto-top1.png / pareto-ppl.png
    exports/              # transient candidates (deleted after measuring)
  release/ or release_provisional/   # copies of the above
  feedback.json          # on FAIL
  output.json
```

The **Quantization Report Card** includes: architecture (layers/tensors), size vs baseline, compression by role, a per-layer matrix (attn_q/k/v/o + ffn_*), and per-group Δbytes / ΔKLD.

---

## Done when

- [x] Report written
- [x] Release staged **or** optimizer feedback produced

## Pipeline complete

Return to the [step index](./README.md).
